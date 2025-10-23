from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from data.data_loading import ImageNetDataset
from modeling import LookWhereDownstream
from sklearn.metrics import accuracy_score
import math
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from utils import get_learning_rates, adjust_learning_rate
import timm

def train(gpu_id, model, learning_rate, epochs, log_file):
    train_dataset = ImageNetDataset(
        dataset=load_dataset("imagenet-1k", split="train"),
        do_augment="3aug",
        img_size=224,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        drop_last=False,
    )

    val_dataset = ImageNetDataset(
        dataset=load_dataset("imagenet-1k", split="validation"),
        do_augment=False,
        img_size=224,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    total_steps = int(epochs * len(train_loader))
    warmup_steps = math.ceil(total_steps * 0.1)
    sched_config = {
        "max_lr": learning_rate,
        "min_lr": 5.0e-6,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01)
    ce_loss = CrossEntropyLoss()
    step_counter = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_images, batch_labels in tqdm(train_loader):
            batch_images = batch_images.to(gpu_id)
            batch_labels = batch_labels.to(gpu_id)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch_images)
                loss = ce_loss(logits, batch_labels)

            loss.backward()
            losses.append(loss.detach().cpu().item())

            new_lr = adjust_learning_rate(step_counter, sched_config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            
            current_lr = list(get_learning_rates(optimizer=optimizer).values())[0]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step_counter += 1
        
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(gpu_id)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch_images)
                
                predictions += logits.argmax(dim=-1).cpu().tolist()
                labels += batch_labels.tolist()

        accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        losses = torch.Tensor(losses).mean().item()
        print(epoch, accuracy, losses, current_lr)

        with open(log_file, "a") as f:
            print(f"epoch={epoch}, val_acc={accuracy}, train_loss={losses}, current_lr={current_lr}", file=f)
    
    torch.save(
        obj=model.state_dict(),
        f=log_file.replace(".txt", ".pt")
    )


if __name__ == "__main__":
    # user settings
    gpu_id = 0
    is_lookwhere = True
    learning_rate = 1.0e-5
    k = 128  # only used if is_lookwhere == True
    num_classes = 1_000
    high_res_img_size = 224
    epochs = 30

    if is_lookwhere == True:
        model = LookWhereDownstream(
            pretrained_params_path="lookwhere_dinov2.pt",
            high_res_size=high_res_img_size,
            num_classes=num_classes,
            k=k,
            is_cls=True,
            device=gpu_id
        )
        log_file = f"is_LW={is_lookwhere}_K={k}_LR={learning_rate}_E={epochs}.txt"
    else:
        class DINOv2(torch.nn.Module):
            def __init__(self, num_classes, img_size):
                super().__init__()
                self.backbone = timm.create_model(
                        "vit_base_patch14_reg4_dinov2.lvd142m",
                        num_classes=0,
                        patch_size=14,
                        pretrained=True,
                        img_size=img_size
                )
                self.linear = torch.nn.Linear(self.backbone.embed_dim, num_classes)

            def forward(self, x):
                x = self.backbone.forward_features(x)
                return self.linear(x[:, 0, :])
        
        model = DINOv2(num_classes, high_res_img_size).to(gpu_id)
        log_file = f"is_LW={is_lookwhere}_LR={learning_rate}_E={epochs}.txt"

    train(gpu_id, model, learning_rate, epochs, log_file)


