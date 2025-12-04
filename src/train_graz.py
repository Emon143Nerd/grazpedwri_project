# src/train_graz.py
import os, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from config import TRAIN_CSV, TEST_CSV, MODELS_DIR, IMG_SIZE, BATCH_SIZE, LR, EPOCHS, DEVICE, NUM_WORKERS, BACKBONE
from dataset_graz import GrazDataset
from model_graz import SwinFractureNet

device = torch.device(DEVICE)
print("Device:", device)

# Read train/test CSVs
import pandas as pd
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

# Build label maps for type and severity based on df_train
type_list = sorted(df_train["type"].dropna().unique().tolist())
sev_list = sorted(df_train["severity"].dropna().unique().tolist())

type_to_idx = {t:i for i,t in enumerate(type_list)}
sev_to_idx = {s:i for i,s in enumerate(sev_list)}

# transforms
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_ds = GrazDataset(TRAIN_CSV, type_to_idx=type_to_idx, sev_to_idx=sev_to_idx, transform=train_tf)
val_ds = GrazDataset(TEST_CSV, type_to_idx=type_to_idx, sev_to_idx=sev_to_idx, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print("Train samples:", len(train_ds), "Val samples:", len(val_ds))
print("Type classes:", type_list)
print("Severity classes:", sev_list)

# model
model = SwinFractureNet(num_type_classes=len(type_list), num_severity_classes=len(sev_list), backbone=BACKBONE).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

best_f1 = 0.0

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
    for imgs, frac_labels, type_labels, sev_labels, _ in pbar:
        imgs = imgs.to(device)
        frac_labels = frac_labels.to(device)
        type_labels = type_labels.to(device)
        sev_labels = sev_labels.to(device)

        optimizer.zero_grad()
        out_frac, out_type, out_sev = model(imgs)
        loss_frac = criterion(out_frac, frac_labels)
        loss_type = criterion(out_type, type_labels)
        loss_sev = criterion(out_sev, sev_labels)
        # weight fracture higher
        loss = loss_frac + 0.5 * loss_type + 0.5 * loss_sev
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=running_loss/len(train_loader))

    # validation
    model.eval()
    all_true = []
    all_preds = []
    with torch.no_grad():
        for imgs, frac_labels, *_ in tqdm(val_loader, desc="Validating"):
            imgs = imgs.to(device)
            frac_labels = frac_labels.to(device)
            out_frac, _, _ = model(imgs)
            preds = torch.argmax(out_frac, dim=1)
            all_true.extend(frac_labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds)
    print(f"Epoch {epoch} Val Acc: {acc:.4f} F1: {f1:.4f}")

    scheduler.step(f1)

    # checkpoint
    if f1 > best_f1:
        best_f1 = f1
        ckpt = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'type_list': type_list, 'sev_list': sev_list}
        torch.save(ckpt, os.path.join(MODELS_DIR, "swin_graz_best.pth"))
        with open(os.path.join(MODELS_DIR, "label_maps.json"), "w") as f:
            json.dump({'type_list': type_list, 'sev_list': sev_list}, f, indent=2)
        print("Saved best model with F1:", best_f1)

print("Training finished. Best F1:", best_f1)
