# src/evaluate_graz.py
import os, json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from torch.utils.data import DataLoader
from dataset_graz import GrazDataset
from model_graz import SwinFractureNet
from config import TEST_CSV, MODELS_DIR, RESULTS_DIR, IMG_SIZE, BATCH_SIZE, DEVICE, BACKBONE

device = torch.device(DEVICE)
print("Device:", device)

# load label maps
label_map_path = os.path.join(MODELS_DIR, "label_maps.json")
if not os.path.exists(label_map_path):
    raise FileNotFoundError("label_maps.json not found. Train first.")
with open(label_map_path, "r") as f:
    maps = json.load(f)
type_list = maps.get("type_list", ["Simple"])
sev_list = maps.get("sev_list", ["Mild"])

# dataset
val_ds = GrazDataset(TEST_CSV, type_to_idx={t:i for i,t in enumerate(type_list)}, sev_to_idx={s:i for i,s in enumerate(sev_list)},
                     transform=None)
# default transform inside dataset if None
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# load model
ckpt_path = os.path.join(MODELS_DIR, "swin_graz_best.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError("Model checkpoint not found. Train first.")
ckpt = torch.load(ckpt_path, map_location=device)
model = SwinFractureNet(num_type_classes=len(type_list), num_severity_classes=len(sev_list), backbone=BACKBONE)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

all_true = []
all_pred = []
all_probs = []

with torch.no_grad():
    for imgs, frac_labels, _, _, paths in val_loader:
        imgs = imgs.to(device)
        out_frac, _, _ = model(imgs)
        probs = torch.softmax(out_frac, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_probs.extend(probs.tolist())
        all_pred.extend(preds.tolist())
        all_true.extend(frac_labels.numpy().tolist())

# metrics
report = classification_report(all_true, all_pred, target_names=["Normal", "Fracture"])
cm = confusion_matrix(all_true, all_pred)
# mAP
y_true = np.array(all_true)
y_scores = np.array(all_probs)
one_hot = np.zeros_like(y_scores)
one_hot[np.arange(len(y_true)), y_true] = 1
mAP = average_precision_score(one_hot, y_scores, average="macro")

print("Classification report:\n", report)
print("Confusion matrix:\n", cm)
print("mAP:", mAP)

# save results
os.makedirs(RESULTS_DIR, exist_ok=True)
out = {
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "mAP": float(mAP)
}
with open(os.path.join(RESULTS_DIR, "graz_evaluation_summary.json"), "w") as f:
    json.dump(out, f, indent=2)

# per-image CSV
df = pd.DataFrame({
    "true": all_true,
    "pred": all_pred,
})
df.to_csv(os.path.join(RESULTS_DIR, "graz_eval_per_image.csv"), index=False)
print("Saved results to", RESULTS_DIR)
