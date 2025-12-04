# src/inference_graz.py
import os, json
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from config import EVAL_CSV, MODELS_DIR, RESULTS_DIR, IMG_SIZE, DEVICE, BACKBONE
from model_graz import SwinFractureNet
from dataset_graz import GrazDataset
from gradcam_heatmap import generate_gradcam
import torchvision.transforms as transforms

device = torch.device(DEVICE)

# load label maps
label_map_path = os.path.join(MODELS_DIR, "label_maps.json")
with open(label_map_path, "r") as f:
    maps = json.load(f)
type_list = maps.get("type_list", ["Simple"])
sev_list = maps.get("sev_list", ["Mild"])

# load model
ckpt = torch.load(os.path.join(MODELS_DIR, "swin_graz_best.pth"), map_location=device)
model = SwinFractureNet(num_type_classes=len(type_list), num_severity_classes=len(sev_list), backbone=BACKBONE)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

os.makedirs(RESULTS_DIR, exist_ok=True)

def predict_and_save(image_path, out_dir=RESULTS_DIR):
    img = Image.open(image_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out_frac, out_type, out_sev = model(inp)
        frac_probs = torch.softmax(out_frac, dim=1).cpu().numpy().tolist()[0]
        type_probs = torch.softmax(out_type, dim=1).cpu().numpy().tolist()[0]
        sev_probs = torch.softmax(out_sev, dim=1).cpu().numpy().tolist()[0]
        frac_pred = int(np.argmax(frac_probs))
        type_pred = int(np.argmax(type_probs))
        sev_pred = int(np.argmax(sev_probs))
    # heatmap
    mask_uint8, overlay_bgr = generate_gradcam(model, inp, target_cat=frac_pred, use_cuda=(device.type=="cuda"))
    base = os.path.splitext(os.path.basename(image_path))[0]
    heat_path = os.path.join(out_dir, base + "_heat.png")
    cv2.imwrite(heat_path, overlay_bgr)
    # json
    out = {
        "image": image_path,
        "fracture_pred": int(frac_pred),
        "fracture_conf": float(max(frac_probs)),
        "fracture_probs": frac_probs,
        "type_pred": type_list[type_pred] if type_list else str(type_pred),
        "type_conf": float(type_probs[type_pred]),
        "severity_pred": sev_list[sev_pred] if sev_list else str(sev_pred),
        "severity_conf": float(sev_probs[sev_pred]),
        "heatmap": heat_path
    }
    json_path = os.path.join(out_dir, base + "_result.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved:", heat_path, json_path)

if __name__ == "__main__":
    # run on eval csv (careful: large)
    df = pd.read_csv(EVAL_CSV)
    paths = df["path"].tolist()
    for p in tqdm(paths):
        predict_and_save(p)
