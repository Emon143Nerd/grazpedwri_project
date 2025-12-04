# src/dataset_graz.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from config import IMG_SIZE

class GrazDataset(Dataset):
    def __init__(self, csv_file, type_to_idx=None, sev_to_idx=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform or T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor()
        ])
        # build maps if not given
        if type_to_idx is None:
            unique_types = sorted(self.df["type"].dropna().unique().tolist())
            self.type_to_idx = {t: i for i, t in enumerate(unique_types)}
        else:
            self.type_to_idx = type_to_idx
        if sev_to_idx is None:
            unique_sev = sorted(self.df["severity"].dropna().unique().tolist())
            self.sev_to_idx = {s: i for i, s in enumerate(unique_sev)}
        else:
            self.sev_to_idx = sev_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        frac = int(row["fracture"])
        t = row.get("type", "Simple")
        s = row.get("severity", "Mild")
        type_idx = self.type_to_idx.get(t, 0)
        sev_idx = self.sev_to_idx.get(s, 0)
        return img, frac, type_idx, sev_idx, img_path
