# src/build_graz_csv.py
import os
import pandas as pd
from config import DATA_ROOT, IMAGE_PARTS_TRAIN, IMAGE_PART_TEST, ORIG_CSV, TRAIN_CSV, TEST_CSV, EVAL_CSV

# Mapping rules:
# AO codes: '/1.' -> mild/simple, '/2.' -> moderate/simple, '/3.' -> comminuted/complex
def map_ao_to_type_and_severity(ao_code, fracture_visible):
    # default
    if fracture_visible == 0:
        return "Normal", "None"
    if pd.isna(ao_code) or str(ao_code).strip() == "":
        return "Simple", "Mild"
    s = str(ao_code)
    if "/3." in s:
        return "Comminuted", "Severe"
    if "/2." in s:
        return "Simple", "Moderate"
    if "/1." in s:
        return "Simple", "Mild"
    # fallback
    return "Simple", "Moderate"

def find_image_path(root_parts, filestem):
    fname = filestem + ".png"
    for p in root_parts:
        candidate = os.path.join(p, fname)
        if os.path.exists(candidate):
            return candidate
    return None

def main():
    print("Reading original CSV:", ORIG_CSV)
    df = pd.read_csv(ORIG_CSV)
    # ensure columns exist
    if "filestem" not in df.columns or "fracture_visible" not in df.columns:
        raise RuntimeError("dataset.csv must contain 'filestem' and 'fracture_visible' columns.")
    rows_train = []
    rows_test = []
    rows_all = []

    # iterate and assign to train/test depending on which part contains the image
    for _, r in df.iterrows():
        filestem = str(r["filestem"]).strip()
        fracture = int(r["fracture_visible"]) if not pd.isna(r["fracture_visible"]) else 0
        ao = r.get("ao_classification", "") if "ao_classification" in r.index else ""
        # find path in train parts first, then test
        p_train = find_image_path(IMAGE_PARTS_TRAIN, filestem)
        p_test = find_image_path([IMAGE_PART_TEST], filestem)
        path = p_train or p_test
        if not path:
            # skip missing images but log
            # print("Missing image for", filestem)
            continue
        # map AO -> type, severity
        ftype, severity = map_ao_to_type_and_severity(ao, fracture)
        # append row (path, fracture, type, severity)
        row = {"path": path, "fracture": fracture, "type": ftype, "severity": severity, "filestem": filestem}
        rows_all.append(row)
        # if path in part4 -> test, else train
        if path.startswith(os.path.abspath(IMAGE_PART_TEST)):
            rows_test.append(row)
        else:
            rows_train.append(row)

    # save CSVs
    pd.DataFrame(rows_all).to_csv(EVAL_CSV, index=False)
    pd.DataFrame(rows_train).to_csv(TRAIN_CSV, index=False)
    pd.DataFrame(rows_test).to_csv(TEST_CSV, index=False)

    print(f"Saved: {TRAIN_CSV} ({len(rows_train)} rows)")
    print(f"Saved: {TEST_CSV} ({len(rows_test)} rows)")
    print(f"Saved: {EVAL_CSV} ({len(rows_all)} rows)")

if __name__ == "__main__":
    main()
