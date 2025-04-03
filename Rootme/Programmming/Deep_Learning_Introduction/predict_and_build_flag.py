import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import re
import joblib

# === Config
INPUT_FOLDER = "flag"
OUTPUT_IMAGE = "flag.png"
LABELS_FILE = "supervision.txt"
IMG_SIZE = (224, 224)
NUM_SUPERVISED = 100

# === Charger modèle MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
model = base_model

def extract_position(filename):
    match = re.match(r"(\d+)_(\d+)\.jpg", filename)
    return (int(match.group(1)), int(match.group(2))) if match else None

filenames = sorted(
    [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.jpg')],
    key=lambda x: extract_position(x)
)

# === Lire labels depuis labels.txt
if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError(f"{LABELS_FILE} introuvable. Crée un fichier contenant 100 lignes (0 ou 1).")

with open(LABELS_FILE, "r") as f:
    lines = f.readlines()

labels = [int(line.strip()) for line in lines if line.strip() in ['0', '1']]

if len(labels) < NUM_SUPERVISED:
    raise ValueError(f"{LABELS_FILE} contient seulement {len(labels)} labels. Il en faut {NUM_SUPERVISED}.")

# === Extraction des features supervisées
print(f"🔧 Chargement des {NUM_SUPERVISED} images supervisées...")

features_list = []
for i, fname in enumerate(filenames[:NUM_SUPERVISED]):
    img_path = os.path.join(INPUT_FOLDER, fname)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    features_list.append(features[0])

features_array = np.array(features_list)
labels_array = np.array(labels[:NUM_SUPERVISED])

# === Entraînement du classifieur
print("\n📈 Entraînement du modèle supervisé...")
clf = LogisticRegression()
clf.fit(features_array, labels_array)
joblib.dump(clf, "logistic_model.pkl")

# === Génération du drapeau
positions = []
max_row, max_col = 0, 0

for fname in filenames:
    pos = extract_position(fname)
    if not pos:
        continue
    row, col = pos
    positions.append((fname, row, col))
    max_row = max(max_row, row)
    max_col = max(max_col, col)

flag_array = np.zeros((max_row + 1, max_col + 1), dtype=np.uint8)

print("\n📷 Prédiction automatique avec le modèle entraîné...")
for fname, row, col in tqdm(positions):
    img_path = os.path.join(INPUT_FOLDER, fname)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    prediction = clf.predict(features)[0]
    flag_array[row, col] = 255 if prediction == 1 else 0

# === Sauvegarde finale
flag_image = Image.fromarray(flag_array, mode='L')
flag_image.save(OUTPUT_IMAGE)
print(f"\n🏁 Drapeau généré avec succès → {OUTPUT_IMAGE}")