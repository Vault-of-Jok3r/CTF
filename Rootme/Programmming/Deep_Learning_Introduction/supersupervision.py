import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import re
import joblib

# === Configurations
INPUT_FOLDER = "flag"
OUTPUT_IMAGE = "flag.png"
LABELS_FILE = "supervision.txt"
IMG_SIZE = (224, 224)
NUM_SUPERVISED = 100       # Les 100 premières images sont annotées depuis supervision.txt
NUM_EPOCHS = 5             # Nombre d'époques d'entraînement
UNCERTAINTY_MARGIN = 0.1   # Marge autour de 0.5 pour considérer qu'une prédiction est incertaine

# === Charger MobileNetV2 comme extracteur de features
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
model = base_model

def extract_position(filename):
    match = re.match(r"(\d+)_(\d+)\.jpg", filename)
    return (int(match.group(1)), int(match.group(2))) if match else None

# Récupérer et trier les images selon leur position
filenames = sorted(
    [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.jpg')],
    key=lambda x: extract_position(x)
)

# === Charger les labels supervisés depuis supervision.txt
if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError(f"{LABELS_FILE} introuvable. Créez un fichier contenant {NUM_SUPERVISED} lignes (0 ou 1).")

with open(LABELS_FILE, "r") as f:
    lines = f.readlines()

labels = [int(line.strip()) for line in lines if line.strip() in ['0', '1']]
if len(labels) < NUM_SUPERVISED:
    raise ValueError(f"{LABELS_FILE} contient seulement {len(labels)} labels. Il en faut {NUM_SUPERVISED}.")

# === Extraction des features pour les images supervisées (les 100 premières)
print(f"🔧 Chargement des {NUM_SUPERVISED} images supervisées...")
X_supervised = []
for fname in filenames[:NUM_SUPERVISED]:
    img_path = os.path.join(INPUT_FOLDER, fname)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    X_supervised.append(features[0])
X_supervised = np.array(X_supervised)
y_supervised = np.array(labels[:NUM_SUPERVISED])

# === Initialisation de l'apprentissage actif
active_features = []  # Pour stocker les features des images annotées via l'apprentissage actif
active_labels = []    # Pour stocker les labels correspondants
active_labeled_files = set()  # Pour éviter de redemander sur une même image

# === Boucle sur les époques
for epoch in range(NUM_EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")

    # Constitution du jeu d'entraînement combiné : données supervisées + données issues de l'apprentissage actif
    if len(active_features) > 0:
        X_train = np.concatenate([X_supervised, np.array(active_features)], axis=0)
        y_train = np.concatenate([y_supervised, np.array(active_labels)], axis=0)
    else:
        X_train = X_supervised
        y_train = y_supervised

    # Entraînement du classifieur
    print("📈 Entraînement du modèle supervisé...")
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    joblib.dump(clf, "logistic_model.pkl")
    print("✅ Modèle sauvegardé dans logistic_model.pkl")

    # Apprentissage actif : parcourir les images non supervisées et vérifier l'incertitude de la prédiction
    print("📷 Analyse des images pour détection d'incertitude...")
    # On itère sur toutes les images, en ignorant celles déjà annotées ou dans les 100 premières
    for fname in tqdm(filenames):
        if fname in filenames[:NUM_SUPERVISED]:
            continue
        if fname in active_labeled_files:
            continue

        img_path = os.path.join(INPUT_FOLDER, fname)
        img = image.load_img(img_path, target_size=IMG_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x, verbose=0)
        # Obtenir la probabilité pour la classe 1
        prob = clf.predict_proba(features)[0, 1]
        # Si la probabilité est trop proche de 0.5, on considère que le modèle est incertain
        if abs(prob - 0.5) < UNCERTAINTY_MARGIN:
            pos = extract_position(fname)
            pos_str = f"({pos[0]}, {pos[1]})" if pos else ""
            print(f"\nImage {fname} à la position {pos_str} a une probabilité de {prob:.3f}.")
            user_input = input("Veuillez saisir le label (0 ou 1) pour renforcer le modèle (appuyez sur Entrée pour ignorer) : ")
            if user_input.strip() in ['0', '1']:
                label = int(user_input.strip())
                active_features.append(features[0])
                active_labels.append(label)
                active_labeled_files.add(fname)
                print(f"Label {label} enregistré pour {fname}.")
            else:
                print("Aucune annotation enregistrée pour cette image.")

# === Génération finale du drapeau avec le modèle le plus récent
print("\n🏁 Génération finale du drapeau avec le modèle entraîné...")
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

for fname, row, col in tqdm(positions):
    img_path = os.path.join(INPUT_FOLDER, fname)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    prediction = clf.predict(features)[0]
    flag_array[row, col] = 255 if prediction == 1 else 0

flag_image = Image.fromarray(flag_array, mode='L')
flag_image.save(OUTPUT_IMAGE)
print(f"\n🏁 Drapeau généré avec succès → {OUTPUT_IMAGE}")
