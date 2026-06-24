import pandas as pd
import numpy as np
import os
import joblib
import torch
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from collections import Counter

# -----------------------------
# تنظیمات
# -----------------------------
DATA_PATH = "JEOPARDY_CSV.csv"
SAMPLE_SIZE = None

MODEL_NAME = "all-mpnet-base-v2"

UMAP_PATH = "umap_model.pkl"
HDBSCAN_PATH = "hdbscan_model.pkl"
LABEL_PATH = "labels.npy"

# -----------------------------
# GPU Detection
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

# batch adaptive
BATCH_SIZE = 256 if DEVICE == "cuda" else 64

# -----------------------------
# Load Data
# -----------------------------
print("Loading data...")
data = pd.read_csv(DATA_PATH)
questions = data["Question"].astype(str)

if SAMPLE_SIZE:
    questions = questions.sample(n=SAMPLE_SIZE, random_state=42)

texts = questions.tolist()

# -----------------------------
# Embedding Model
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

print("Encoding texts...")

with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
    print("Encoding started...")
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    print("Encoding finished!")

emb = emb.astype(np.float32)

# normalize (خیلی مهم برای UMAP)
emb = normalize(emb)

# -----------------------------
# Load or Train
# -----------------------------

print("CWD:", os.getcwd())
print("Absolute UMAP path:", os.path.abspath(UMAP_PATH))
print("Absolute HDBSCAN path:", os.path.abspath(HDBSCAN_PATH))
print("Absolute LABEL path:", os.path.abspath(LABEL_PATH))
print("Files in cwd:", os.listdir(os.getcwd()))



if os.path.exists(UMAP_PATH) and os.path.exists(HDBSCAN_PATH) and os.path.exists(LABEL_PATH):
    print("Loading saved models...")
    umap_model = joblib.load(UMAP_PATH)
    clusterer = joblib.load(HDBSCAN_PATH)
    labels = np.load(LABEL_PATH)
    umap_emb = umap_model.transform(emb)

else:
    print("Training UMAP...")

    umap_model = umap.UMAP(
        n_neighbors=30, n_components=10, min_dist=0.0, metric="euclidean", low_memory=True, random_state=42
    )

    umap_emb = umap_model.fit_transform(emb)

    print("Training HDBSCAN...")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5, metric="euclidean", core_dist_n_jobs=-1)

    labels = clusterer.fit_predict(umap_emb)

    joblib.dump(umap_model, UMAP_PATH)
    joblib.dump(clusterer, HDBSCAN_PATH)
    np.save(LABEL_PATH, labels)

# -----------------------------
# Evaluation
# -----------------------------
mask = labels != -1

if mask.sum() > 1 and len(set(labels[mask])) > 1:
    score = silhouette_score(umap_emb[mask], labels[mask])
    print("Silhouette score:", score)
else:
    print("Not enough clusters for silhouette.")

# -----------------------------
# Cluster Keywords
# -----------------------------
print("Extracting cluster keywords...")

df = pd.DataFrame({"text": texts, "cluster": labels})
df = df[df["cluster"] != -1]

clusters = sorted(df["cluster"].unique())
cluster_top_words = {}

for c in clusters:
    cluster_texts = df[df["cluster"] == c]["text"]
    words = " ".join(cluster_texts).lower().split()
    counter = Counter(words)
    cluster_top_words[c] = [f"{w}({cnt})" for w, cnt in counter.most_common(20)]

final_matrix = pd.DataFrame({f"cluster_{c}": cluster_top_words[c] for c in clusters})

final_matrix.to_csv("cluster_top_words.csv", index=False)

print("Done ")
