import os
import re
import time
from pathlib import Path

import requests
import torch
from PIL import Image
import pandas as pd
import numpy as np
import open_clip
from chromadb import PersistentClient
from langchain_community.docstore.document import Document


# =======================
# Paths + cache (cluster-safe)
# =======================
BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = (BASE_DIR.parent / "reports" / "cv10_splits").resolve()
IMAGES_DIR = BASE_DIR / "images"
CHROMA_DIR = BASE_DIR / "chroma_db_combined"

# Cache for model/weight downloads (useful on HPC)
HF_HOME = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
TORCH_HOME = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch")))
HF_HOME.mkdir(parents=True, exist_ok=True)
TORCH_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / (n + 1e-12)


# ====== CLEANING FOR TEXT CONTENT ======
def clean_label(text):
    """
    Removes URLs in parentheses and normalizes whitespace as in the evaluation.
    Example:
    'plastic (http://...) , paper (http://...)' → 'plastic, paper'
    """
    if not isinstance(text, str):
        return ""

    cleaned = re.sub(r"\s*\(https?://[^\)]*\)", "", text)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned.strip())
    return cleaned.strip()


# === Download images from multiple URLs ===
def download_images(image_field, save_dir: Path = IMAGES_DIR, timeout_s: int = 20, retries: int = 2):
    save_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    image_urls = []

    if not isinstance(image_field, str) or image_field.strip() == "":
        return [], []

    urls = [u.strip() for u in image_field.replace(",", ";").split(";") if u.strip()]

    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    for url in urls:
        # local filename: URL basename (without the query string)
        basename = os.path.basename(url.split("?")[0])
        if not basename:
            # fallback for unusual URLs
            basename = f"img_{abs(hash(url))}.jpg"

        filename = save_dir / basename

        # if already downloaded, do not download again
        if filename.exists() and filename.stat().st_size > 0:
            image_paths.append(str(filename))
            image_urls.append(url)
            continue

        ok = False
        for attempt in range(retries + 1):
            try:
                resp = session.get(url, timeout=timeout_s, headers=headers)
                if resp.status_code == 200 and resp.content:
                    filename.write_bytes(resp.content)
                    image_paths.append(str(filename))
                    image_urls.append(url)
                    ok = True
                    break
                else:
                    print(f"[download] status={resp.status_code} url={url}")
            except Exception as e:
                print(f"[download] error (attempt {attempt+1}/{retries+1}) url={url}: {e}")
                time.sleep(1.0)

        if not ok:
            # do not add anything: image download failed
            pass

    return image_paths, image_urls


# === Load CSV as a list of Documents (one per image) ===
def load_csv_as_documents(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")

    # Read Picture as a string and disable automatic NA parsing
    df = pd.read_csv(csv_path, dtype={'Picture': str}, keep_default_na=False, na_filter=False)

    documents = []
    skipped_rows = []

    for idx, row in df.iterrows():
        contained_waste = clean_label(str(row.get("ContainedWaste", "")))
        size = str(row.get("Size", ""))
        raw_image_field = row.get("Picture", None)

        # Minimal diagnostics: show how the field appears
        if raw_image_field is None:
            skipped_rows.append((idx, "None"))
            continue

        # Normalize: strip whitespace and check for obvious missing values
        image_field = str(raw_image_field).strip()
        if image_field.lower() in ("", "nan", "none", "null"):
            skipped_rows.append((idx, f"empty_like:{repr(image_field)}"))
            continue

        # Check URL/format before splitting
        from urllib.parse import urlparse
        def looks_like_url(s):
            try:
                s = s.strip()
                p = urlparse(s)
                return p.scheme in ('http', 'https') and p.netloc != ''
            except Exception:
                return False

        # Split on the separators in use (handles both "a, b" and "a; b")
        urls = [u.strip() for u in image_field.replace(";", ",").split(",") if u.strip()]

        # Keep only URLs that look valid
        urls = [u for u in urls if looks_like_url(u)]

        if not urls:
            skipped_rows.append((idx, f"no_valid_urls_after_split:{repr(image_field)}"))
            continue

        image_paths, image_urls = download_images("; ".join(urls))

        if not image_paths:
            skipped_rows.append((idx, f"download_failed:{urls[:3]}"))
            continue

        for img_path, img_url in zip(image_paths, image_urls):
            content = (
                f"ContainedWaste: {contained_waste}\n"
                f"Size: {size}\n"
                f"Picture: {img_path}"
            )

            metadata = {
                "row_index": int(idx),
                "filename": csv_path.name,
                "image_path": img_path,
                "image_url": img_url
            }

            documents.append(Document(page_content=content, metadata=metadata))

    if skipped_rows:
        print("Skipped rows (idx, reason) — first 20:", skipped_rows[:20])

    return documents


# === MULTIMODAL EMBEDDING (text + image) ===
class UnifiedCLIPEmbedding:
    def __init__(self):
        print("Loading CLIP model (ViT-L-14 openai) for text and images...")
        self.device = os.environ.get("FORCE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
        emb = emb[0].detach().cpu().numpy().astype(np.float32)
        return _safe_normalize(emb)

    def embed_image(self, img_path: str) -> np.ndarray:
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(image_tensor)
        emb = emb[0].detach().cpu().numpy().astype(np.float32)
        return _safe_normalize(emb)

    def embed_document(self, document: Document) -> np.ndarray:
        text_emb = self.embed_text(document.page_content)

        img_path = document.metadata.get("image_path")
        img_emb = np.zeros_like(text_emb, dtype=np.float32)

        if img_path and os.path.exists(img_path):
            try:
                img_emb = self.embed_image(img_path)
            except Exception as e:
                print(f"Error embedding image {img_path}: {e}")

        # concatenation: [text_emb | img_emb]
        return np.concatenate([text_emb, img_emb]).astype(np.float32)


# === CREATE CHROMA ===
def create_vectorstore(documents, embedding_model: UnifiedCLIPEmbedding):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = PersistentClient(path=str(CHROMA_DIR))
    collection_name = "waste_combined"

    # compatibility: list_collections may return objects or dicts depending on the version
    existing = []
    for c in client.list_collections():
        name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
        if name:
            existing.append(name)

    if collection_name in existing:
        print(f"Collection '{collection_name}' exists → deleting...")
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    for i, doc in enumerate(documents):
        emb = embedding_model.embed_document(doc)

        # Chroma requires JSON-serializable metadata → convert everything to strings
        clean_metadata = {k: str(v) for k, v in doc.metadata.items()}

        collection.add(
            ids=[str(i)],
            embeddings=[emb.tolist()],
            metadatas=[clean_metadata],
            documents=[doc.page_content]
        )

        if (i + 1) % 200 == 0:
            print(f"  added {i+1}/{len(documents)} documents...")

    print(f"Database saved in {CHROMA_DIR}")
    return collection


# === MAIN ===
if __name__ == "__main__":
    csv_path = REPORTS_DIR / "split_01_train.csv"

    print(f"CSV: {csv_path}")
    print("Loading CSV and downloading images...")
    documents = load_csv_as_documents(csv_path)
    print(f"Total images loaded: {len(documents)}")

    print("Initializing unified CLIP embedding...")
    embedding_model = UnifiedCLIPEmbedding()

    print("Creating vector store...")
    _ = create_vectorstore(documents, embedding_model)

    print("✅ Updated multimodal database created successfully.")
