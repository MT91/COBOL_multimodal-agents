import os
import json
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import requests
from PIL import Image
import open_clip
from chromadb import PersistentClient
import ollama


# =======================
# Paths + cache (cluster-safe)
# =======================
BASE_DIR = Path(__file__).resolve().parent            # .../cobol/agents
PROJECT_DIR = BASE_DIR.parent                         # .../cobol (project root)

# input CSV (as in your script)
REPORTS_DIR = PROJECT_DIR / "reports" / "cv10_splits"

QUERY_IMAGES_DIR = PROJECT_DIR / "query_images"
CHROMA_DIR = PROJECT_DIR / "chroma_db_combined"

OUT_DIR = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

HF_HOME = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
TORCH_HOME = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch")))
HF_HOME.mkdir(parents=True, exist_ok=True)
TORCH_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / (n + 1e-12)


# =========================================================
# ID helpers (adjust if your CSV has different columns)
# =========================================================
def get_row_report_id(row: pd.Series) -> str:
    """
    Extract a stable ID from the CSV if present; otherwise, fall back to the row index.
    """
    for k in ["ReportId", "report_id", "Id", "ID", "reportId", "source_id", "uuid", "report_uuid"]:
        if k in row and pd.notna(row[k]):
            s = str(row[k]).strip()
            if s:
                return s
    return f"row_{int(row.name)}"


def get_meta_report_id(md: dict) -> str:
    """
    Extract a stable ID from the Chroma metadata, if present. Useful for debugging and context.
    """
    if not md:
        return ""
    for k in ["ReportId", "report_id", "source_id", "id", "reportId", "uuid", "report_uuid"]:
        v = md.get(k)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return ""


# =========================================================
# 1) Download images (cache + retry)
# =========================================================
def download_image(url: str, save_dir: Path = QUERY_IMAGES_DIR, timeout_s: int = 20, retries: int = 2):
    save_dir.mkdir(parents=True, exist_ok=True)

    basename = os.path.basename(url.split("?")[0])
    if not basename:
        basename = f"img_{abs(hash(url))}.jpg"

    filename = save_dir / basename

    # if already present, do not download again
    if filename.exists() and filename.stat().st_size > 0:
        return str(filename)

    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=timeout_s, headers=headers)
            if resp.status_code == 200 and resp.content:
                filename.write_bytes(resp.content)
                print(f"[LOG] Image downloaded: {url} -> {filename}")
                return str(filename)
            else:
                print(f"[LOG] HTTP error downloading {url}: {resp.status_code}")
        except Exception as e:
            print(f"[LOG] Download error (attempt {attempt+1}/{retries+1}) {url}: {e}")
            time.sleep(1.0)

    return None


# =========================================================
# 2) CLIP Embedding (cluster-safe: FORCE_DEVICE)
# =========================================================
class UnifiedCLIPEmbedding:
    def __init__(self):
        print("[LOG] Initializing CLIP...")
        self.device = os.environ.get("FORCE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LOG] Device: {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model = self.model.to(self.device).eval()

    def embed_image(self, img_path: str) -> np.ndarray:
        print(f"[LOG] Embedding image: {img_path}")
        img = Image.open(img_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(tensor)[0].detach().cpu().numpy().astype(np.float32)

        emb = _safe_normalize(emb)
        return np.concatenate([np.zeros_like(emb, dtype=np.float32), emb]).astype(np.float32)


# =========================================================
# 3) INFERENZA QWEN2.5-VL (ollama.generate)
# =========================================================
def qwen_infer(prompt: str, image_paths: list[str], model: str):
    try:
        result = ollama.generate(
            model=model,
            prompt=prompt,
            images=image_paths
        )
        # some versions return a dict with "response", others with "message"
        return result.get("response") or result.get("message", {}).get("content", "")
    except Exception as e:
        print("[LOG] QWEN ERROR:", e)
        return ""


# =========================================================
# 4) Robust JSON parsing for duplicate check
# =========================================================
def robust_parse_json(text: str, default_obj: dict) -> dict:
    clean = (
        str(text).strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
        .replace("\\_", "_")
    )

    if "{" in clean and "}" in clean:
        clean = clean[clean.find("{"): clean.rfind("}") + 1]

    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return data
        return default_obj
    except Exception:
        print("[LOG] JSON COULD NOT BE PARSED:")
        print(text)
        return default_obj


def parse_duplicate_json(text: str, top_k: int) -> dict:
    default_obj = {
        "is_duplicate": "FALSE",
        "duplicate_rank": "",
        "confidence": "",
        "notes": ""
    }
    data = robust_parse_json(text, default_obj)

    def norm_bool(v):
        if isinstance(v, bool):
            return "TRUE" if v else "FALSE"
        if isinstance(v, str):
            u = v.strip().upper()
            if u in ["TRUE", "FALSE"]:
                return u
        return "FALSE"

    is_dup = norm_bool(data.get("is_duplicate"))

    dup_rank = data.get("duplicate_rank", "")
    conf = data.get("confidence", "")
    notes = data.get("notes", "")

    # Rule: if FALSE -> empty rank/confidence
    if is_dup == "FALSE":
        return {"is_duplicate": "FALSE", "duplicate_rank": "", "confidence": "", "notes": str(notes)}

    # normalize rank (1..top_k)
    try:
        if dup_rank != "":
            r = int(float(dup_rank))
            dup_rank = str(r) if (1 <= r <= top_k) else ""
    except Exception:
        dup_rank = ""

    # normalize confidence (0..1)
    try:
        if conf != "":
            cf = float(conf)
            cf = max(0.0, min(1.0, cf))
            conf = str(cf)
    except Exception:
        conf = ""

    return {
        "is_duplicate": is_dup,
        "duplicate_rank": str(dup_rank),
        "confidence": str(conf),
        "notes": str(notes)
    }


# =========================================================
# 5) QWEN2.5-VL – DUPLICATE CHECK (only)
# =========================================================
def qwen_duplicate_check(
    input_images: list[str],
    retrieved_images: list[str],
    retrieved_candidates_text: str,
    description_text: str,
    top_k: int,
    model: str
) -> dict:
    # IMPORTANT: Qwen sees a single list of images.
    # Convention: FIRST images = input, SUBSEQUENT images = retrieved
    all_images = input_images + retrieved_images

    # FIX: literal braces in an f-string must be doubled: {{ }}
    prompt = f"""
You are an assistant tasked with analyzing reports.

You have:
- ORIGINAL IMAGES of the report (the FIRST images you are given).
- Similar IMAGES from the knowledge base + textual information (the SUBSEQUENT images you are given).
- Textual information about the candidates (rank 1..{top_k}):
{retrieved_candidates_text}
- Report description: "{description_text}" (use it only as supporting information; it may be generic or inaccurate)

Determine whether the report in the FIRST images is a DUPLICATE of one of the retrieved candidates.

Definition of DUPLICATE:
- TRUE ONLY if the images clearly show the SAME event/scene/subject (the same pile or waste object, the same arrangement or distinctive features, the same context),
  even with minor differences (angle, zoom, quality).
- FALSE if it is merely similar (same category or generic scene) but is NOT the same case.
- If there is insufficient evidence, respond FALSE with low confidence.

Output rules:
- "is_duplicate" must be ONLY "TRUE" or "FALSE" (strings)
- If "is_duplicate" is "FALSE", then "duplicate_rank" and "confidence" must be empty strings

Required fields in the final JSON:
1) "is_duplicate": "TRUE/FALSE"
2) "duplicate_rank": "1..{top_k} or empty"
3) "confidence": "0..1 or empty"
4) "notes": "brief rationale (cite 1–2 pieces of visual evidence and use the description only if consistent)"

Example of a valid response:
{{
  "is_duplicate": "TRUE",
  "duplicate_rank": "2",
  "confidence": "0.8",
  "notes": "The images show the same pile with the same arrangement and distinctive feature; the description is consistent."
}}

Do not include any text outside the JSON.
"""

    print("[LOG] Calling Qwen2.5-VL (duplicate check) with images:", all_images)
    raw = qwen_infer(prompt, all_images, model=model)

    print("[LOG] Raw Qwen response (duplicate check):")
    print(raw)

    return parse_duplicate_json(raw, top_k=top_k)


# =========================================================
# 6) Pipeline: duplicate check ONLY -> output CSV
# =========================================================
def duplicate_check_only(input_csv: Path, output_csv: str, top_k: int = 3):
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:7b")
    print(f"[LOG] Qwen-VL model: {model}")

    print("[LOG] Loading CSV:", input_csv)
    df = pd.read_csv(input_csv)

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Chroma DB non trovato: {CHROMA_DIR}")

    client = PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("waste_combined")

    embedder = UnifiedCLIPEmbedding()

    # CSV output columns
    out_report_id = []
    out_dup_check = []
    out_dup_rank = []
    out_conf = []
    out_notes = []
    out_error = []

    for idx, row in df.iterrows():
        print(f"\n[LOG] === Row {idx} ===")
        report_id = get_row_report_id(row)

        description_text = str(row.get("Description", "")).strip()
        picture_field = str(row.get("Picture", "")).strip()
        urls = [u.strip() for u in picture_field.replace(";", ",").split(",") if u.strip()]

        input_imgs = []
        for u in urls:
            p = download_image(u)
            if p:
                input_imgs.append(p)

        if not input_imgs:
            out_report_id.append(report_id)
            out_dup_check.append("FALSE")
            out_dup_rank.append("")
            out_conf.append("")
            out_notes.append("")
            out_error.append("no_images")
            continue

        # retrieval via CLIP (use the first image as the query)
        qvec = embedder.embed_image(input_imgs[0]).tolist()
        retr = collection.query(
            query_embeddings=[qvec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        docs = retr.get("documents", [[]])[0] if retr.get("documents") else []
        mds = retr.get("metadatas", [[]])[0] if retr.get("metadatas") else []
        dists = retr.get("distances", [[]])[0] if retr.get("distances") else []

        retrieved_imgs = []
        for md in mds:
            img_path = (md or {}).get("image_path")
            if img_path and os.path.exists(img_path):
                retrieved_imgs.append(img_path)

        # numbered candidate text
        cand_lines = []
        n = min(top_k, len(docs), len(mds), len(dists))
        for i in range(n):
            md = mds[i] or {}
            rid = get_meta_report_id(md)
            imgp = md.get("image_path", "")
            dist = dists[i]
            snippet = str(docs[i]).replace("\n", " ").strip()
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            cand_lines.append(
                f"[CANDIDATE rank={i+1}] distance={dist:.4f} report_id={rid} image_path={imgp}\n"
                f"doc: {snippet}"
            )

        retrieved_candidates_text = "\n\n".join(cand_lines) if cand_lines else "(no candidates available)"

        dup = qwen_duplicate_check(
            input_images=input_imgs,
            retrieved_images=retrieved_imgs,
            retrieved_candidates_text=retrieved_candidates_text,
            description_text=description_text,
            top_k=top_k,
            model=model
        )

        out_report_id.append(report_id)
        out_dup_check.append(dup.get("is_duplicate", "FALSE"))
        out_dup_rank.append(dup.get("duplicate_rank", ""))
        out_conf.append(dup.get("confidence", ""))
        out_notes.append(dup.get("notes", ""))
        out_error.append("")

    out_df = pd.DataFrame({
        "report_id": out_report_id,
        "duplicate_check": out_dup_check,
        "duplicate_rank": out_dup_rank,
        "confidence": out_conf,
        "notes": out_notes,
        "error": out_error
    })

    out_path = OUT_DIR / Path(output_csv).name
    out_df.to_csv(out_path, index=False)
    print("[LOG] CSV output saved:", out_path)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    duplicate_check_only(
        input_csv=REPORTS_DIR / "split_01_test.csv",
        output_csv="duplicate_check_results_qwen.csv",
        top_k=int(os.environ.get("TOP_K", "3"))
    )
