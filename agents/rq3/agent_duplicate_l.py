import os
import torch
import numpy as np
import pandas as pd
import requests
import json
import time
from pathlib import Path
from PIL import Image
import open_clip
from chromadb import PersistentClient
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# =======================
# Paths + cache (cluster-safe)
# =======================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
REPORTS_DIR = PROJECT_ROOT / "reports" / "cv10_splits"

QUERY_IMAGES_DIR = BASE_DIR / "query_images"
QUERY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_DIR = BASE_DIR / "chroma_db_combined"

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
# ID helpers (adatta se hai colonne/metadata diversi)
# =========================================================
def get_row_report_id(row: pd.Series) -> str:
    """
    Prova a ricavare un identificatore stabile del report dal CSV.
    Adatta la lista di chiavi in base al tuo CSV.
    """
    for k in ["ReportId", "report_id", "Id", "ID", "reportId", "source_id", "uuid", "report_uuid"]:
        if k in row and pd.notna(row[k]):
            s = str(row[k]).strip()
            if s:
                return s
    return f"row_{int(row.name)}"


def get_meta_report_id(md: dict) -> str:
    """
    Prova a ricavare un identificatore stabile del report dai metadata Chroma.
    Adatta la lista di chiavi in base al tuo ingestion.
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
# 1) Download immagini (cache + retry)
# =========================================================
def download_image(url: str, save_dir: Path = QUERY_IMAGES_DIR, timeout_s: int = 20, retries: int = 2):
    save_dir.mkdir(parents=True, exist_ok=True)

    basename = os.path.basename(url.split("?")[0])
    if not basename:
        basename = f"img_{abs(hash(url))}.jpg"

    filename = save_dir / basename

    if filename.exists() and filename.stat().st_size > 0:
        return str(filename)

    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=timeout_s, headers=headers)
            if resp.status_code == 200 and resp.content:
                filename.write_bytes(resp.content)
                print(f"[LOG] Scaricata immagine: {url} -> {filename}")
                return str(filename)
            else:
                print(f"[LOG] Errore HTTP scaricando {url}: {resp.status_code}")
        except Exception as e:
            print(f"[LOG] Errore download (tentativo {attempt+1}/{retries+1}) {url}: {e}")
            time.sleep(1.0)

    return None


# =========================================================
# 2) CLIP Embedding (cluster-safe: FORCE_DEVICE)
# =========================================================
class UnifiedCLIPEmbedding:
    def __init__(self):
        print("[LOG] Inizializzazione CLIP...")
        self.device = os.environ.get("FORCE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LOG] Device: {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model = self.model.to(self.device).eval()

    def embed_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(tensor)[0].detach().cpu().numpy().astype(np.float32)

        emb = _safe_normalize(emb)
        return np.concatenate([np.zeros_like(emb, dtype=np.float32), emb]).astype(np.float32)


# =========================================================
# 3) JSON parsing robusto
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
        print("[LOG] JSON NON PARSABILE:")
        print(text)
        return default_obj


def parse_duplicate_json(text: str, top_k: int) -> dict:
    default_obj = {
        "is_duplicate": "FALSE",      # TRUE/FALSE
        "duplicate_rank": "",         # 1..top_k se TRUE
        "confidence": "",             # 0..1
        "notes": ""                   # breve
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

    # normalize confidence
    try:
        if conf != "":
            conf_f = float(conf)
            conf_f = max(0.0, min(1.0, conf_f))
            conf = str(conf_f)
    except Exception:
        conf = ""

    # normalize rank
    try:
        if dup_rank != "":
            r = int(float(dup_rank))
            if 1 <= r <= top_k:
                dup_rank = str(r)
            else:
                dup_rank = ""
    except Exception:
        dup_rank = ""

    return {
        "is_duplicate": is_dup,
        "duplicate_rank": dup_rank,
        "confidence": conf,
        "notes": str(notes)
    }


# =========================================================
# 4) LLaVA – DUPLICATE CHECK (solo)
# =========================================================
def ask_duplicate_llava(
    input_images,
    retrieved_images,
    retrieved_candidates_text,
    description_text,
    llm,
    top_k: int
):
    msg_input = HumanMessage(
        content="Queste sono le IMMAGINI ORIGINALI del report da verificare.",
        additional_kwargs={"images": input_images}
    )

    msg_retrieved = HumanMessage(
        content=(
            "Queste sono le IMMAGINI SIMILI recuperate dalla knowledge base.\n"
            "Ogni candidato è numerato (rank 1..k) e ha anche un riepilogo testuale.\n\n"
            f"{retrieved_candidates_text}\n\n"
            "Valutale come possibili duplicati dell'input."
        ),
        additional_kwargs={"images": retrieved_images}
    )

    # FIX: in una f-string le graffe letterali vanno raddoppiate: {{ }}
    msg_instruction = HumanMessage(
        content=f"""
Sei un assistente che deve analizzare.

Hai:
- IMMAGINI ORIGINALI del report (primo messaggio)
- IMMAGINI simili dalla KB + info testuali (secondo messaggio)
- Descrizione del report: "{description_text}" (usala solo come supporto, potrebbe essere generica o imprecisa)

Devi rilevare se il report nel primo messaggio è un DUPLICATO di uno dei candidati recuperati.

Definizione di DUPLICATO:
- TRUE SOLO se le immagini mostrano chiaramente lo STESSO evento/scena/soggetto (stesso cumulo/oggetto rifiuti, stessa disposizione o elementi distintivi, stesso contesto),
  anche con piccole variazioni (angolo, zoom, qualità).
- FALSE se è solo simile (stessa categoria di rifiuto, scena generica, contesto simile) ma NON è lo stesso caso.
- Se non hai evidenza sufficiente, rispondi FALSE con confidence bassa.

Se TRUE:
- indica il candidato (rank 1..{top_k}) più probabile
- indica confidence tra 0 e 1

Regole di output:
- "is_duplicate" deve essere SOLO "TRUE" o "FALSE" (stringhe)
- Se "is_duplicate" è "FALSE", allora "duplicate_rank" e "confidence" devono essere stringhe vuote

Campi richiesti nel JSON finale:
1) "is_duplicate": "TRUE/FALSE"
2) "duplicate_rank": "1..{top_k} oppure vuoto"
3) "confidence": "0..1 oppure vuoto"
4) "notes": "breve motivazione (cita 1-2 evidenze visive, e usa la descrizione solo se coerente)"

Esempio di risposta valida:
{{
  "is_duplicate": "TRUE",
  "duplicate_rank": "2",
  "confidence": "0.8",
  "notes": "Le immagini mostrano lo stesso cumulo con stessa disposizione e stesso elemento distintivo; la descrizione è coerente."
}}

Nessun testo fuori dal JSON.
"""
    )

    print("[LOG] Invio messaggi a LLaVA (duplicate-check)...")
    response = llm.invoke([msg_input, msg_retrieved, msg_instruction])

    print("[LOG] Risposta grezza duplicate-check:")
    print(response.content)

    return parse_duplicate_json(response.content, top_k=top_k)


# =========================================================
# 5) Pipeline: SOLO duplicate-check -> output CSV
# =========================================================
def duplicate_check_only(input_csv: Path, output_csv: str, top_k: int = 3):
    print("[LOG] Caricamento CSV:", input_csv)
    df = pd.read_csv(input_csv)

    print("[LOG] Caricamento Chroma...")
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Chroma DB non trovato: {CHROMA_DIR}")

    client = PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("waste_combined")

    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    llava_model = os.environ.get("OLLAMA_MODEL", "llava:13b")
    print(f"[LOG] OLLAMA_HOST={ollama_host}  MODEL={llava_model}")

    llm = ChatOllama(model=llava_model, base_url=ollama_host)
    embedder = UnifiedCLIPEmbedding()

    # colonne output
    out_report_id = []
    out_dup_check = []
    out_dup_rank = []
    out_conf = []
    out_notes = []
    out_error = []

    for idx, row in df.iterrows():
        print(f"\n[LOG] === Riga {idx} ===")
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
            print("[LOG] Nessuna immagine disponibile per questa riga.")
            out_report_id.append(report_id)
            out_dup_check.append("FALSE")
            out_dup_rank.append("")
            out_conf.append("")
            out_notes.append("")
            out_error.append("no_images")
            continue

        # retrieval
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
                f"[CANDIDATO rank={i+1}] distance={dist:.4f} report_id={rid} image_path={imgp}\n"
                f"doc: {snippet}"
            )

        retrieved_candidates_text = "\n\n".join(cand_lines) if cand_lines else "(nessun candidato disponibile)"

        dup = ask_duplicate_llava(
            input_images=input_imgs,
            retrieved_images=retrieved_imgs,
            retrieved_candidates_text=retrieved_candidates_text,
            description_text=description_text,
            llm=llm,
            top_k=top_k
        )

        out_report_id.append(report_id)
        out_dup_check.append(dup.get("is_duplicate", "FALSE"))
        out_dup_rank.append(dup.get("duplicate_rank", ""))
        out_conf.append(dup.get("confidence", ""))
        out_notes.append(dup.get("notes", ""))
        out_error.append("")

    # scrivi CSV (solo duplicate-check; non riscrive il CSV di input)
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
    print("[LOG] Salvato output CSV:", out_path)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    duplicate_check_only(
        input_csv=REPORTS_DIR / "split_01_test.csv",
        output_csv="duplicate_check_results_llava.csv",
        top_k=int(os.environ.get("TOP_K", "3"))
    )
