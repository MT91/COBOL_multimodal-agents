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
BASE_DIR = Path(__file__).resolve().parent

# root del progetto cobol
PROJECT_ROOT = BASE_DIR.parent

# input CSV sempre qui
REPORTS_DIR = PROJECT_ROOT / "reports" / "cv10_splits"

# immagini temporanee per le query
QUERY_IMAGES_DIR = BASE_DIR / "query_images"
QUERY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# output dell'agente (restano nel case)
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
# 1) Download immagini (cache + retry)
# =========================================================
def download_image(url: str, save_dir: Path = QUERY_IMAGES_DIR, timeout_s: int = 20, retries: int = 2):
    save_dir.mkdir(parents=True, exist_ok=True)

    basename = os.path.basename(url.split("?")[0])
    if not basename:
        basename = f"img_{abs(hash(url))}.jpg"

    filename = save_dir / basename

    # se già presente, non riscaricare
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

    def embed_image(self, img_path: str) -> np.ndarray:
        print(f"[LOG] Embedding immagine: {img_path}")
        img = Image.open(img_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(tensor)[0].detach().cpu().numpy().astype(np.float32)

        emb = _safe_normalize(emb)

        return np.concatenate([np.zeros_like(emb, dtype=np.float32), emb]).astype(np.float32)


# =========================================================
# 3) INFERENZA QWEN2.5-VL (robusta a diverse versioni)
# =========================================================
def qwen_infer(prompt, image_paths):
    try:
        result = ollama.generate(
            model="qwen2.5vl:7b",
            prompt=prompt,
            images=image_paths
        )
        # alcune versioni ritornano dict con "response", altre "message"
        return result.get("response") or result.get("message", {}).get("content", "")
    except Exception as e:
        print("[LOG] ERRORE QWEN:", e)
        return ""


# =========================================================
# 4) JSON Parsing robusto
# =========================================================
def robust_parse_json(text):
    clean = (
        text.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
        .replace("\\_", "_")
    )

    if "{" in clean and "}" in clean:
        clean = clean[clean.find("{"): clean.rfind("}") + 1]

    try:
        data = json.loads(clean)
        print("[LOG] JSON parsing riuscito.")
    except Exception:
        print("[LOG] JSON NON PARSABILE:")
        print(text)
        return {
            "ContainedWaste": "",
            "Size": "",
            "multiple_subjects": "",
            "accurate_description": "",
            "human_in_frame": "",
            "clear_subject": "",
            "Notes": ""
        }

    cw = (
        data.get("containedWaste_generated")
        or data.get("ContainedWaste")
        or ""
    )
    if isinstance(cw, list):
        cw = ", ".join([str(x) for x in cw])

    size = (
        data.get("size_generated")
        or data.get("Size")
        or ""
    )

    def norm_bool(x):
        if isinstance(x, bool):
            return "TRUE" if x else "FALSE"
        if isinstance(x, str):
            return x.strip().upper()
        return ""

    mult = norm_bool(data.get("multiple_subjects", ""))
    accdesc = norm_bool(data.get("accurate_description", ""))
    human = norm_bool(data.get("human_in_frame", ""))
    clear = norm_bool(data.get("clear_subject", ""))

    notes = data.get("notes", data.get("Notes", ""))
    if not isinstance(notes, str):
        notes = str(notes)

    return {
        "ContainedWaste": str(cw),
        "Size": str(size),
        "multiple_subjects": mult,
        "accurate_description": accdesc,
        "human_in_frame": human,
        "clear_subject": clear,
        "Notes": notes
    }


# =========================================================
# 5) QWEN2.5-VL – matching e classificazione
# =========================================================
def verify_riga_qwen(
    input_images,
    original_cw,
    original_size,
    description_text
):
    all_images = input_images

    # PROMPT (NON MODIFICATO)
    prompt = f"""
Sei un assistente per la **verifica della classificazione dei rifiuti**.

Per ogni record hai:
- IMMAGINI da analizzare 
- Una classificazione fatta dall'utente.

Devi:
1) Valutare SE la classificazione dell'utente è corretta.
2) Proporre la TUA classificazione indipendente (anche se coincide con quella dell'utente).
3) Spiegare brevemente il tuo ragionamento.

Classificazione dell'utente:
- ContainedWaste (utente): "{original_cw}"
- Size (utente): "{original_size}"

La colonna Description dell'utente contiene:
"{description_text}"

Campi richiesti nel JSON finale:

1) containedWaste_generated: la tua classificazione completa per i tipi dei rifiuti presenti nelle immagini che possono essere solo quelli in questa lista: [aluminum/metal,waste not identifiable,construction materials,glass,plastic,textiles,wood,
bulky waste,electronic appareil,tyres,paper,chemicals and drugs,organic,other]
2) size_generated: la tua classificazione completa per la dimensione del rifiuto che può essere: [small,medium,big]
3) notes: motivazione dettagliata e spiegazione del ragionamento che ti ha portato a quella classificazione
4) multiple_subjects: TRUE/FALSE (se sono presenti più soggetti distinti nelle immagini)
5) accurate_description: TRUE/FALSE (se la Description indicata dall'utente in "{description_text}" riflette accuratamente i rifiuti presenti nelle immagini analizzate)
6) human_in_frame: TRUE/FALSE (se ci sono persone nelle immagini)
7) clear_subject: TRUE/FALSE (se i rifiuti sono chiaramente visibili)

Rispondi SOLO in JSON valido. Nessun testo fuori dal JSON.

Esempio di risposta valida:
{{
  "containedWaste_generated": "plastic, paper",
  "size_generated": "medium",
  "notes": "La dimensione è medium. L'utente ha indicato 'plastic, paper' che coincide con la mia valutazione.",
  "multiple_subjects": "TRUE",
  "accurate_description": "FALSE",
  "human_in_frame": "FALSE",
  "clear_subject": "TRUE"
}}
"""

    print("[LOG] Chiamata Qwen2.5-VL con immagini:", all_images)
    raw = qwen_infer(prompt, all_images)

    print("[LOG] Risposta grezza Qwen:")
    print(raw)

    return robust_parse_json(raw)


# =========================================================
# 6) MATCHING RULES
# =========================================================
def match_contained_waste(original, generated):
    if not original or not generated:
        return "WRONG"

    orig_set = {x.strip().lower() for x in original.split(",") if x.strip()}
    gen_set = {x.strip().lower() for x in generated.split(",") if x.strip()}

    if not gen_set:
        return "WRONG"

    if orig_set == gen_set:
        return "CORRECT"
    if orig_set.issubset(gen_set):
        return "INCOMPLETE"
    return "WRONG"


def match_size(original, generated):
    if not original or not generated:
        return "WRONG"
    return "CORRECT" if original.strip().lower() == generated.strip().lower() else "WRONG"


# =========================================================
# 7) PIPELINE COMPLETA (cluster-safe: paths + mkdir + times)
# =========================================================
def verify_csv(input_csv, output_csv, top_k=3):
    print("[LOG] Caricamento CSV:", input_csv)
    df = pd.read_csv(input_csv)

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Chroma DB non trovato: {CHROMA_DIR}")

    client = PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("waste_combined")

    embedder = UnifiedCLIPEmbedding()

    gen_cw, gen_sz = [], []
    gen_mult, gen_accdesc, gen_human, gen_clear = [], [], [], []
    match_cw, match_sz = [], []
    notes_list = []
    timings = []

    for idx, row in df.iterrows():
        start = time.time()
        print(f"\n[LOG] === Riga {idx} ===")

        original_cw = str(row.get("ContainedWaste", ""))
        original_size = str(row.get("Size", ""))
        description_text = str(row.get("Description", ""))

        picture_field = str(row.get("Picture", "")).strip()
        urls = [u.strip() for u in picture_field.replace(";", ",").split(",") if u.strip()]

        input_imgs = []
        for u in urls:
            p = download_image(u)
            if p:
                input_imgs.append(p)

        if not input_imgs:
            gen_cw.append("")
            gen_sz.append("")
            gen_mult.append("")
            gen_accdesc.append("")
            gen_human.append("")
            gen_clear.append("")
            match_cw.append("WRONG")
            match_sz.append("WRONG")
            notes_list.append("Nessuna immagine disponibile.")
            timings.append({"row": int(idx), "seconds": 0.0})
            continue

        # NB: embedder/collection non usati qui (LLM-only), ma lasciati invariati per minimizzare diff.

        parsed = verify_riga_qwen(
            input_images=input_imgs,
            original_cw=original_cw,
            original_size=original_size,
            description_text=description_text
        )

        generated_cw = parsed.get("ContainedWaste", "")
        generated_sz = parsed.get("Size", "")
        gen_mult.append(parsed.get("multiple_subjects", ""))
        gen_accdesc.append(parsed.get("accurate_description", ""))
        gen_human.append(parsed.get("human_in_frame", ""))
        gen_clear.append(parsed.get("clear_subject", ""))
        notes = parsed.get("Notes", "")

        gen_cw.append(generated_cw)
        gen_sz.append(generated_sz)
        match_cw.append(match_contained_waste(original_cw, generated_cw))
        match_sz.append(match_size(original_size, generated_sz))
        notes_list.append(notes)

        end = time.time()
        timings.append({"row": int(idx), "seconds": round(end - start, 3)})
        print(f"[LOG] Tempo riga {idx}: {end - start:.2f} sec")

    df["containedWaste_generated"] = gen_cw
    df["size_generated"] = gen_sz
    df["multiple_subjects"] = gen_mult
    df["accurate_description"] = gen_accdesc
    df["human_in_frame"] = gen_human
    df["clear_subject"] = gen_clear
    df["ContainedWaste_match"] = match_cw
    df["Size_match"] = match_sz
    df["Notes"] = notes_list

    output_csv = OUT_DIR / Path(output_csv).name
    df.to_csv(output_csv, index=False)
    print("[LOG] Salvato report agente:", output_csv)
    
    # --- tempi ---
    agent_name = Path(__file__).stem   # es. agent_A
    times_path = OUT_DIR / f"classification_times_{agent_name}.csv"
    pd.DataFrame(timings).to_csv(times_path, index=False)
    print("[LOG] Salvati i tempi in:", times_path)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    verify_csv(
        input_csv=REPORTS_DIR / "split_01_test.csv",
        output_csv="reports_agent_F.csv",
        top_k=3
    )
