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
from chromadb import PersistentClient  # lasciato per minimizzare diff, ma non usato in Agent C
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


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

# Cache per download modelli/pesi (cluster-safe)
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
# 2) CLIP Embedding (non necessario per Agent C, ma lasciato)
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
        print(f"[LOG] Embedding immagine: {img_path}")
        img = Image.open(img_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(tensor)[0].detach().cpu().numpy().astype(np.float32)

        emb = _safe_normalize(emb)

        return np.concatenate([np.zeros_like(emb, dtype=np.float32), emb]).astype(np.float32)


# =========================================================
# 3) JSON Parsing robusto
# =========================================================
def robust_parse_json(text):
    clean = (
        text.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    clean = clean.replace("\\_", "_")

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
            "Notes": "",
            "multiple_subjects": "",
            "accurate_description": "",
            "human_in_frame": "",
            "clear_subject": ""
        }

    cw_gen = (
        data.get("containedWaste_generated")
        or data.get("ContainedWaste_generated")
        or data.get("GeneratedContainedWaste")
        or data.get("ContainedWaste")
        or ""
    )

    size_gen = (
        data.get("size_generated")
        or data.get("Size_generated")
        or data.get("GeneratedSize")
        or data.get("Size")
        or ""
    )

    if isinstance(cw_gen, list):
        cw_gen = ", ".join(str(x) for x in cw_gen)

    cw_gen = str(cw_gen)
    size_gen = str(size_gen)

    notes = data.get("notes", data.get("Notes", ""))
    notes = str(notes)

    def norm_bool(val):
        if isinstance(val, bool):
            return "TRUE" if val else "FALSE"
        if isinstance(val, str):
            return val.upper()
        return ""

    return {
        "ContainedWaste": cw_gen,
        "Size": size_gen,
        "Notes": notes,
        "multiple_subjects": norm_bool(data.get("multiple_subjects")),
        "accurate_description": norm_bool(data.get("accurate_description")),
        "human_in_frame": norm_bool(data.get("human_in_frame")),
        "clear_subject": norm_bool(data.get("clear_subject"))
    }


# =========================================================
# 4) LLaVA – verifica valori originali e generazione nuovi
# =========================================================
def verify_riga_llava(
    input_images,
    original_cw,
    original_size,
    description_text,
    llm
):
    print("[LOG] Invio immagini input:", input_images)
    msg_input = HumanMessage(
        content="Queste sono le IMMAGINI ORIGINALI da valutare.",
        additional_kwargs={"images": input_images}
    )

    # PROMPT (NON MODIFICATO)
    msg_instruction = HumanMessage(
        content=f"""
Sei un assistente per la **verifica della classificazione dei rifiuti**.

Per ogni record hai:
- IMMAGINI da analizzare (primo messaggio)
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
    )

    print("[LOG] Invio messaggi a LLaVA...")
    response = llm.invoke([msg_input, msg_instruction])

    print("[LOG] Risposta grezza:")
    print(response.content)

    return robust_parse_json(response.content)


# =========================================================
# 5) MATCHING RULES (invariato)
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
# 6) PIPELINE COMPLETA (cluster-safe: paths + ollama env)
# =========================================================
def verify_csv(input_csv, output_csv, top_k=3):
    print("[LOG] Caricamento CSV:", input_csv)
    df = pd.read_csv(input_csv)

    # Configurazione Ollama (server deve essere attivo sul nodo)
    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    llava_model = os.environ.get("OLLAMA_MODEL", "llava:13b")
    print(f"[LOG] OLLAMA_HOST={ollama_host}  MODEL={llava_model}")

    llm = ChatOllama(model=llava_model, base_url=ollama_host)

    gen_cw, gen_sz = [], []
    match_cw, match_sz = [], []
    notes_list = []
    out_mult, out_accdesc, out_human, out_clear = [], [], [], []
    timings = []

    for idx, row in df.iterrows():
        print(f"\n[LOG] === Riga {idx} ===")
        start = time.time()

        original_cw = str(row.get("ContainedWaste", ""))
        original_size = str(row.get("Size", ""))
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
            gen_cw.append("")
            gen_sz.append("")
            match_cw.append("WRONG")
            match_sz.append("WRONG")
            notes_list.append("Nessuna immagine disponibile.")
            out_mult.append("")
            out_accdesc.append("")
            out_human.append("")
            out_clear.append("")
            timings.append({"row": int(idx), "seconds": 0.0})
            continue

        # retrieval via CLIP (non usato in Agent C)

        parsed = verify_riga_llava(
            input_images=input_imgs,
            original_cw=original_cw,
            original_size=original_size,
            description_text=description_text,
            llm=llm
        )

        generated_cw = parsed.get("ContainedWaste", "")
        generated_sz = parsed.get("Size", "")
        notes = parsed.get("Notes", "")

        out_mult.append(parsed.get("multiple_subjects", ""))
        out_accdesc.append(parsed.get("accurate_description", ""))
        out_human.append(parsed.get("human_in_frame", ""))
        out_clear.append(parsed.get("clear_subject", ""))

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
    df["ContainedWaste_match"] = match_cw
    df["Size_match"] = match_sz
    df["Notes"] = notes_list
    df["multiple_subjects"] = out_mult
    df["accurate_description"] = out_accdesc
    df["human_in_frame"] = out_human
    df["clear_subject"] = out_clear

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
        output_csv="reports_agent_C.csv",
        top_k=3
    )








