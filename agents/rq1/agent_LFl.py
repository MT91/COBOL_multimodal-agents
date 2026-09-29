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
from chromadb import PersistentClient  # kept to minimize the diff, but not used in Agent C
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# =======================
# Paths + cache (cluster-safe)
# =======================
BASE_DIR = Path(__file__).resolve().parent

# root of the cobol project
PROJECT_ROOT = BASE_DIR.parent

# input CSV always here
REPORTS_DIR = PROJECT_ROOT / "reports" / "cv10_splits"

# temporary images for queries
QUERY_IMAGES_DIR = BASE_DIR / "query_images"
QUERY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# agent output (stays in the case directory)
OUT_DIR = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_DIR = BASE_DIR / "chroma_db_combined"

# Cache for model/weight downloads (cluster-safe)
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
# 2) CLIP Embedding (not needed for Agent C, but retained)
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

    def embed_image(self, img_path):
        print(f"[LOG] Embedding image: {img_path}")
        img = Image.open(img_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(tensor)[0].detach().cpu().numpy().astype(np.float32)

        emb = _safe_normalize(emb)

        return np.concatenate([np.zeros_like(emb, dtype=np.float32), emb]).astype(np.float32)


# =========================================================
# 3) Robust JSON Parsing
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
        print("[LOG] JSON parsing succeeded.")
    except Exception:
        print("[LOG] JSON COULD NOT BE PARSED:")
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
# 4) LLaVA – verify original values and generate new ones
# =========================================================
def verify_riga_llava(
    input_images,
    original_cw,
    original_size,
    description_text,
    llm
):
    print("[LOG] Sending input images:", input_images)
    msg_input = HumanMessage(
        content="These are the ORIGINAL IMAGES to evaluate.",
        additional_kwargs={"images": input_images}
    )

    # PROMPT (UNCHANGED)
    msg_instruction = HumanMessage(
        content=f"""
You are an assistant for **verifying waste classification**.
Below are two brief examples (input -> expected JSON output) to show you the format — use them as a reference.

Example 1:
Input (description): "Transparent bag containing two plastic bottles and paper."
Output JSON:
{{
  "containedWaste_generated": "plastic, paper",
  "size_generated": "medium",
  "notes": "The bag clearly contains two plastic bottles and paper; its estimated size is larger than a standard bottle.",
  "multiple_subjects": "TRUE",
  "accurate_description": "TRUE",
  "human_in_frame": "FALSE",
  "clear_subject": "TRUE"
}}

Example 2:
Input (description): "Discarded beverage can near a bush."
Output JSON:
{{
  "containedWaste_generated": "aluminum/metal",
  "size_generated": "small",
  "notes": "The shape and shiny color indicate a can; its size is similar to a standard bottle or can.",
  "multiple_subjects": "FALSE",
  "accurate_description": "TRUE",
  "human_in_frame": "FALSE",
  "clear_subject": "TRUE"
}}

Now evaluate the actual case.

For each record, you have:
- IMAGES to analyze (first message)
- A classification provided by the user.

You must:
1) Assess WHETHER the user's classification is correct.
2) Provide YOUR independent classification (even if it matches the user's).
3) Briefly explain your reasoning.

User's classification:
- ContainedWaste (user): "{original_cw}"
- Size (user): "{original_size}"

The user's Description column contains:
"{description_text}"

Required fields in the final JSON:

1) containedWaste_generated: your complete classification of the waste types present in the images, using only the types in this list: [aluminum/metal,waste not identifiable,construction materials,glass,plastic,textiles,wood,
bulky waste,electronic appareil,tyres,paper,chemicals and drugs,organic,other]
2) size_generated: your complete classification of the waste size, which can be: [small,medium,big]
3) notes: detailed rationale and explanation of the reasoning behind your classification
4) multiple_subjects: TRUE/FALSE (whether multiple distinct subjects are present in the images)
5) accurate_description: TRUE/FALSE (whether the Description provided by the user in "{description_text}" accurately reflects the waste present in the analyzed images)
6) human_in_frame: TRUE/FALSE (whether people are present in the images)
7) clear_subject: TRUE/FALSE (whether the waste is clearly visible)

Respond ONLY in valid JSON. Do not include any text outside the JSON.

Example of a valid response:
{{
  "containedWaste_generated": "plastic, paper",
  "size_generated": "medium",
  "notes": "The size is medium. The user indicated 'plastic, paper', which matches my assessment.",
  "multiple_subjects": "TRUE",
  "accurate_description": "FALSE",
  "human_in_frame": "FALSE",
  "clear_subject": "TRUE"
}}
"""
    )

    print("[LOG] Sending messages to LLaVA...")
    response = llm.invoke([msg_input, msg_instruction])

    print("[LOG] Raw response:")
    print(response.content)

    return robust_parse_json(response.content)


# =========================================================
# 5) MATCHING RULES (unchanged)
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
# 6) COMPLETE PIPELINE (cluster-safe: paths + ollama env)
# =========================================================
def verify_csv(input_csv, output_csv, top_k=3):
    print("[LOG] Loading CSV:", input_csv)
    df = pd.read_csv(input_csv)

    # Ollama configuration (the server must be running on the node)
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
        print(f"\n[LOG] === Row {idx} ===")
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
            print("[LOG] No images available for this row.")
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

        # retrieval via CLIP (not used in Agent C)

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
        print(f"[LOG] Time for row {idx}: {end - start:.2f} sec")

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
    print("[LOG] Agent report saved:", output_csv)
    
    # --- timings ---
    agent_name = Path(__file__).stem   # e.g. agent_A
    times_path = OUT_DIR / f"classification_times_{agent_name}.csv"
    pd.DataFrame(timings).to_csv(times_path, index=False)
    print("[LOG] Timings saved to:", times_path)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    verify_csv(
        input_csv=REPORTS_DIR / "split_01_test.csv",
        output_csv="reports_agent_LFl.csv",
        top_k=3
    )







