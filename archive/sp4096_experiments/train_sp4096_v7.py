"""SP4096 tokenizer pipeline — download raw text to C:, tokenize, move shards to E:"""
import os, sys, shutil, time, subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download

LOG_PATH = Path(r"e:\Projects\Proj\golf\records\track_non_record_16mb\2026-04-01_EliteUTv22p8_12StepRecurrence_Windows3090\logs\p8_A1_SP4096_v7.log")
WORK_DIR = Path(r"C:\sp4096_temp")
DEST_DIR = Path(r"e:\Projects\Proj\golf\data")
REPO_ID = "willdepueoai/parameter-golf"
FILENAME = "datasets/docs_selected.jsonl"

def log(msg: str):
    ts = time.strftime("[%d-%m-%Y %H:%M:%S")
    ts += f".{int(time.time()*100)%100:02d}]"
    line = f"{ts} {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        f.write("")
    
    log("=== SP4096 v7: Pipeline start ===")
    
    # Clean + create work dir
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    WORK_DIR.mkdir(parents=True)
    log(f"Work dir: {WORK_DIR}")
    
    # Download with huggingface_hub (proven working)
    log(f"Downloading {FILENAME} from {REPO_ID}...")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        local_dir=WORK_DIR,
        local_dir_use_symlinks=False,
    )
    log(f"Downloaded to: {downloaded}")
    
    # Copy tokenizer_specs.json
    spec_src = DEST_DIR / "tokenizer_specs.json"
    if spec_src.exists():
        shutil.copy(spec_src, WORK_DIR / "tokenizer_specs.json")
        log("Copied tokenizer_specs.json")
    
    # Find the tokenization script
    script = DEST_DIR / "download_hf_docs_and_tokenize.py"
    if not script.exists():
        # Also check relative to CWD
        script = Path(r"e:\Projects\Proj\golf\records\track_non_record_16mb\2026-04-01_EliteUTv22p8_12StepRecurrence_Windows3090") / ".." / ".." / ".." / "data" / "download_hf_docs_and_tokenize.py"
    if not script.exists():
        log(f"ERROR: Cannot find download_hf_docs_and_tokenize.py")
        sys.exit(1)
    
    log(f"Running: python {script} --output-root {WORK_DIR}")
    result = subprocess.run(
        [sys.executable, str(script), "--output-root", str(WORK_DIR)],
        cwd=str(WORK_DIR),
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        log(f"ERROR: Tokenization failed (code {result.returncode})")
        log(f"STDERR: {result.stderr[-2000:]}")
        sys.exit(result.returncode)
    log(f"Tokenization output: {result.stdout[-2000:]}")
    
    # Move results to E:
    shard_dir = WORK_DIR / "fineweb10B_sp4096"
    if shard_dir.exists():
        dest_shard = DEST_DIR / "fineweb10B_sp4096"
        if dest_shard.exists():
            shutil.rmtree(dest_shard)
        shutil.move(str(shard_dir), str(dest_shard))
        log(f"Moved {shard_dir} -> {dest_shard}")
    
    # Copy tokenizer files
    for fname in ["fineweb_4096_bpe.model", "fineweb_4096_bpe.vocab"]:
        src = WORK_DIR / fname
        if src.exists():
            shutil.copy(src, DEST_DIR / fname)
            log(f"Copied {fname}")
    
    # Clean up
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    log("=== SP4096 v7: Pipeline complete! ===")

if __name__ == "__main__":
    main()