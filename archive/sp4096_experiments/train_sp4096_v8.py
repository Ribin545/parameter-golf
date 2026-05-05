"""SP4096 pipeline v8 — direct HTTP download, tokenize, move shards to E:"""
import os, sys, shutil, time, subprocess, urllib.request
from pathlib import Path

LOG_PATH = Path(r"e:\Projects\Proj\golf\records\track_non_record_16mb\2026-04-01_EliteUTv22p8_12StepRecurrence_Windows3090\logs\p8_A1_SP4096_v8.log")
WORK_DIR = Path(r"C:\sp4096_temp")
DEST_DIR = Path(r"e:\Projects\Proj\golf\data")
HF_URL = "https://huggingface.co/datasets/willdepueoai/parameter-golf/resolve/main/datasets/docs_selected.jsonl"

def log(msg: str):
    ts = time.strftime("[%d-%m-%Y %H:%M:%S")
    ts += f".{int(time.time()*100)%100:02d}]"
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        f.write("")

    log("=== SP4096 v8: Pipeline start ===")

    # Clean + create work dir
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    WORK_DIR.mkdir(parents=True)
    os.chdir(str(WORK_DIR))
    log(f"Work dir: {WORK_DIR} (free: {shutil.disk_usage(WORK_DIR).free/1e9:.1f}GB)")

    # Step 1: Direct HTTP download
    datasets_dir = WORK_DIR / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    out_file = datasets_dir / "docs_selected.jsonl"

    log(f"Downloading {HF_URL} ...")
    log("(no progress bar — 48GB over internet, ~10-20 min)")

    start = time.time()
    try:
        urllib.request.urlretrieve(HF_URL, str(out_file))
    except Exception as e:
        log(f"ERROR: Download failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start
    size_mb = out_file.stat().st_size / 1e6
    log(f"Downloaded {size_mb:.0f}MB in {elapsed:.0f}s ({size_mb/elapsed:.1f} MB/s)")

    # Step 2: Copy tokenizer_specs.json
    spec_src = DEST_DIR / "tokenizer_specs.json"
    if spec_src.exists():
        shutil.copy(spec_src, WORK_DIR / "tokenizer_specs.json")
        log("Copied tokenizer_specs.json")
    else:
        log(f"WARNING: tokenizer_specs.json not found at {spec_src}")

    # Step 3: Run tokenization
    script = DEST_DIR / "download_hf_docs_and_tokenize.py"
    if not script.exists():
        log(f"ERROR: {script} not found")
        sys.exit(1)

    log(f"Running: python {script} --output-root {WORK_DIR}")
    result = subprocess.run(
        [sys.executable, str(script), "--output-root", str(WORK_DIR)],
        cwd=str(WORK_DIR),
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max
    )

    if result.returncode != 0:
        log(f"ERROR: Tokenization failed (code {result.returncode})")
        log(f"STDERR: {result.stderr[-3000:]}")
        log(f"STDOUT: {result.stdout[-3000:]}")
        sys.exit(result.returncode)

    log(f"Tokenization succeeded. Output (last 2k chars): {result.stdout[-2000:]}")

    # Step 4: Move shards to E:
    shard_dir = WORK_DIR / "fineweb10B_sp4096"
    if shard_dir.exists():
        dest_shard = DEST_DIR / "fineweb10B_sp4096"
        if dest_shard.exists():
            shutil.rmtree(dest_shard)
        shutil.move(str(shard_dir), str(dest_shard))
        shard_size = sum(f.stat().st_size for f in dest_shard.rglob("*.bin")) / 1e6
        log(f"Moved fineweb10B_sp4096 to E: ({shard_size:.0f}MB)")

        # Also copy manifest/snapshot files if they were generated
        for fname in ["manifest.json", "snapshot_meta.json"]:
            src = WORK_DIR / fname
            if src.exists():
                shutil.copy(src, DEST_DIR / fname)
                log(f"Copied {fname}")
    else:
        log("WARNING: fineweb10B_sp4096 not found in work dir!")

    # Step 5: Copy tokenizer model files
    for fname in ["fineweb_4096_bpe.model", "fineweb_4096_bpe.vocab"]:
        src = WORK_DIR / "tokenizers" / fname
        if not src.exists():
            src = WORK_DIR / fname
        if src.exists():
            dest_tokenizers = DEST_DIR / "tokenizers"
            dest_tokenizers.mkdir(exist_ok=True)
            shutil.copy(src, dest_tokenizers / fname)
            log(f"Copied {fname}")

    # Step 6: Clean up
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    log("=== SP4096 v8: Pipeline complete! ===")
    log(f"E: drive free: {shutil.disk_usage(DEST_DIR).free/1e9:.1f}GB")

if __name__ == "__main__":
    main()