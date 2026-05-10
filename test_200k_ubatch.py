#!/usr/bin/env python3
"""Quick OOM test: 200k micro-batch (no grad accum) + encoder-only ckpt."""
import os, sys

os.environ.setdefault("SAFETY_CLAMP_DISABLE", "1")
os.environ.setdefault("MODEL_TYPE", "multilayer")
os.environ.setdefault("NUM_LAYERS", "5")
os.environ.setdefault("MODEL_DIM", "512")
os.environ.setdefault("NUM_HEADS", "8")
os.environ.setdefault("NUM_KV_HEADS", "4")
os.environ.setdefault("MLP_MULT", "2")
os.environ.setdefault("RECURRENCE_STEPS", "2")
os.environ.setdefault("MULTILAYER_LORA_RANK", "8")
os.environ.setdefault("MICRO_BATCH_TOKENS", "204800")
os.environ.setdefault("TRAIN_BATCH_TOKENS", "204800")
os.environ.setdefault("TRAIN_SEQ_LEN", "1024")
os.environ.setdefault("OPTIM_MODE", "muon_adam")
os.environ.setdefault("MATRIX_OPTIM", "muon")
os.environ.setdefault("MATRIX_LR", "0.12")
os.environ.setdefault("SCALAR_LR", "0.03")
os.environ.setdefault("BETA2", "0.92")
os.environ.setdefault("MUON_MOMENTUM", "0.95")
os.environ.setdefault("MUON_BACKEND_STEPS", "5")
os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("MULTILAYER_ACTIVATION_CHECKPOINT", "1")
os.environ.setdefault("MULTILAYER_ACTIVATION_CHECKPOINT_MODE", "encoder")
os.environ.setdefault("SCHEDULE_FREE", "1")
os.environ.setdefault("SAVE_BEST_CHECKPOINT", "0")
os.environ.setdefault("QUANT_EVAL", "0")
os.environ.setdefault("MAX_WALLCLOCK_SECONDS", "60")
os.environ.setdefault("ITERATIONS", "10")
os.environ.setdefault("DATA_DETERMINISTIC", "1")
os.environ.setdefault("DATA_SEED", "3623123517")
os.environ.setdefault("BIGRAM_HASH_ENABLED", "1")
os.environ.setdefault("SHELL_CENTERING_ENABLED", "1")
os.environ.setdefault("TIE_EMBEDDINGS", "1")
os.environ.setdefault("DROPOUT_P", "0.4")
os.environ.setdefault("LABEL_SMOOTHING", "0.15")
os.environ.setdefault("VOCAB_SIZE", "1024")
os.environ.setdefault("WARMUP_STEPS", "10")
os.environ.setdefault("TRAIN_LOG_EVERY", "10")

os.environ["RUN_ID"] = "200k_ubatch_test"

# Ensure data paths are correct
if "DATA_PATH" not in os.environ:
    os.environ["DATA_PATH"] = "/mnt/e/Projects/Proj/golf/data/datasets/fineweb10B_sp1024"
if "TOKENIZER_PATH" not in os.environ:
    os.environ["TOKENIZER_PATH"] = "/mnt/e/Projects/Proj/golf/data/tokenizers/fineweb_1024_bpe.model"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gpt
train_gpt.main()