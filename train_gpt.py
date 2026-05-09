from __future__ import annotations
import io
import os
import shutil
import time
import uuid
import math
import random
import zlib
from collections import deque

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch._inductor.config as inductor_config
from torch.nn.parallel import DistributedDataParallel as DDP
import sentencepiece as spm

# Modular Imports
from model import GPT
from model_multilayer import GPTMultiLayer
from data_utils import DistributedTokenLoader
try:
    from optimizer_utils import Muon, ShampooLite, Lion
except ImportError:
    from optimizer_utils import Muon
    ShampooLite = None
    Lion = None
from eval_utils import eval_val, build_sentencepiece_luts, load_validation_tokens
from quant_utils import quantize_state_dict_int8, dequantize_state_dict_int8

# --- LOGGING ---
_LOG_FILE: str | None = None
_MASTER_PROCESS: bool = True

def log0(msg: str, console: bool = True) -> None:
    if not _MASTER_PROCESS:
        return
    if console:
        print(msg)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            print(msg, file=f)


# --- LOSS FILTER ---
class LossFilter:
    """Per-micro-batch z-score filter on loss deltas to skip pathological batches."""
    def __init__(self, warmup: int, delta_window: int = 20, z_threshold: float = 4.5,
                 stability_window: int = 20, max_recent_drop: float = 0.02,
                 max_retries: int = 8):
        self.warmup = warmup
        self.delta_window = delta_window
        self.z_threshold = z_threshold
        self.stability_window = stability_window
        self.max_recent_drop = max_recent_drop
        self.max_retries = max_retries
        self._history: deque[float] = deque(maxlen=max(delta_window, stability_window) + 2)
        self.global_step = 0
        self.accepted = 0
        self.skipped = 0
        self.fallback_accepts = 0

    def should_skip(self, loss: float) -> bool:
        self.global_step += 1
        if self.global_step <= self.warmup or len(self._history) < 4:
            self._history.append(loss)
            self.accepted += 1
            return False

        history = list(self._history)
        recent = history[-self.delta_window:]
        if len(recent) >= 2:
            deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            new_delta = loss - history[-1]
            mean_d = sum(deltas) / len(deltas)
            var_d = sum((d - mean_d) ** 2 for d in deltas) / len(deltas)
            std_d = math.sqrt(var_d) + 1e-8
            z = (new_delta - mean_d) / std_d
            if z > self.z_threshold:
                self.skipped += 1
                return True

        recent_losses = history[-self.stability_window:]
        if recent_losses:
            min_recent = min(recent_losses)
            if loss > min_recent * (1.0 + self.max_recent_drop * 10):
                self.skipped += 1
                return True

        self._history.append(loss)
        self.accepted += 1
        return False

    def force_accept(self, loss: float) -> None:
        self._history.append(loss)
        self.accepted += 1
        self.fallback_accepts += 1

    def summary(self) -> str:
        return f"accepted={self.accepted} skipped={self.skipped} fallback_accepts={self.fallback_accepts}"


# --- HYPERPARAMETERS ---
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "../../../data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "../../../data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 65_536))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 100))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 1))

    iterations = int(os.environ.get("ITERATIONS", 500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 16))
    cosine_min = float(os.environ.get("COSINE_MIN", "0.1"))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))  # default competition standard
    micro_batch_tokens = int(os.environ.get("MICRO_BATCH_TOKENS", 32_768))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))

    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 1024))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 5))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 10.0))

    model_type = os.environ.get("MODEL_TYPE", "recurrent")

    num_steps = int(os.environ.get("RECURRENCE_STEPS", os.environ.get("NUM_STEPS", 1)))
    num_layers = int(os.environ.get("NUM_LAYERS", "11"))
    lora_rank = int(os.environ.get("LORA_RANK", 512))
    lora_scope = os.environ.get("LORA_SCOPE", "q")
    # Multilayer LoRA: separate env var so we can tune independently from recurrent LoRA.
    # Default 0 = disabled. Set to e.g. 8 to enable rank-8 LoRA on all attention projections.
    multilayer_lora_rank = int(os.environ.get("MULTILAYER_LORA_RANK", "0"))

    # Feature toggles
    bigram_hash_enabled = bool(int(os.environ.get("BIGRAM_HASH_ENABLED", "0")))
    bigram_hash_size = int(os.environ.get("BIGRAM_HASH_SIZE", 2048))
    bigram_hash_scale = float(os.environ.get("BIGRAM_HASH_SCALE", "0.05"))
    level_signal_enabled = bool(int(os.environ.get("LEVEL_SIGNAL_ENABLED", "0")))
    level_signal_rank = int(os.environ.get("LEVEL_SIGNAL_RANK", "0")) or None
    shell_centering_enabled = bool(int(os.environ.get("SHELL_CENTERING_ENABLED", "0")))
    shell_centering_lam = float(os.environ.get("SHELL_CENTERING_LAM", "0.008"))
    parallel_residual = bool(int(os.environ.get("PARALLEL_RESIDUAL", "0")))
    dropout_p = float(os.environ.get("DROPOUT_P", "0.15"))
    label_smoothing = float(os.environ.get("LABEL_SMOOTHING", "0.05"))
    recurrent_attn_every = int(os.environ.get("RECURRENT_ATTN_EVERY", "1"))
    recurrent_refine_mlp_ratio = float(os.environ.get("RECURRENT_REFINE_MLP_RATIO", "1.0"))
    recurrent_attend_last = bool(int(os.environ.get("RECURRENT_ATTEND_LAST", "1")))
    schedule_free = bool(int(os.environ.get("SCHEDULE_FREE", "0")))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", "0.0004"))

    # Curriculum
    seq_len_curriculum = bool(int(os.environ.get("SEQ_LEN_CURRICULUM", "0")))
    short_train_seq_len = int(os.environ.get("SHORT_TRAIN_SEQ_LEN", "128"))
    seq_len_curriculum_steps = int(os.environ.get("SEQ_LEN_CURRICULUM_STEPS", "20"))
    recurrence_curriculum = bool(int(os.environ.get("RECURRENCE_CURRICULUM", "0")))

    # Optimizer LRs and WDs
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.08))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.015))
    lora_lr = float(os.environ.get("LORA_LR", os.environ.get("SCALAR_LR", 0.015)))
    control_lr = float(os.environ.get("CONTROL_LR", os.environ.get("SCALAR_LR", 0.015)))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    embed_lr = float(os.environ.get("EMBED_LR", 0.7))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.06))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    scalar_weight_decay = float(os.environ.get("SCALAR_WEIGHT_DECAY", "0.1"))
    lora_weight_decay = float(os.environ.get("LORA_WEIGHT_DECAY", "0.0"))
    control_weight_decay = float(os.environ.get("CONTROL_WEIGHT_DECAY", "0.0"))

    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    dynamic_lr_norm = bool(int(os.environ.get("DYNAMIC_LR_NORM", "0")))
    target_grad_norm = float(os.environ.get("TARGET_GRAD_NORM", "0.5"))

    # Checkpoint / export
    save_best_checkpoint = bool(int(os.environ.get("SAVE_BEST_CHECKPOINT", "1")))
    save_best_int8 = bool(int(os.environ.get("SAVE_BEST_INT8", "1")))
    export_best_checkpoint = bool(int(os.environ.get("EXPORT_BEST_CHECKPOINT", "1")))
    quant_eval = bool(int(os.environ.get("QUANT_EVAL", "1")))
    quant_eval_max_steps = int(os.environ.get("QUANT_EVAL_MAX_STEPS", "50"))
    quant_eval_stride = int(os.environ.get("QUANT_EVAL_STRIDE", "64"))

    # Loss filter
    loss_filter_enabled = bool(int(os.environ.get("LOSS_FILTER_ENABLED", "0")))
    loss_filter_warmup = int(os.environ.get("LOSS_FILTER_WARMUP", "250"))
    loss_filter_delta_window = int(os.environ.get("LOSS_FILTER_DELTA_WINDOW", "20"))
    loss_filter_z = float(os.environ.get("LOSS_FILTER_Z_THRESHOLD", "4.5"))
    loss_filter_stability_window = int(os.environ.get("LOSS_FILTER_STABILITY_WINDOW", "20"))
    loss_filter_max_recent_drop = float(os.environ.get("LOSS_FILTER_MAX_RECENT_DROP", "0.02"))
    loss_filter_max_retries = int(os.environ.get("LOSS_FILTER_MAX_RETRIES", "8"))
    data_deterministic = bool(int(os.environ.get("DATA_DETERMINISTIC", "0")))
    data_seed = int(os.environ.get("DATA_SEED", "0")) or None


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
    "q_gain,skip_weight,skip_weights,v_step_bias,level_gain"
).split(",")

LORA_TENSOR_NAME_PATTERNS = ("lora_A,lora_B").split(",")


def _vram_stats(device: torch.device) -> dict[str, float]:
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    resv = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_resv = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    return dict(alloc_gib=alloc, resv_gib=resv, peak_alloc_gib=peak_alloc, peak_resv_gib=peak_resv)


def _vram_str(stats: dict[str, float]) -> str:
    return (f"vram_alloc_gib={stats['alloc_gib']:.2f} vram_resv_gib={stats['resv_gib']:.2f} "
            f"vram_peak_alloc_gib={stats['peak_alloc_gib']:.2f} vram_peak_resv_gib={stats['peak_resv_gib']:.2f}")


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


def _count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    lora = sum(p.numel() for n, p in model.named_parameters()
               if any(pat in n for pat in LORA_TENSOR_NAME_PATTERNS))
    return total, lora


def main() -> None:
    print("[debug] main() started")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ and os.environ.get("FORCE_SINGLE_GPU") != "1"
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, args.train_batch_tokens // (args.micro_batch_tokens * world_size))
    model_type = args.model_type.strip()

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
    master_process = rank == 0

    # VRAM safety: multilayer U-Net with recurrence stores intermediate tensors
    # that can 3-4× the micro-batch memory. Clamp micro-batch based on actual GPU memory.
    _gpu_mem_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if model_type in ("multilayer", "stage_repeat"):
        if _gpu_mem_gib >= 30:       # 5090 (32GB) or A6000 (48GB)
            _max_safe_ubatch = 262144
        elif _gpu_mem_gib >= 22:     # 3090/4090 (24GB)
            _max_safe_ubatch = 65536
        else:                         # lower VRAM GPUs
            _max_safe_ubatch = 32768
    else:
        _max_safe_ubatch = 524288
    if args.micro_batch_tokens > _max_safe_ubatch:
        args.micro_batch_tokens = _max_safe_ubatch
        grad_accum_steps = max(1, args.train_batch_tokens // (args.micro_batch_tokens * world_size))
        log0(f"[vram] clamped MICRO_BATCH_TOKENS={args.micro_batch_tokens} grad_accum_steps={grad_accum_steps} (GPU={_gpu_mem_gib:.0f}GiB, max_safe={_max_safe_ubatch})")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # --- Linux/Ampere+ optimisations ---
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    if os.environ.get("ENABLE_RECURRENT_TRAIN_COMPILE", "0") == "1":
        try:
            # Keep CUDA graphs DISABLED for training — the compiled 12-step
            # unrolled forward shares graph-owned intermediate tensors across
            # micro-batches, which is unsafe under grad accumulation.
            # Speed comes from Inductor kernel fusion (not graph replay).
            if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                inductor_config.triton.cudagraphs = False
        except Exception:
            pass

    logfile = f"logs/{args.run_id}.txt" if master_process else None
    global _LOG_FILE, _MASTER_PROCESS
    _MASTER_PROCESS, _LOG_FILE = master_process, logfile
    if master_process:
        os.makedirs("logs", exist_ok=True)
    print("[debug] logging initialized")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- Config dump ---
    if master_process:
        cuda_ver = torch.version.cuda or "?"
        log0(f"[config] --------------------------------------------------")
        log0(f"[config] run_id={args.run_id} seed={args.seed}")
        log0(f"[config] torch={torch.__version__} cuda={cuda_ver} device={torch.cuda.get_device_name(device)}")
        free, total_mem = torch.cuda.mem_get_info(device)
        log0(f"[vram] gpu_mem_total_gib={total_mem/(1024**3):.2f} gpu_mem_free_gib={free/(1024**3):.2f} (before init)")
        log0(f"[config] distributed={distributed} world_size={world_size} rank={rank} local_rank={local_rank}")
        log0(f"[config] stop: iterations={args.iterations} max_wallclock_seconds={args.max_wallclock_seconds}")
        log0(f"[config] schedule: warmup_steps={args.warmup_steps} cosine_min={args.cosine_min}")
        log0(f"[config] batch: train_batch_tokens={args.train_batch_tokens} micro_batch_tokens={args.micro_batch_tokens} grad_accum_steps={grad_accum_steps}")
        log0(f"[config] data: TRAIN_SEQ_LEN(eval)={args.train_seq_len} VAL_BATCH_SIZE={args.val_batch_size} VAL_LOSS_EVERY={args.val_loss_every}")
        log0(f"[config] model: dim={args.model_dim} heads={args.num_heads} kv_heads={args.num_kv_heads} mlp_mult={args.mlp_mult} steps={args.num_steps} lora_rank={args.lora_rank} lora_scope={args.lora_scope}")
        log0(f"[config] recurrent_speed: ATTN_EVERY={args.recurrent_attn_every} REFINE_MLP_RATIO={args.recurrent_refine_mlp_ratio} ATTEND_LAST={int(args.recurrent_attend_last)} DROPOUT_P={args.dropout_p} LABEL_SMOOTHING={args.label_smoothing}")
        log0(f"[config] features: BIGRAM_HASH={int(args.bigram_hash_enabled)} BIGRAM_HASH_SIZE={args.bigram_hash_size} BIGRAM_HASH_SCALE={args.bigram_hash_scale} LEVEL_SIGNAL={int(args.level_signal_enabled)} LEVEL_SIGNAL_RANK={args.level_signal_rank or 0} SHELL_CENTERING={int(args.shell_centering_enabled)} SHELL_CENTERING_LAM={args.shell_centering_lam}")
        log0(f"[config] optim: MATRIX_LR={args.matrix_lr} MUON_BACKEND_STEPS={args.muon_backend_steps} MUON_MOMENTUM={args.muon_momentum} SCALAR_LR={args.scalar_lr}")
        log0(f"[config] optim_groups: LORA_LR={args.lora_lr} CONTROL_LR={args.control_lr} SCALAR_WD={args.scalar_weight_decay} LORA_WD={args.lora_weight_decay} CONTROL_WD={args.control_weight_decay}")
        log0(f"[config] clip: GRAD_CLIP_NORM={args.grad_clip_norm} DYNAMIC_LR_NORM={int(args.dynamic_lr_norm)} TARGET_GRAD_NORM={args.target_grad_norm}")
        log0(f"[config] curriculum: SEQ_LEN_CURRICULUM={int(args.seq_len_curriculum)} SHORT_TRAIN_SEQ_LEN={args.short_train_seq_len} SEQ_LEN_CURRICULUM_STEPS={args.seq_len_curriculum_steps}")
        log0(f"[config] curriculum: RECURRENCE_CURRICULUM={int(args.recurrence_curriculum)}")
        log0(f"[config] ttt: TTT_ENABLED={int(args.ttt_enabled)} TTT_LR={args.ttt_lr}")
        log0(f"[config] schedule_free: SCHEDULE_FREE={int(args.schedule_free)}")
        log0(f"[config] checkpoint: SAVE_BEST_CHECKPOINT={int(args.save_best_checkpoint)} SAVE_BEST_INT8={int(args.save_best_int8)} EXPORT_BEST_CHECKPOINT={int(args.export_best_checkpoint)}")
        log0(f"[config] quant_eval: QUANT_EVAL={int(args.quant_eval)} QUANT_EVAL_MAX_STEPS={args.quant_eval_max_steps} QUANT_EVAL_STRIDE={args.quant_eval_stride}")
        log0(f"[config] loss_filter: enabled={int(args.loss_filter_enabled)} warmup={args.loss_filter_warmup} delta_window={args.loss_filter_delta_window} z={args.loss_filter_z} stability_window={args.loss_filter_stability_window} max_recent_drop={args.loss_filter_max_recent_drop} max_retries_per_step={args.loss_filter_max_retries} allow_distributed=0")
        log0(f"[config] --------------------------------------------------")

    print("[debug] loading validation tokens...")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    model_type = args.model_type.strip()
    print(f"[debug] initializing base_model... MODEL_TYPE='{model_type}'")

    if model_type == "multilayer":
        base_model = GPTMultiLayer(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            num_steps=args.num_steps,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            bigram_hash_size=args.bigram_hash_size,
            bigram_hash_scale=args.bigram_hash_scale,
            lora_rank=args.multilayer_lora_rank,
        ).to(device).bfloat16()
        # torch.compile the forward pass for speed (compile-safe Rotary precomputes RoPE)
        # Use 'default' mode: enables Inductor kernel fusion without CUDA graph
        # caching, which is unsafe across micro-batches with our multi-block U-Net.
        if os.environ.get("DISABLE_COMPILE") != "1":
            compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")
            base_model.forward_logits = torch.compile(base_model.forward_logits, mode=compile_mode)
    elif model_type == "stage_repeat":
        # Relaxed recursive layers: NUM_STAGES blocks, each repeated REPEATS_PER_STAGE times.
        # Per-repeat adapters: LoRA (mlp up/down + attn out), residual gates, repeat embeddings.
        num_stages = int(os.environ.get("NUM_STAGES", "5"))
        repeats_per_stage = int(os.environ.get("REPEATS_PER_STAGE", "2"))
        repeat_lora_rank = int(os.environ.get("REPEAT_LORA_RANK", "32"))
        # Defaults chosen for speed: norm offsets + attention LoRA are expensive.
        norm_offset_enabled = os.environ.get("REPEAT_NORM_OFFSET", "0") != "0"
        attn_out_lora = os.environ.get("REPEAT_ATTN_OUT_LORA", "0") != "0"
        mlp_lora_enabled = os.environ.get("REPEAT_MLP_LORA", "1") != "0"
        repeat_attn_every = int(os.environ.get("REPEAT_ATTN_EVERY", "1"))
        repeat_refine_mlp_ratio = float(os.environ.get("REPEAT_REFINE_MLP_RATIO", "1.0"))

        base_model = GPTStageRepeat(
            vocab_size=args.vocab_size,
            num_stages=num_stages,
            repeats_per_stage=repeats_per_stage,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            repeat_lora_rank=repeat_lora_rank,
            bigram_hash_enabled=args.bigram_hash_enabled,
            bigram_hash_size=args.bigram_hash_size,
            bigram_hash_scale=args.bigram_hash_scale,
            shell_centering_enabled=args.shell_centering_enabled,
            shell_centering_lam=args.shell_centering_lam,
            dropout_p=args.dropout_p,
            label_smoothing=args.label_smoothing,
            norm_offset_enabled=norm_offset_enabled,
            attn_out_lora=attn_out_lora,
            mlp_lora_enabled=mlp_lora_enabled,
            repeat_attn_every=repeat_attn_every,
            repeat_refine_mlp_ratio=repeat_refine_mlp_ratio,
        ).to(device).bfloat16()

        log0(
            f"[config] stage_repeat: NUM_STAGES={num_stages} REPEATS_PER_STAGE={repeats_per_stage} "
            f"REPEAT_LORA_RANK={repeat_lora_rank} REPEAT_NORM_OFFSET={int(norm_offset_enabled)} "
            f"REPEAT_ATTN_OUT_LORA={int(attn_out_lora)} REPEAT_MLP_LORA={int(mlp_lora_enabled)} "
            f"REPEAT_ATTN_EVERY={repeat_attn_every} REPEAT_REFINE_MLP_RATIO={repeat_refine_mlp_ratio}"
        )

    else:
        base_model = GPT(
            vocab_size=args.vocab_size,
            num_steps=args.num_steps,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            lora_rank=args.lora_rank,
            lora_scope=args.lora_scope,
            bigram_hash_enabled=args.bigram_hash_enabled,
            bigram_hash_size=args.bigram_hash_size,
            bigram_hash_scale=args.bigram_hash_scale,
            level_signal_enabled=args.level_signal_enabled,
            level_rank=args.level_signal_rank,
            shell_centering_enabled=args.shell_centering_enabled,
            shell_centering_lam=args.shell_centering_lam,
            parallel_residual=args.parallel_residual,
            dropout_p=args.dropout_p,
            label_smoothing=args.label_smoothing,
            recurrent_attn_every=args.recurrent_attn_every,
            recurrent_refine_mlp_ratio=args.recurrent_refine_mlp_ratio,
            recurrent_attend_last=args.recurrent_attend_last,
        ).to(device).bfloat16()

    total_params, lora_params = _count_params(base_model)
    lora_frac = lora_params / total_params if total_params > 0 else 0.0
    log0(f"[config] vocab_size={args.vocab_size} embed_params={args.vocab_size * args.model_dim}")
    log0(f"[params] total={total_params:,} lora={lora_params:,} lora_frac={lora_frac:.4f}")

    model = base_model
    if distributed:
        print("[debug] wrapping in DDP...")
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    # --- Optimizer param splitting: matrix | lora | control | scalar | embed ---
    print("[debug] splitting params for optimizers...")
    matrix_params, lora_params_list, control_params, scalar_params = [], [], [], []
    for name, p in base_model.named_parameters():
        if "tok_emb" in name or (base_model.lm_head is not None and "lm_head" in name):
            continue
        is_lora = any(pat in name for pat in LORA_TENSOR_NAME_PATTERNS)
        is_control = any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
        is_matrix = p.ndim == 2 and not is_lora and not is_control and "step_embeddings" not in name and "bigram_hash" not in name
        if is_lora:
            lora_params_list.append(p)
        elif is_control:
            control_params.append(p)
        elif is_matrix:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    log0(f"[optim_split] matrix={len(matrix_params)} scalar={len(scalar_params)} lora={len(lora_params_list)} control={len(control_params)} (lr: scalar={args.scalar_lr} lora={args.lora_lr} control={args.control_lr}; wd: scalar={args.scalar_weight_decay} lora={args.lora_weight_decay} control={args.control_weight_decay})")

    print("[debug] initializing optimizers...")
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr

    matrix_optim_mode = os.environ.get("MATRIX_OPTIM", "muon").strip().lower()
    # Only apply Shampoo/SOAP-style preconditioning to *large* 2D matrices.
    # For 512-dim models most matrices are ~262k elems; a threshold like 500k
    # targets the big combined projections (e.g., qkv) without touching every mat.
    shampoo_min_numel = int(os.environ.get("SHAMPOO_MIN_NUMEL", "500000"))
    shampoo_beta2 = float(os.environ.get("SHAMPOO_BETA2", "0.999"))
    shampoo_eps = float(os.environ.get("SHAMPOO_EPS", "1e-8"))
    shampoo_momentum = float(os.environ.get("SHAMPOO_MOMENTUM", "0.0"))
    # ShampooLite update magnitudes are on a different scale than Muon.
    # If you don't set this, we default to a conservative fraction of MATRIX_LR.
    shampoo_lr = float(os.environ.get("SHAMPOO_LR", str(args.matrix_lr * 0.1)))

    matrix_optimizers: list[torch.optim.Optimizer] = []
    if matrix_optim_mode in {"shampoo_lite", "hybrid"}:
        shampoo_params = [p for p in matrix_params if p.numel() >= shampoo_min_numel]
        muon_params = [p for p in matrix_params if p.numel() < shampoo_min_numel]

        if shampoo_params:
            shampoo_opt = (
                ShampooLite(
                    shampoo_params,
                    lr=shampoo_lr,
                    beta2=shampoo_beta2,
                    eps=shampoo_eps,
                    momentum=shampoo_momentum,
                    grad_clip=args.grad_clip_norm,
                    min_numel=0,
                )
            )
            # Tag for LR scheduling (so we can scale Muon and Shampoo with different base LRs)
            for g in shampoo_opt.param_groups:
                g["_is_shampoo"] = True
                g["_target_lr"] = shampoo_lr
            matrix_optimizers.append(shampoo_opt)
            print(f"[optim] matrix_opt=shampoo_lite mats={len(shampoo_params)} (min_numel={shampoo_min_numel} lr={shampoo_lr} beta2={shampoo_beta2} mom={shampoo_momentum})")
        if matrix_optim_mode == "hybrid" and muon_params:
            muon_opt = (
                Muon(muon_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                     backend_steps=args.muon_backend_steps)
            )
            for g in muon_opt.param_groups:
                g["_is_shampoo"] = False
                g["_target_lr"] = args.matrix_lr
            matrix_optimizers.append(muon_opt)
            print(f"[optim] matrix_opt=hybrid + muon mats={len(muon_params)}")

        if matrix_optim_mode == "shampoo_lite" and not shampoo_params:
            # Fallback: if threshold filtered everything out, use Muon.
            matrix_optimizers.append(
                Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                     backend_steps=args.muon_backend_steps)
            )
            print(f"[optim] matrix_opt=shampoo_lite (no mats >= min_numel={shampoo_min_numel}; fallback to muon)")
    else:
        muon_opt = (
            Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                 backend_steps=args.muon_backend_steps)
        )
        for g in muon_opt.param_groups:
            g["_is_shampoo"] = False
            g["_target_lr"] = args.matrix_lr
        matrix_optimizers.append(muon_opt)

    # Merged AdamW — single optimizer with per-group LR/WD settings.
    # Eliminates 4× kernel launch scheduling gaps vs separate instances.
    adam_groups = []
    # tok_emb group
    adam_groups.append({"params": [base_model.tok_emb.weight],
                        "lr": token_lr, "target_lr": token_lr,
                        "weight_decay": 0.0})
    # lora group
    if lora_params_list:
        adam_groups.append({"params": lora_params_list,
                            "lr": args.lora_lr, "target_lr": args.lora_lr,
                            "weight_decay": args.lora_weight_decay})
    # control group (scales, biases)
    if control_params:
        adam_groups.append({"params": control_params,
                            "lr": args.control_lr, "target_lr": args.control_lr,
                            "weight_decay": args.control_weight_decay})
    # scalar group (step_embeddings etc.)
    if scalar_params:
        adam_groups.append({"params": scalar_params,
                            "lr": args.scalar_lr, "target_lr": args.scalar_lr,
                            "weight_decay": args.scalar_weight_decay})
    # lm_head group (if separate from tied emb)
    if base_model.lm_head is not None:
        adam_groups.append({"params": [base_model.lm_head.weight],
                            "lr": args.head_lr, "target_lr": args.head_lr,
                            "weight_decay": 0.1})

    optimizer_adam = torch.optim.AdamW(
        adam_groups,
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )

    # Optimizer mode: controls which optimizers to use
    # "muon_adam" = Muon + AdamW every step (default/baseline)
    # "muon_alt" = Muon every 2 steps + AdamW every step
    # "adam_only" = Fused AdamW only, no Muon
    # "lion" = Lion optimizer for *all* params (matrix+non-matrix)
    # "muon_lion" = Muon for matrix params + Lion for non-matrix params
    optim_mode = os.environ.get("OPTIM_MODE", "muon_adam")
    
    if optim_mode == "lion":
        # Lion: single optimizer with sign-based updates + momentum
        # Combines matrix + scalar + lora + control into one Lion instance
        lion_groups = []
        for g in adam_groups:
            lion_groups.append(dict(
                params=g["params"], lr=g["lr"], target_lr=g["target_lr"],
                weight_decay=g.get("weight_decay", 0.0),
            ))
        if matrix_params:
            lion_groups.append(dict(
                params=matrix_params, lr=args.matrix_lr, target_lr=args.matrix_lr,
                weight_decay=0.0,
            ))
        optimizer_lion = Lion(
            lion_groups,
            lr=args.matrix_lr, betas=(0.9, 0.99), weight_decay=0.0,
        )
        optimizers = [optimizer_lion]
        print("[optim] mode=lion (single Lion optimizer)")
    elif optim_mode == "muon_lion":
        # Muon for matrix params (already in matrix_optimizers), Lion for everything else.
        lion_groups = []
        for g in adam_groups:
            lion_groups.append(dict(
                params=g["params"], lr=g["lr"], target_lr=g["target_lr"],
                weight_decay=g.get("weight_decay", 0.0),
            ))
        optimizer_lion_nonmatrix = Lion(
            lion_groups,
            lr=args.scalar_lr, betas=(0.9, 0.99), weight_decay=0.0,
        )
        optimizers = [*matrix_optimizers, optimizer_lion_nonmatrix]
        print("[optim] mode=muon_lion (Muon matrices + Lion non-matrices)")
    else:
        optimizers = [*matrix_optimizers, optimizer_adam]
        if optim_mode == "muon_alt":
            print("[optim] mode=muon_alt (Muon every 2 steps)")
        elif optim_mode == "adam_only":
            print("[optim] mode=adam_only (no Muon)")
        else:
            print("[optim] mode=muon_adam (Muon + AdamW every step)")

    # Loss filter
    loss_filter = LossFilter(
        warmup=args.loss_filter_warmup * grad_accum_steps,
        delta_window=args.loss_filter_delta_window,
        z_threshold=args.loss_filter_z,
        stability_window=args.loss_filter_stability_window,
        max_recent_drop=args.loss_filter_max_recent_drop,
        max_retries=args.loss_filter_max_retries,
    ) if args.loss_filter_enabled else None

    print("[debug] initializing data loader...")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    training_time_ms = 0.0
    t0 = time.perf_counter()
    step = 0

    print("[debug] initializing model EMA...")
    model_ema = {n.replace("module.", ""): p.clone().detach() for n, p in model.named_parameters()}

    # Best checkpoint tracking
    best_val_bpb = float("inf")
    best_val_loss = float("inf")
    best_step = -1
    best_ema: dict | None = None

    print("[debug] entering training loop...")
    global_start_time = time.perf_counter()

    while True:
        elapsed_sec = time.perf_counter() - global_start_time
        last_step = step >= args.iterations or elapsed_sec >= args.max_wallclock_seconds

        if step > 0 and args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            torch.cuda.synchronize()
            vram_pre = _vram_stats(device)
            log0(f"[vram_eval] pre_swap step={step} alloc_gib={vram_pre['alloc_gib']:.2f} resv_gib={vram_pre['resv_gib']:.2f} peak_alloc_gib={vram_pre['peak_alloc_gib']:.2f} peak_resv_gib={vram_pre['peak_resv_gib']:.2f}")

            # Zero-copy EMA swap: swap .data references instead of cloning
            ema_swap = {}
            for n, p in base_model.named_parameters():
                if n in model_ema:
                    ema_swap[n] = p.data
                    p.data = model_ema[n]

            vram_post_swap = _vram_stats(device)
            log0(f"[vram_eval] post_swap step={step} alloc_gib={vram_post_swap['alloc_gib']:.2f} resv_gib={vram_post_swap['resv_gib']:.2f} peak_alloc_gib={vram_post_swap['peak_alloc_gib']:.2f} peak_resv_gib={vram_post_swap['peak_resv_gib']:.2f}")

            eval_stride = args.quant_eval_stride if last_step else args.train_seq_len
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                         val_tokens, base_bytes_lut, has_leading_space_lut,
                                         is_boundary_token_lut, max_steps=50, stride=eval_stride,
                                         ttt_lr=args.ttt_lr)

            # Restore trained weights by swapping .data references back
            for n, p in base_model.named_parameters():
                if n in ema_swap:
                    p.data = ema_swap[n]

            vram_post_restore = _vram_stats(device)
            log0(f"[vram_eval] post_restore step={step} alloc_gib={vram_post_restore['alloc_gib']:.2f} resv_gib={vram_post_restore['resv_gib']:.2f} peak_alloc_gib={vram_post_restore['peak_alloc_gib']:.2f} peak_resv_gib={vram_post_restore['peak_resv_gib']:.2f}")

            final_tag = " [FINAL STRIDE 64]" if last_step else ""
            log0(f"step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms{final_tag} {_vram_str(vram_post_restore)}")

            # Best checkpoint tracking
            if args.save_best_checkpoint and val_bpb < best_val_bpb:
                prev_bpb = best_val_bpb
                best_val_bpb = val_bpb
                best_val_loss = val_loss
                best_step = step
                best_ema = {n: p.clone() for n, p in model_ema.items()}

                # Save best_model.pt using current EMA weights
                original_params2 = {n: p.data.clone() for n, p in base_model.named_parameters()}
                for n, p in base_model.named_parameters():
                    if n in model_ema:
                        p.data.copy_(model_ema[n])
                sd = base_model.state_dict()
                sz_bytes = sum(v.numel() * v.element_size() for v in sd.values())
                torch.save(sd, "best_model.pt.tmp")
                try:
                    os.remove("best_model.pt")
                except FileNotFoundError:
                    pass
                shutil.move("best_model.pt.tmp", "best_model.pt")
                sz_mib = sz_bytes / (1024 ** 2)
                log0(f"[best] new_best step={step} val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} (prev={prev_bpb:.4f}) saved=best_model.pt ({sz_mib:.2f} MiB)")

                if args.save_best_int8:
                    int8_sd, _ = quantize_state_dict_int8(sd)
                    buf = io.BytesIO()
                    torch.save(int8_sd, buf)
                    payload = zlib.compress(buf.getvalue(), level=9)
                    with open("best_model.int8.ptz", "wb") as f:
                        f.write(payload)
                    int8_mib = len(payload) / (1024 ** 2)
                    log0(f"[best] int8_saved=best_model.int8.ptz ({int8_mib:.2f} MiB) int8_payload_bytes={len(payload)} baseline_bytes={sz_bytes}")

                for n, p in base_model.named_parameters():
                    p.data.copy_(original_params2[n])

            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            reason = "iterations" if step >= args.iterations else "wallclock"
            log0(f"[stop] reason:{reason} step:{step} elapsed:{elapsed_sec:.1f}s (max_wallclock_seconds={args.max_wallclock_seconds}, iterations={args.iterations})")
            break

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        loss_log_scale = 1.0 / grad_accum_steps
        backward_scale = loss_log_scale

        for _ in range(grad_accum_steps):
            # Determine training seq_len (curriculum if enabled)
            if args.seq_len_curriculum and step < args.seq_len_curriculum_steps:
                cur_seq = args.short_train_seq_len
            else:
                cur_seq = args.train_seq_len

            retries = 0
            while True:
                x, y = train_loader.next_batch(args.micro_batch_tokens, cur_seq)
                loss = model(x, y)
                loss_val = loss.item()

                if loss_filter is not None and loss_filter.should_skip(loss_val):
                    retries += 1
                    if retries >= args.loss_filter_max_retries:
                        loss_filter.force_accept(loss_val)
                        (loss * backward_scale).backward()
                        step_loss += loss_val * loss_log_scale
                        break
                    continue
                else:
                    (loss * backward_scale).backward()
                    step_loss += loss_val * loss_log_scale
                    break

        if loss_filter is not None and step % 100 == 0:
            log0(f"[filter] totals {loss_filter.summary()}")

        # LR schedule: linear warmup → plateau → gentle cosine decay
        # CRITICAL FIX: Previous time-based cosine decay collapsed too early.
        # At step 400 (226s into 600s run), LR was at 66% — the model starved.
        # Now: 80% of steps at full LR, then gentle decay over final 20%.
        # This matches the 5090 pattern (200 steps at full LR by step 200).
        if args.schedule_free:
            total_scale = 1.0
        else:
            warmup_frac = min(step / max(args.warmup_steps, 1), 1.0)
            # Estimate total steps for schedule planning (483ms/step → ~1240 steps)
            plateau_steps = max(int(0.75 * 1200), args.warmup_steps * 2)  # ~900 steps at full LR
            if step < plateau_steps:
                decay_frac = 1.0  # plateau — full LR
            else:
                progress = min((step - plateau_steps) / max(1200 - plateau_steps, 1), 1.0)
                decay_frac = args.cosine_min + (1.0 - args.cosine_min) * 0.5 * (
                    1.0 + math.cos(progress * math.pi)
                )
            total_scale = warmup_frac * decay_frac

        # Muon momentum warmup
        frac = min(step / max(args.warmup_steps, 1), 1.0)
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum

        # Optional dynamic gradient norm scaling
        if args.dynamic_lr_norm:
            gnorm = torch.nn.utils.get_total_norm(
                [p.grad for p in model.parameters() if p.grad is not None], norm_type=2
            )
            if gnorm > 0:
                dyn_scale = min(1.0, args.target_grad_norm / (gnorm + 1e-6))
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(dyn_scale)

        # --- No gradient hooks — use clip scaling instead ---
        # The 12-step unrolled graph accumulates gradients 12× on shared
        # weights but only 1× on per-step params (LoRA). Dividing shared
        # grads by 12 before clip slows learning by 12× on those params.
        # Instead: use backward_scale=1.0 (no division) and set
        # clip_norm=3.46 (sqrt(12)) so all params pass through clip
        # at full strength. Clip at 3.46 protects against non-finite grads
        # while preserving the 12× signal on shared weights.
        num_steps_div = 1  # no gradient hook division
        # ---------------------------------------------------------

        for opt in matrix_optimizers:
            for group in opt.param_groups:
                if "momentum" in group:
                    group["momentum"] = muon_momentum
                base_lr = float(group.get("_target_lr", args.matrix_lr))
                group["lr"] = base_lr * total_scale

        grad_clip_val = args.grad_clip_norm

        for group in optimizer_adam.param_groups:
            group["lr"] = group["target_lr"] * total_scale

        if optim_mode == "lion":
            for group in optimizer_lion.param_groups:
                group["lr"] = group["target_lr"] * total_scale
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer_lion.param_groups for p in g["params"] if p.grad is not None],
                max_norm=grad_clip_val,
                error_if_nonfinite=False,
            )
            optimizer_lion.step()

        elif optim_mode == "muon_lion":
            # Step matrix optimizers (Muon/ShampooLite)
            for opt in matrix_optimizers:
                try:
                    opt.step(grad_clip=grad_clip_val)
                except TypeError:
                    opt.step()

            # Lion for non-matrix params
            for group in optimizer_lion_nonmatrix.param_groups:
                group["lr"] = group["target_lr"] * total_scale
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer_lion_nonmatrix.param_groups for p in g["params"] if p.grad is not None],
                max_norm=grad_clip_val,
                error_if_nonfinite=False,
            )
            optimizer_lion_nonmatrix.step()

        elif optim_mode == "adam_only":
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer_adam.param_groups for p in g["params"] if p.grad is not None],
                max_norm=grad_clip_val,
                error_if_nonfinite=False,
            )
            optimizer_adam.step()

        else:  # "muon_adam" or "muon_alt"
            use_muon = (optim_mode == "muon_adam") or (step % 2 == 0)
            if use_muon:
                for opt in matrix_optimizers:
                    try:
                        opt.step(grad_clip=grad_clip_val)
                    except TypeError:
                        opt.step()

            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer_adam.param_groups for p in g["params"] if p.grad is not None],
                max_norm=grad_clip_val,
                error_if_nonfinite=False,
            )
            optimizer_adam.step()

        # EMA removed from hot path — only needed at export time.
        # Compute lazily: update model_ema snapshot once every 100 steps
        # instead of every single step.
        if step % 100 == 0:
            for n, p in model.named_parameters():
                ema_n = n.replace("module.", "")
                if ema_n in model_ema:
                    model_ema[ema_n].data.copy_(p.data)

        dt = (time.perf_counter() - t0) * 1000.0
        training_time_ms += dt
        t0 = time.perf_counter()
        # Print dt to console only (not to disk) for measurement
        print(f"step:{step} loss:{step_loss:.4f} dt:{dt:.2f}ms")

        step += 1

    # --- Final export ---
    log0(f"[final] Starting final export...")
    use_best = args.save_best_checkpoint and best_ema is not None and args.export_best_checkpoint
    if use_best:
        log0(f"[final] Export source: BEST checkpoint step={best_step} best_val_loss={best_val_loss:.4f} best_val_bpb={best_val_bpb:.4f}")
        export_ema = best_ema
    else:
        log0(f"[final] Export source: FINAL EMA weights")
        export_ema = model_ema

    # Swap in export EMA weights
    original_params_final = {n: p.data.clone() for n, p in base_model.named_parameters()}
    for n, p in base_model.named_parameters():
        if n in export_ema:
            p.data.copy_(export_ema[n])
    final_sd = base_model.state_dict()
    sz_bytes = sum(v.numel() * v.element_size() for v in final_sd.values())

    log0(f"[final] Saving raw model to final_model.pt ...")
    torch.save(final_sd, "final_model.pt.tmp")
    os.replace("final_model.pt.tmp", "final_model.pt")
    log0(f"[final] Saved: final_model.pt ({sz_bytes / (1024**2):.2f} MiB)")

    log0(f"[final] Quantizing + zlib-compressing to final_model.int8.ptz ...")
    int8_sd, _ = quantize_state_dict_int8(final_sd)
    buf = io.BytesIO()
    torch.save(int8_sd, buf)
    payload = zlib.compress(buf.getvalue(), level=9)
    with open("final_model.int8.ptz", "wb") as f:
        f.write(payload)
    log0(f"[final] Saved: final_model.int8.ptz ({len(payload)/(1024**2):.2f} MiB) int8_payload_bytes={len(payload)} baseline_bytes={sz_bytes}")

    # Restore original weights before quant_eval
    for n, p in base_model.named_parameters():
        p.data.copy_(original_params_final[n])

    # --- Quant eval: compare FP vs INT8 ---
    if args.quant_eval:
        log0(f"[final][quant_eval] running FP vs INT8(dequantized) validation ...")
        # Swap in export EMA for FP eval
        for n, p in base_model.named_parameters():
            if n in export_ema:
                p.data.copy_(export_ema[n])
        fp_loss, fp_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                   val_tokens, base_bytes_lut, has_leading_space_lut,
                                   is_boundary_token_lut,
                                   max_steps=args.quant_eval_max_steps,
                                   stride=args.quant_eval_stride, ttt_lr=args.ttt_lr)
        # Swap in dequantized INT8 weights
        dq_sd = dequantize_state_dict_int8(int8_sd)
        base_model.load_state_dict(dq_sd, strict=False)
        int8_loss, int8_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                       val_tokens, base_bytes_lut, has_leading_space_lut,
                                       is_boundary_token_lut,
                                       max_steps=args.quant_eval_max_steps,
                                       stride=args.quant_eval_stride, ttt_lr=args.ttt_lr)
        delta_loss = int8_loss - fp_loss
        delta_bpb = int8_bpb - fp_bpb
        pct = (delta_bpb / fp_bpb) * 100.0 if fp_bpb > 0 else 0.0
        sign = "+" if delta_bpb >= 0 else ""
        log0(f"[final][quant_eval] fp_val_loss={fp_loss:.6f} fp_val_bpb={fp_bpb:.6f} | int8_val_loss={int8_loss:.6f} int8_val_bpb={int8_bpb:.6f}")
        log0(f"[final][quant_eval] delta_loss={sign}{delta_loss:.6f} delta_bpb={sign}{delta_bpb:.6f} bpb_degradation_pct={sign}{pct:.3f}%")
        # Restore export EMA weights as final state
        for n, p in base_model.named_parameters():
            if n in export_ema:
                p.data.copy_(export_ema[n])

    log0(f"[final] Export complete.")


if __name__ == "__main__":
    main()