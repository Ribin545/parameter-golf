"""
Windows-compatible training launcher for modular Elite Universal Transformer
========================================================================
Patches model.py and train_gpt.py at runtime to ensure stability on Windows/RTX 3090.
"""

from __future__ import annotations
import sys
import os
import pathlib
import types
import torch
import torch.distributed as dist
import torch.backends.cuda as _tbc
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True

# ---------------------------------------------------------------------------
# Patch 1: SDP backends (Prevent Flash SDP crashes on Windows)
# ---------------------------------------------------------------------------
_tbc.enable_flash_sdp(False)
_tbc.enable_mem_efficient_sdp(True)
_tbc.enable_math_sdp(True)
_tbc.enable_cudnn_sdp(True)
print("[windows] SDP backends set: cudnn=ON  flash=OFF  mem_efficient=ON  math=ON")

# ---------------------------------------------------------------------------
# Patch 2: torch.distributed — swap nccl → gloo
# ---------------------------------------------------------------------------
_orig_init_pg = dist.init_process_group
def _patched_init_pg(backend=None, **kwargs):
    if backend == "nccl":
        print("[windows] dist backend: nccl → gloo")
        backend = "gloo"
    return _orig_init_pg(backend=backend, **kwargs)
dist.init_process_group = _patched_init_pg

# ---------------------------------------------------------------------------
# Module Loading Infrastructure
# ---------------------------------------------------------------------------
_root = pathlib.Path(__file__).parent

def _load_module(name: str, path: pathlib.Path) -> types.ModuleType:
    source = path.read_text(encoding="utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__path__ = [str(path.parent)]
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module

# Load supporting modules first
print("[windows] Loading supporting modules...")
_load_module("optimizer_utils", _root / "optimizer_utils.py")
_load_module("data_utils", _root / "data_utils.py")
_load_module("quant_utils", _root / "quant_utils.py")
_load_module("eval_utils", _root / "eval_utils.py")
_load_module("model", _root / "model.py")

# Finally, launch the main script
print("[windows] Launching train_gpt.py...")
_main_path = _root / "train_gpt.py"
_main_source = _main_path.read_text(encoding="utf-8")
_main_code = compile(_main_source, str(_main_path), "exec")

_globals = {
    "__name__": "__main__",
    "__file__": str(_main_path),
    "__builtins__": __builtins__,
}
exec(_main_code, _globals)
