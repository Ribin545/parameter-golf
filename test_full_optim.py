import os, time, torch

os.environ.update({
    "MLP_MEMORY_MODE": "off",
    "ATTN_MEMORY_MODE": "off",
    "SDPA_BACKEND": "auto",
    "MULTILAYER_ACTIVATION_CHECKPOINT": "0",
    "MLP_RECOMPUTE": "0",
})

from model_multilayer import GPTMultiLayer

device = torch.device("cuda")
model = GPTMultiLayer(
    1024, 5, 512, 8, 4, 2, False, 0.005, 2, 30.0, 10000.0, 3.0,
    4096, 0.05, 8, 2, True, 0.005, 0.08, 0.0
).to(device).bfloat16()

# Build actual optimizer like train_gpt.py
matrix_params, lora_params_list, control_params, scalar_params = [], [], [], []
for name, p in model.named_parameters():
    if "tok_emb" in name:
        continue
    is_lora = "lora" in name
    is_control = any(x in name for x in ["attn_scale","mlp_scale","resid_mix","q_gain","skip_weight"])
    is_matrix = p.ndim == 2 and not is_lora and not is_control and "bigram_hash" not in name
    if is_lora:
        lora_params_list.append(p)
    elif is_control:
        control_params.append(p)
    elif is_matrix:
        matrix_params.append(p)
    else:
        scalar_params.append(p)

# Use the actual Muon + AdamW setup
from optimizer_utils import Muon
muon_opt = Muon(matrix_params, lr=0.15, momentum=0.95, backend_steps=7)
adam_groups = [
    {"params": [model.tok_emb.weight], "lr": 0.03, "target_lr": 0.03, "weight_decay": 0.0},
    {"params": lora_params_list, "lr": 0.04, "target_lr": 0.04, "weight_decay": 0.0},
    {"params": control_params, "lr": 0.04, "target_lr": 0.04, "weight_decay": 0.0},
    {"params": scalar_params, "lr": 0.04, "target_lr": 0.04, "weight_decay": 0.2},
]
if model.lm_head is not None:
    adam_groups.append({"params": [model.lm_head.weight], "lr": 0.008, "target_lr": 0.008, "weight_decay": 0.1})
adam_opt = torch.optim.AdamW(adam_groups, betas=(0.9, 0.95), eps=1e-8, fused=True)

x = torch.randint(0, 1024, (150, 1024), device=device)
y = torch.randint(0, 1024, (150, 1024), device=device)

# Test: fwd + bwd + optim (no compile)
for _ in range(3):
    muon_opt.zero_grad(set_to_none=True)
    adam_opt.zero_grad(set_to_none=True)
    loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
    muon_opt.step()
    adam_opt.step()
    model.zero_grad(set_to_none=True)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    muon_opt.zero_grad(set_to_none=True)
    adam_opt.zero_grad(set_to_none=True)
    loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
    muon_opt.step()
    adam_opt.step()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
no_compile = (time.perf_counter()-t0)*1000/5

print(f"Full step (no compile): {no_compile:.1f}ms")

# Compile
cm = torch.compile(model, mode="reduce-overhead")
for _ in range(5):
    muon_opt.zero_grad(set_to_none=True)
    adam_opt.zero_grad(set_to_none=True)
    loss = cm(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
    muon_opt.step()
    adam_opt.step()
    model.zero_grad(set_to_none=True)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    muon_opt.zero_grad(set_to_none=True)
    adam_opt.zero_grad(set_to_none=True)
    loss = cm(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
    muon_opt.step()
    adam_opt.step()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
compiled = (time.perf_counter()-t0)*1000/5

print(f"Full step (compiled):   {compiled:.1f}ms")
print(f"Speedup:                {no_compile/compiled:.2f}x")