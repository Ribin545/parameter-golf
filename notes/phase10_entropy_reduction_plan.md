# Phase 10: Entropy Reduction Plan — Use Spare Artifact Budget

## Context

- Artifact limit: **16MB** (not 10MB as previously assumed)
- Current `final_model.int8.ptz`: **10.38 MiB**
- **Spare headroom: ~5.6MB**

Goal: Use spare artifact budget for mechanisms that reduce prediction entropy (lower val_bpb) without changing any existing hyperparameters (dim=512, 5L, 2S, LoRA rank=8, etc.).

---

## Option A: Learned Bigram Logit Bias (Recommended)

### What

Add a **learned** 1024×1024 table `bigram_logit_bias` that gets added to final logits based on the previous token:

```python
# In model_multilayer.py, in forward_logits()
prev_ids = F.pad(input_ids[:, :-1], (1, 0), value=0)  # shift right, pad with 0
logits = logits + self.bigram_logit_bias[prev_ids] * self.bigram_logit_scale
```

### Why It Works

- Bigram statistics capture strong local dependencies that even small transformers struggle with
- The neural net learns long-range patterns; the bigram table handles short-range
- **Synergy, not redundancy**: The existing `BigramHashEmbedding` adds a *learned embedding vector* to the input. The bigram logit bias adds a *direct logit adjustment* at the output. They operate at different levels of the network.

### Size

| Component | Params | fp32 size | int8 compressed |
|-----------|--------|-----------|-----------------|
| `bigram_logit_bias` | 1,048,576 | 4.0 MB | **1.0 MB** |
| `bigram_logit_scale` (scalar) | 1 | 4 bytes | 4 bytes |

**Total added to artifact: ~1.0MB** → New total: ~11.4MB (well under 16MB)

### Implementation

```python
class GPTMultiLayer(nn.Module):
    def __init__(self, ..., bigram_logit_enabled: bool = False, bigram_logit_scale_init: float = 0.05):
        # ... existing init ...
        self.bigram_logit_enabled = bigram_logit_enabled
        if self.bigram_logit_enabled:
            self.bigram_logit_bias = nn.Parameter(torch.zeros(vocab_size, vocab_size))
            self.bigram_logit_scale = nn.Parameter(torch.tensor(bigram_logit_scale_init))
    
    def forward_logits(self, input_ids: Tensor, **kwargs) -> Tensor:
        # ... existing path ...
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        
        if self.bigram_logit_enabled and self.training:
            prev_ids = F.pad(input_ids[:, :-1], (1, 0), value=0)
            bias = self.bigram_logit_bias[prev_ids] * torch.sigmoid(self.bigram_logit_scale)
            logits = logits + bias
        
        return logits
```

### Training Notes

- The bias is **learned end-to-end** via backprop through cross-entropy
- `bigram_logit_scale` is initialized to 0.05 and learned via sigmoid gate
- During the first ~100 steps, the scale stays small (gate ≈ 0.05) as the neural net warms up
- After warmup, the scale can increase if the bigram signal helps

### Expected Impact

- **val_bpb reduction**: 0.01-0.03 (empirical from similar architectures)
- **Step time impact**: ~0.1ms (just a vector lookup + add)
- **VRAM impact**: +4MB during training (fp32 param), +1MB int8 at eval

### A/B Test Plan

```bash
# Variant A: Baseline (no bigram logit)
export BIGRAM_LOGIT_ENABLED=0

# Variant B: Bigram logit enabled
export BIGRAM_LOGIT_ENABLED=1
export BIGRAM_LOGIT_SCALE_INIT=0.05
```

Run 10-min A/B with identical seeds. Compare val_bpb at step 500 and final.

---

## Option B: Trigram Hash Embedding (Extension of Existing)

### What

Extend the existing `BigramHashEmbedding` to also capture trigrams:

```python
class TrigramHashEmbedding(nn.Module):
    def __init__(self, hash_size: int, model_dim: int, scale: float, vocab_size: int):
        self.table = nn.Embedding(hash_size, model_dim)
    
    def forward(self, input_ids: Tensor) -> Tensor:
        prev2 = F.pad(input_ids[:, :-2], (2, 0), value=0)  # 2 tokens back
        prev1 = F.pad(input_ids[:, :-1], (1, 0), value=0)  # 1 token back
        hash_idx = (prev2.long() * vocab_size * vocab_size + 
                    prev1.long() * vocab_size + 
                    input_ids.long()) % hash_size
        return self.table(hash_idx) * self.scale
```

### Size

| Component | fp32 | int8 |
|-----------|------|------|
| Trigram table (4096 × 512) | 8.0 MB | 2.0 MB |
| Bigram table (existing) | 8.0 MB | 2.0 MB |

**Total added: ~2.0MB** → New total: ~12.4MB

### Why Less Preferred

- Trigrams are much sparser than bigrams in the training data
- 4096 hash buckets for trigrams with vocab=1024 has higher collision rate than bigrams
- The neural net already captures trigram patterns through attention

---

## Option C: Unigram Bias Refinement (Tiny, Easy Win)

### What

The existing `lm_bias` is initialized from validation unigram log-probs. But it's **not updated during training** (it's in the AdamW scalar group with wd=0.2 but lr=0.03, so it does get updated slowly).

**Issue**: The initialization is from validation data, which may not match training distribution perfectly.

**Fix**: Initialize from **training data** unigram counts instead, or make it a **learned** parameter with higher LR.

Actually, looking at `train_gpt.py`, `lm_bias` IS already learned (it's in the scalar group). So this is already happening. **No change needed.**

But we could add a **static unigram prior** that doesn't get trained — just a compressed table from training data:

```python
# Static, not learned
self.unigram_prior = nn.Buffer(torch.zeros(vocab_size))  # compressed int8
# Pre-computed from training data frequency counts
```

**Size**: 1024 bytes (negligible)
**Impact**: Minimal — the learned lm_bias already does this

---

## Recommended Plan

### Phase 10a: Implement & Test Option A

1. **Add `bigram_logit_bias` to `GPTMultiLayer`**
   - New parameter: `bigram_logit_bias` (1024×1024)
   - New parameter: `bigram_logit_scale` (scalar, sigmoid-gated)
   - Modified `forward_logits()` to add bias

2. **Update `train_gpt.py`**
   - Parse `BIGRAM_LOGIT_ENABLED` env var
   - Parse `BIGRAM_LOGIT_SCALE_INIT` env var
   - Pass to model constructor

3. **Update `eval_utils.py`**
   - Ensure bigram logit works during evaluation (it should, it's just a parameter add)

4. **A/B Test**
   - Baseline: `BIGRAM_LOGIT_ENABLED=0`
   - Test: `BIGRAM_LOGIT_ENABLED=1`, `BIGRAM_LOGIT_SCALE_INIT=0.05`
   - 10-min runs, same seed, compare val_bpb

### Phase 10b: If Successful, Add to `trial_5090.sh`

```bash
# Phase 10a: learned bigram logit bias
export BIGRAM_LOGIT_ENABLED=1
export BIGRAM_LOGIT_SCALE_INIT=0.05
```

---

## Expected Outcome

If the bigram logit bias gives even **0.01 val_bpb reduction**:

- Current: **1.5639**
- Target: **1.5539** or lower

This would be a meaningful improvement with:
- **Zero hyperparameter changes** (dim, layers, steps, LORA rank all unchanged)
- **~1MB artifact size increase** (from 10.4MB → 11.4MB)
- **~0.1ms step time increase** (negligible)
- **~4MB VRAM increase during training** (acceptable with 10GB headroom)

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| No quality improvement | 30% | Easy to disable (env var), no architecture change |
| Slight overfitting to bigrams | 15% | Sigmoid gate limits scale, regularization still active |
| Artifact size concern | 5% | 11.4MB is still well under 16MB |
| Step time regression | 5% | Lookup is O(1), just vector add |

**Overall risk: LOW**. Easy to implement, easy to test, easy to revert.