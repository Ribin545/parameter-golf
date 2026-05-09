# Phase 8: Auxiliary Loss at Intermediate Layers Plan

**Date:** 2026-05-09 | **Status:** PLANNED

## Concept

Instead of only supervising the final output:

```
final hidden → LM head → loss
```

Also supervise an intermediate stage:

```
stage 3 hidden → same LM head → aux loss
final hidden   → same LM head → final loss
```

```
loss = final_loss + AUX_LOSS_WEIGHT * aux_loss
```

## Why It Helps

Early/middle blocks get direct next-token training signal instead of waiting for gradients to pass through every later layer. This can reduce entropy faster without significant parameter cost (aux head is **tied** — same weights as final LM head).

## Implementation Plan

### 1. Add config env vars

```
AUX_LOSS_ENABLED=1          # 0=disabled, 1=enabled
AUX_LOSS_LAYER=3            # which layer's output to supervise (1-indexed)
AUX_LOSS_WEIGHT=0.075       # weight of auxiliary loss in total loss
AUX_HEAD_TIED=1             # 1=reuse lm_head/tok_emb, 0=separate head
```

### 2. Modify `GPTMultiLayer.forward_logits()` in `model_multilayer.py`

- During the recurrence loop, capture the hidden state after `blocks[AUX_LOSS_LAYER - 1]` at the **last recurrence step**
- Return a dict/tuple: `(logits, aux_hidden_dict)` or modify forward to return multiple values
- Actually simpler: modify `forward()` to compute aux loss internally

### 3. Modify `GPTMultiLayer.forward()` in `model_multilayer.py`

Current signature: `forward(self, input_ids, target_ids) -> Tensor`
New behavior:
- Run forward_logits which returns final logits
- Also capture intermediate hidden at AUX_LOSS_LAYER
- Compute final_loss from final logits
- Compute aux_loss from intermediate hidden using same LM head
- Return weighted sum: `final_loss + AUX_LOSS_WEIGHT * aux_loss`

### 4. Key Design Decisions

- **AUX_HEAD_TIED=1**: Reuse `tok_emb.weight` (tied) or `lm_head.weight` (untied) for the auxiliary head. Zero additional parameters.
- **Aux loss at last recurrence step only**: We supervise the intermediate layer's hidden state from the final recurrence pass (most mature representation).
- **Same softcap**: Apply same `logit_softcap * tanh(logits / logit_softcap)` to intermediate logits.

### 5. Files to Modify

| File | Changes |
|------|---------|
| `model_multilayer.py` | Add `forward_aux` with intermediate hidden capture, modify `forward()` to compute combined loss |
| `train_gpt.py` | Add AUX_LOSS env var parsing, pass to GPTMultiLayer init |
| `notes/phase8_path_report.md` | Update with aux loss results once tested |

### 6. No Impact on Step Time

- Aux loss computation is just one extra LM head forward (small matmul) + cross_entropy
- Expected overhead: < 1ms per step (negligible)
- No new parameters (tied head)