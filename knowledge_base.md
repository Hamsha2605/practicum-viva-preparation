# Knowledge base — cold-recall reference

Everything here is verified against `final/Practicum_Final_Paper.pdf` (the actual
submission) and `src/notebook/XAI_Final_Notebook.ipynb` (the actual final code). Nothing
here is sourced from `src/notebook/FINDINGS.md` or `MODEL_DECISIONS.md` — those describe
an earlier, abandoned CNN-LSTM/window-level pipeline and should not be quoted in the
viva or the presentation.

---

## 1. Headline numbers (memorise these cold)

| Metric | Value |
|---|---|
| Recall | 75.0% |
| Precision | 78.7% |
| PR-AUC | 0.79 |
| False alarm rate (FAR) | 1.8% |
| Accuracy | 96.3% |
| Test traces | 1,548 (128 attack, 1,420 normal) |
| Confusion matrix | TN 1,394 · FP 26 · FN 32 · TP 96 |
| Survey sample | 17 participants |
| Attention "most effective" vote | 53% (9/17) |
| Occlusion "most effective" vote | 29% (5/17) |
| LIME "most effective" vote | 18% (3/17) |
| LIME CPU cost | 11,583.3 ms explanation / 11,673.3 ms total |
| LIME GPU cost | 431.8 ms explanation / 518.7 ms total |
| Attention cost (GPU / CPU) | 188.4 ms / 196.9 ms |
| Occlusion cost (GPU / CPU) | 171.1 ms / 195.3 ms |
| Baseline prediction (GPU / CPU) | 86.9 ms / 90.0 ms |

## 2. Dataset facts (NGIDS-DS)

- Source: W. Haider, *"Developing Reliable Anomaly Detection System for Critical Hosts:
  A Proactive Defense Paradigm"*, UNSW PhD thesis, 2018 — reference [20].
- 90,054,239 total events → 1,262,427 attack (1.4%), 88,791,812 normal.
- Simulated on a 32-bit Linux host, 6 days: 11–16 March 2016.
- 99 CSV files, ~10GB raw → ~300MB as parquet after compression.
- Schema: `date, time, pro_id, path, sys_call, event_id, attack_cat, attack_subcat, label`.
- Attack categories and event counts:

  | Category | Events |
  |---|---|
  | Exploits | 900,828 |
  | Denial of Service | 129,185 |
  | Generic | 79,624 |
  | Backdoors | 70,712 |
  | Shellcode | 57,263 |
  | Worms | 14,125 |
  | Reconnaissance | 10,690 |

- Split (`XAI_Final_Notebook.ipynb`, cell 3): `TRAIN_END = "2016-03-14"`,
  `VALIDATION_DAY = "2016-03-15"`, `TEST_DAY = "2016-03-16"`.

## 3. Cleaning and feature engineering

- Stop-words removed: `clock_gettime`, `gettimeofday` — 52% of all events, same attack
  rate as the overall dataset, so zero separating power.
- Duplicates removed via `SELECT DISTINCT` on the full row.
- Result: 90,054,239 → 42,647,584 events.
- Run-length encoding: consecutive repeats of the same syscall collapse into one
  (token, repeat_count) run.
- Fixed sequence length `L = 256`: evenly downsample traces with more runs, zero-pad
  shorter ones.
- Vocabulary built only from syscalls seen in training days (`date <= TRAIN_END`); pad
  token = 0, unknown token = 1. Verified directly against the dataset: **117 distinct
  syscalls** survive in the training days after stop-word/duplicate cleaning, out of
  **347** total syscall IDs defined in the 32-bit Linux syscall lookup table
  (`dataset/syscall_32.parquet`) this dataset uses. Any syscall at validation/test time
  that isn't in that 117-token training vocabulary maps to the unknown token (1) rather
  than being dropped or erroring — `vocab.get(int(v), 1)` in the tokenisation code.
- Program name / process identity is **not** an input feature at any point.
- Trace label = 1 if any event in that (date, pro_id, path) trace is labelled attack.

## 4. Model architecture (exact, from the training cell)

```
dm = 64          # embedding dimension
heads = 4        # attention heads
ff = 128          # feed-forward width
encoder blocks = 2
dropout = 0.3
L = 256           # sequence length
```

- Per position: token embedding (dm=64) + positional embedding (dm=64) +
  `Dense(dm)` on `log1p(repeat_count)`, all summed.
- Padding mask: `tok != 0`, applied both inside `MultiHeadAttention` and at pooling.
- Each encoder block: `MultiHeadAttention(heads=4, key_dim=16)` → residual + LayerNorm →
  `Dense(128, relu) → Dense(64)` feed-forward → residual + LayerNorm.
- Pooling: masked mean over non-padded positions → one 64-d vector per trace.
- Head: `Dropout(0.3) → Dense(1, sigmoid)` → P(attack).
- Loss: binary cross-entropy, class-weighted (`neg/pos` ratio computed on the training
  split only). **No focal loss, no oversampling, no SMOTE** in the final model.
- Optimiser: Adam, batch size 128, up to 25 epochs.
- Early stopping: monitor `val_pr_auc`, patience 4, restore best weights.
- Threshold: chosen on the validation split by maximising F2 (`beta=2.0`), frozen and
  applied unchanged to the test split.
- Random seeds fixed (`tf.random.set_seed(42)`, `np.random.seed(42)`), but GPU/cuDNN
  attention and softmax kernels are not bit-deterministic, so repeated runs can drift
  slightly — the reported numbers are from one frozen, saved run
  (`dataset/final_models/trace_transformer/`).

## 5. Explainability methods — implementation detail

**Behaviour groups** (used to summarise both occlusion and attention):
`network` (socketcall, poll, select, recv/send family, connect, accept…),
`file` (read, write, open/openat, stat64, close, lseek…),
`process` (clone, fork, execve, exit_group, wait4, kill…),
`memory` (mmap2, munmap, mprotect, brk),
`signal` (rt_sigaction, rt_sigprocmask, futex…), else `other`.

- **Occlusion**: ablate a behaviour group or single syscall by zeroing its positions,
  re-score with the full model, record the drop in log-odds
  (`base_logit - logit(score_after)`). Exact, model-agnostic. Cost ≈ a couple of small
  batched `model.predict` calls (one batch per group, one per unique syscall).
- **Attention**: for each `MultiHeadAttention` layer, run a sub-model up to that layer's
  input, then re-invoke the layer with `return_attention_scores=True`; average received
  attention over heads and layers, normalise to sum to 1, then aggregate by behaviour
  group. **Not literally free** — it needs a sub-model re-invocation per layer, which is
  why it costs ~170–200ms, comparable to occlusion.
- **LIME**: `LimeTabularExplainer` over a binary "position present/removed" feature
  space; perturbations generated by the `lime` package, each scored via the same
  batched model-predict wrapper; a linear surrogate is fit to the results, giving one
  weight per position. Approximate, not exact. Cost scales with sample count, which is
  why it's GPU-bound and blows up on CPU.
- The worked example used throughout the paper's figures: a Firefox (`/usr/lib/firefox/
  firefox`) Exploits-category trace from the test day (16 March 2016), a 22-run window,
  base score 0.92-ish against the frozen threshold — poll/clone dominate across all
  three methods.

## 6. RQ1 — survey detail

| Method | Usefulness | Ease | Trust |
|---|---|---|---|
| Occlusion | 6.24 | 6.59 | 6.94 |
| Attention | 7.29 | 7.29 | 7.00 |
| LIME | 7.06 | 6.82 | 6.88 |

- Sample: 17 — 6 software developers, 4 IT professionals, 3 security analysts,
  4 students; cybersecurity/DL experience ranging from none to expert.
- Rating scales run roughly 1–9 (values above are averages on that scale, per the
  paper's Table 4.1).
- No formal significance test is reported in the final paper — the write-up
  deliberately says "no single method was rated significantly higher," which is a
  descriptive, not inferential, claim. Don't invent a p-value if asked for one live.
- Open feedback themes (verbatim intent, 3 bullets in the paper): more plain-language
  explanation alongside visuals; a clearer baseline/threshold to interpret raw numbers
  against; a similarity score against previously seen attacks.

## 7. RQ2 — cost detail

| Method | GPU explanation (ms) | GPU total (ms) | CPU explanation (ms) | CPU total (ms) |
|---|---|---|---|---|
| Baseline | — | 86.9 | — | 90.0 |
| Attention | 188.4 | 275.3 | 196.9 | 286.9 |
| Occlusion | 171.1 | 258.0 | 195.3 | 285.3 |
| LIME | 431.8 | 518.7 | 11,583.3 | 11,673.3 |

- Hardware: NVIDIA RTX 3090 (24GB VRAM), 24-core AMD EPYC 7402P, 32GB RAM, CUDA 13.0.
- Software: Python, TensorFlow/Keras, scikit-learn, `lime` package, DuckDB for data prep.
- Each figure is an average of 10 timed trials (`TRIALS = 10` in the cost-measurement
  cell), timed with `time.perf_counter()`.
- CPU run forces `tf.device("/CPU:0")` and reloads the model on that device.

## 8. Formulas

```
Recall     = TP / (TP + FN)
Precision  = TP / (TP + FP)
FAR        = FP / (FP + TN)
Accuracy   = (TP + TN) / (TP + TN + FP + FN)
PR-AUC     = Σ_n (R_n − R_{n−1}) · P_n      (area under precision-recall curve)
F_beta     = (1+β²)·Precision·Recall / (β²·Precision + Recall);  F2 uses β=2, weights recall higher
Self-attention: Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V
Sigmoid:   σ(x) = 1 / (1 + e^(−x))
```

## 9. Reference gist (for "what does citation [n] say")

| # | Author(s) | One-liner |
|---|---|---|
| 1 | Asharf et al. | Review of ML/DL IDS in IoT — challenges, solutions, future directions |
| 2 | CrowdStrike | 2025 Global Threat Report — 79% of 2024 detections were malware-free |
| 3 | Kalasampath et al. | Literature review on XAI applications (IEEE Access, 2025) |
| 4 | Anton et al. | SVM vs Random Forest anomaly detection, industrial OT (Modbus/OPC UA), RF up to 99.9% |
| 5 | Evangelou & Adams | Poisson regression / regression trees / Quantile Regression Forests on NetFlow volume |
| 6 | Gadal et al. | K-Means + Sequential Minimal Optimization, combined 97.36% CCI |
| 7 | Lee et al. | ML + Google Rapid Response/OSSEC, two-stage detect-then-respond |
| 8 | Du et al. | **DeepLog** — LSTM on system logs, incremental/online learning; widely cited (2,952 citations, user-verified via Google Scholar) |
| 9 | Psychogyios et al. | CNN + dual-LSTM + self-attention, UNSW-NB15, F1 0.83 (vs LSTM 0.75, CNN 0.71) |
| 10 | Cheng | Multi-model ensemble (Isolation Forest, Autoencoder, LSTM, GNN), 97.8% acc |
| 11 | Chaurasia et al. | Federated ResNet, Industrial IoT (X-IIoTID), 99.43% centralised / 99.16% federated |
| 12 | Nakıp & Gelenbe | Self-supervised online Auto-Associative Deep RNN, adapts to time-varying threats (botnets) |
| 13 | Senoussi et al. | **Closest paper** — Transformer encoder on ADFA-LD, 95.83% acc / 93.6% prec / 90% recall / 91.76% F1 |
| 14 | Khan et al. | XAI-based IDS for Industry 5.0 + adversarial XAI systematic review — "black box" citation |
| 15 | Mondal et al. | LADDERS — log correlation via Recursive Parallel Causal Discovery |
| 16 | Salloum & Norozpour | 1D CNN + SHAP/LIME, DDoS detection, F1 0.94 on NSL-KDD |
| 17 | Hariharan et al. | Permutation Importance + SHAP + LIME + CIU on RF/XGBoost, 99.9% Kaggle / 0.769 NSL-KDD binary |
| 18 | Hooshmand et al. | SKM-XGB (SMOTE+K-means+DAE+XGBoost) + SHAP, 99.37% NSL-KDD / 99.01% UNSW-NB15 |
| 19 | Fernandez-Morales et al. | DT/RF/LSTM/MLP + SHAP/LIME with resource-awareness, NF-UQ-NIDS-v2, RF 98.96% low overhead |
| 20 | Haider | NGIDS-DS dataset origin — UNSW PhD thesis, 2018 |
| 21 | Vaswani et al. | *Attention Is All You Need* — Transformer origin |
| 22 | Zeiler & Fergus | Occlusion method origin (visualising CNNs) |
| 23 | Ribeiro et al. | LIME origin |

## 10. Glossary (say these correctly, unprompted, if asked)

- **Trace** — all syscall events for one (date, pro_id, path) process on one day.
- **Run** — a maximal consecutive repeat of the same syscall within a trace, collapsed
  to one (token, count) pair by run-length encoding.
- **Behaviour group** — the six-way bucket (network/file/process/memory/signal/other)
  used to summarise occlusion and attention at a coarser level than individual syscalls.
- **F2 threshold** — the decision cutoff chosen on the validation split to maximise
  F-beta with β=2, i.e. weighting recall above precision; frozen before touching test data.
- **PR-AUC / Average Precision (AP)** — same 0.79 number, reported as PR-AUC in Table
  form and as AP in the precision-recall curve caption; they're the same quantity here.
- **FAR (false alarm rate)** — fraction of *normal* traces incorrectly flagged; not to
  be confused with 1 − precision (which is about alerts, not about normal traces).

## 11. Things to actively avoid saying (stale/unverified claims)

- Do **not** cite an identity-ablation number (e.g. "removing identity dropped the score
  by 0.76") — no such experiment exists in the final codebase for this model.
- Do **not** cite window-level recall figures (e.g. "0.30 recall at window level, 0.81 at
  trace level") — those numbers come from an abandoned CNN-LSTM/window-based pipeline,
  not the Transformer reported in the final paper.
- Do **not** claim focal loss, SMOTE, or program-identity embeddings are part of the
  final model — none of them are; only class-weighted BCE is used.
- Do **not** claim a formal significance test (Friedman/Wilcoxon) result for RQ1 — the
  final paper reports descriptive statistics only.
- Do **not** claim a per-category recall table exists in the paper — it was computed
  internally but deliberately excluded (see viva_qna.md, B6).
