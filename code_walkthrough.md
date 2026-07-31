# Code walkthrough — line by line

A cell-by-cell trace through `src/notebook/XAI_Final_Notebook.ipynb`, the actual
notebook the final paper's numbers come from. This is for depth-of-questioning defence
— if an examiner asks "what does this line actually do," the answer is here, not
paraphrased from memory. Cell numbers match the notebook's actual cell index (0-based,
counting markdown cells too), so you can jump straight to the cell in question while
answering.

Cells not covered here (pure markdown headers, or the CUDA environment hotpatch in
cell 1) are either self-explanatory or not worth defending in depth — cell 1 just
pre-loads the right `.so` files so TensorFlow finds the GPU; it has zero effect on any
number in the paper and isn't worth memorising.

---

## Cell 3 — DuckDB connection and constants

```python
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt

dataset_root = Path("../../dataset")
raw_host_logs = dataset_root / "raw_backup" / "host_logs.parquet"
syscall_lookup = dataset_root / "syscall_32.parquet"

con = duckdb.connect()
con.execute("SET memory_limit='16GB'")
con.execute("SET preserve_insertion_order=false")
con.execute(f"SET temp_directory='{dataset_root / 'duckdb_tmp'}'")
def q(sql): return con.sql(sql).df()

TRAIN_END = "2016-03-14"
VALIDATION_DAY = "2016-03-15"
TEST_DAY = "2016-03-16"
STOPWORDS = ["clock_gettime", "gettimeofday"]
L = 256
```

- `dataset_root` points two directories up from the notebook to the dataset folder.
  `raw_host_logs` is the parquet-compressed version of the 90M-row raw CSV export
  (~300MB); `syscall_lookup` maps numeric syscall IDs to their human-readable names
  (used throughout for readability, e.g. "poll" instead of "168").
- `con = duckdb.connect()` opens an **in-process, in-memory** DuckDB database — no
  server, no separate file, lives for the notebook's Python process lifetime.
- `memory_limit='16GB'` caps DuckDB's own working memory so it doesn't try to grab all
  32GB of host RAM and starve the rest of the notebook (TensorFlow needs headroom too).
- `preserve_insertion_order=false` tells DuckDB it's allowed to return rows in
  whatever order is cheapest internally, *except* wherever an explicit `ORDER BY` is
  written — this is what makes the pipeline reproducible despite that setting: every
  query that needs a specific order (the run-length encoding, the trace ordering)
  states `ORDER BY` explicitly, so this flag only removes an unnecessary
  order-preservation cost on queries that don't care.
- `temp_directory` gives DuckDB a scratch disk location to spill to if a query's
  intermediate result doesn't fit in the 16GB cap.
- `q(sql)` is a one-line convenience wrapper: run SQL, get a pandas DataFrame back.
- The five constants at the bottom are the single source of truth for the whole
  notebook: the day boundaries for the train/validation/test split, the two syscalls
  treated as noise, and the fixed sequence length. Every later cell reads these
  variables rather than hardcoding dates or the length again.

---

## Cell 6 — class balance

```python
comp = q(f"""SELECT count(*) events, sum(label) attack, count(*)-sum(label) normal,
             round(100.0*sum(label)/count(*), 2) attack_pct FROM '{raw_host_logs}'""")
```

- `sum(label)` works because `label` is 0/1 per event — summing a 0/1 column counts
  the 1s, i.e. the attack events. `count(*)-sum(label)` is everything else, i.e. normal.
  `100.0*sum(label)/count(*)` is the attack percentage — the `100.0` (not `100`) forces
  floating-point division rather than DuckDB doing integer division first.
- This is the query behind "1,262,427 attacks and 88,791,812 normal events" reported in
  the paper's dataset section.

## Cell 8 — attack categories

```python
by_cat = q(f"""SELECT attack_cat AS category, count(*) AS events
             FROM '{raw_host_logs}' WHERE label = 1 GROUP BY attack_cat ORDER BY events DESC""")
```

- Straightforward group-and-count, filtered to `label = 1` only (so `attack_cat` is
  never null in the result — normal events have a null `attack_cat`). This produces
  Table 3.1 in the paper directly.

## Cell 10 — trace length distribution

```python
tl = q(f"SELECT count(*) AS n FROM '{raw_host_logs}' GROUP BY date, pro_id, path")["n"]
```

- This is the first place `(date, pro_id, path)` appears as the trace key — the triple
  that defines "one process instance on one day" throughout the rest of the notebook.
  Grouping by it and counting gives one row per trace, with `n` = how many syscall
  events that trace has. The histogram built from this (on `log10(tl+1)`) is what
  motivates fixing a sequence length rather than using raw trace length directly — the
  distribution is extremely right-skewed (a handful of long-running daemons dominate).

## Cell 11 — do individual syscalls carry signal?

```python
top = q(f"""
    SELECT s.syscall_name, count(*) AS total,
           round(100.0*count(*)/sum(count(*)) OVER (), 1) AS pct_of_data,
           round(100.0*sum(h.label)/count(*), 2) AS attack_rate
    FROM '{raw_host_logs}' h LEFT JOIN '{syscall_lookup}' s ON h.sys_call = s.sys_call
    GROUP BY s.syscall_name ORDER BY total DESC LIMIT 10
""")
```

- Joins the raw log to the syscall-name lookup so the output is human-readable.
- `sum(count(*)) OVER ()` is a window function with an **empty** `PARTITION BY` —
  that means "sum over every group in the result," i.e. the grand total across all 10
  rows combined is used as the denominator for `pct_of_data`, giving each syscall's
  share of the overall dataset.
- `attack_rate` per syscall is compared against the dataset's overall attack rate
  (computed separately, `base`) in the resulting plot. The finding this produces —
  every top-10 syscall has an attack rate close to the ~1.4% baseline — is the direct
  evidence behind cell 12's markdown conclusion: no single call separates attack from
  normal, so the model has to read sequences, not individual calls.

## Cell 14 — cleaning

```python
stop_ids = [int(r[0]) for r in con.sql(
    f"SELECT sys_call FROM '{syscall_lookup}' WHERE syscall_name IN {tuple(STOPWORDS)}").fetchall()]

con.execute(f"""
CREATE OR REPLACE TABLE clean AS
SELECT DISTINCT date, time, pro_id, path, sys_call, event_id, attack_cat, attack_subcat, label
FROM '{raw_host_logs}'
WHERE sys_call NOT IN {tuple(stop_ids)}
""")
```

- First: look up the numeric syscall IDs for `clock_gettime` and `gettimeofday` (the
  IDs are `78` and `265` respectively, confirmed directly from the data — see below).
  `.fetchall()` returns a list of one-tuples; `int(r[0])` unwraps each one to a plain
  int, so `stop_ids` ends up as e.g. `[78, 265]`.
- Second: materialise a **persistent DuckDB table** called `clean` (not a Python
  variable — this lives inside the DuckDB session and every later cell in this session
  queries `FROM clean` directly). `SELECT DISTINCT` on the full column list is what
  removes duplicate-logged rows: two rows are only "the same" if every one of those
  nine columns matches exactly. `WHERE sys_call NOT IN (78, 265)` drops every
  occurrence of the two stop-word calls.
- This single statement is where 90,054,239 rows becomes 42,647,584 — both effects
  (dedup + stop-word removal) happen in the same query, so the notebook's own printed
  diff (`raw_n - clean_n`) reports the combined reduction, not the two effects
  separately.

## Cell 17 — trace metadata

```python
meta = q("""
    SELECT
        date, pro_id, path,
        max(label) AS lbl,
        count(*) AS trace_len,
        sum(label) AS attack_events,
        max(CASE WHEN label=1 THEN attack_cat END) AS primary_attack_cat,
        string_agg(DISTINCT attack_cat, '|' ORDER BY attack_cat) FILTER (WHERE label=1) AS attack_cats
    FROM clean
    GROUP BY date, pro_id, path
""")
```

- One row per trace (same `(date, pro_id, path)` grouping as cell 10, but now on the
  cleaned table). Per trace:
  - `max(label)` — the **trace-level label**: 1 if *any* event in the trace is an
    attack event, 0 only if every event is normal. This is exactly the rule the paper
    states ("marking a trace as an attack if any of its events was an attack").
  - `count(*)` — trace length in events (post-cleaning), stored as `trace_len`.
  - `sum(label)` — how many of this trace's events are individually attack-labelled.
  - `primary_attack_cat` — picks one representative category name using a `CASE`
    inside a `max()`: for non-attack rows the `CASE` evaluates to `NULL`, and
    `max()` ignores nulls, so this returns whichever attack category string happens to
    sort highest among the trace's attack events (a display convenience, not used in
    any reported metric).
  - `attack_cats` — `string_agg(DISTINCT ... , '|')` with a `FILTER (WHERE label=1)`
    builds a pipe-separated list of every *distinct* attack category present in the
    trace's attack events (e.g. `"Exploits|Shellcode"` for a trace touching both). This
    is the field the internal `category_recall_table` function (cell 20/24) uses, and
    exactly the field flagged in `viva_qna.md` B6 as unreliable at scale because the
    `(date, pro_id, path)` key can collide across unrelated process launches.

---

## Cell 19 — run-length encoding and downsampling (the densest cell in the notebook)

This is the cell most worth being able to explain slowly, since it's genuinely the
least obvious piece of the pipeline. It builds the vocabulary, then a six-stage SQL
pipeline (five CTEs plus a final `SELECT`) that turns a variable-length trace into
exactly 256 (token, repeat-count) pairs.

### Vocabulary

```python
vocab = {int(r[0]): i + 2 for i, r in enumerate(con.sql(f"""
    SELECT DISTINCT sys_call FROM clean WHERE date <= DATE '{TRAIN_END}' ORDER BY sys_call
""").fetchall())}
V = max(vocab.values(), default=1) + 1
```

- Only syscalls seen on or before `TRAIN_END` (11–14 March) get a vocabulary slot —
  this is deliberate: the model must never be given a vocabulary entry that was built
  using knowledge of validation or test-day data. There are **117** such syscalls
  (verified directly against the data: 347 syscalls total in the 32-bit lookup table,
  117 survive into the training vocabulary after stop-word removal).
- `i + 2` reserves index `0` for padding and index `1` for "unknown" (any syscall seen
  later, at validation/test time, that never appeared in training) — so real syscalls
  occupy indices 2 through 118.
- `V` (vocabulary size for the embedding layer) is one more than the largest index used.

### The SQL pipeline

```sql
WITH ordered AS (
    SELECT date, pro_id, path, sys_call,
           row_number() OVER (PARTITION BY date, pro_id, path ORDER BY time, event_id) AS rn
    FROM clean
),
```
Assigns each event a sequential position `rn` *within its own trace*, ordered by time
then `event_id` as a tiebreaker (host timestamps are 1-second resolution, so many
events in the same trace share the same `time` — `event_id` gives a stable, deterministic
order among them).

```sql
tagged AS (
    SELECT *, CASE WHEN sys_call = lag(sys_call) OVER (PARTITION BY date, pro_id, path ORDER BY rn)
                   THEN 0 ELSE 1 END AS is_new_run
    FROM ordered
),
```
`lag(sys_call)` looks at the *previous* row's syscall within the same trace (ordered by
`rn`). If the current syscall equals the previous one, this event is a continuation of
a run (`is_new_run = 0`); otherwise — including the very first event in a trace, where
`lag()` returns `NULL` and `sys_call = NULL` is never true in SQL — it starts a new run
(`is_new_run = 1`).

```sql
runs AS (
    SELECT *, sum(is_new_run) OVER (PARTITION BY date, pro_id, path ORDER BY rn) AS run_id
    FROM tagged
),
```
A running (cumulative) sum of `is_new_run`, ordered by `rn`, is the standard SQL trick
for numbering consecutive groups: it only increments at the start of each new run and
stays flat across repeats, so every event in the same run ends up with the same
`run_id`.

```sql
collapsed AS (
    SELECT date, pro_id, path, run_id,
           min(rn) AS run_start, min(sys_call) AS sys_call, count(*) AS repeat_count
    FROM runs
    GROUP BY date, pro_id, path, run_id
),
```
Collapses each run to one row: `run_start` is where the run began (used only for
ordering next), `sys_call` is the repeated call (`min()` is safe here since every row
in a run shares the same syscall by construction), `repeat_count` is how many times it
repeated consecutively.

```sql
numbered AS (
    SELECT *,
           row_number() OVER (PARTITION BY date, pro_id, path ORDER BY run_start) AS run_pos,
           count(*) OVER (PARTITION BY date, pro_id, path) AS run_cnt
    FROM collapsed
),
```
Re-numbers the collapsed runs sequentially (`run_pos` = 1, 2, 3... within the trace) and
computes `run_cnt`, the total number of runs in that trace — this is the number that
decides whether downsampling is even needed.

```sql
picks AS (
    SELECT date, pro_id, path, pos,
           CASE WHEN sample_n = 1 THEN 1
                ELSE 1 + CAST(round(CAST(pos AS DOUBLE) * (run_cnt - 1) / (sample_n - 1)) AS BIGINT)
           END AS run_pos
    FROM (SELECT date, pro_id, path, run_cnt, least(run_cnt, {L}) AS sample_n
          FROM numbered GROUP BY date, pro_id, path, run_cnt)
    CROSS JOIN range(0, CAST(sample_n AS BIGINT)) AS r(pos)
    )
```
This is the actual downsampling logic:
- `sample_n = least(run_cnt, L)` — if the trace has 256 runs or fewer, `sample_n` equals
  `run_cnt` and **every run is kept** (no downsampling). If it has more than 256 runs,
  `sample_n` is capped at 256.
- `CROSS JOIN range(0, sample_n)` generates one output row per pick, `pos = 0 .. sample_n-1`.
- The `CASE` is a linear-interpolation formula — equivalent to
  `numpy.linspace(1, run_cnt, sample_n)` rounded to the nearest integer — that maps
  each `pos` evenly across the full range of run positions `1..run_cnt`. At `pos = 0`
  this evaluates to `run_pos = 1` (the first run); at `pos = sample_n - 1` it evaluates
  to `run_pos = run_cnt` (the last run); everything in between is evenly spread. The
  `sample_n = 1` special case exists purely to avoid dividing by zero (`sample_n - 1`
  would be 0), for the edge case of a trace with exactly one run.
- Net effect: short traces (≤256 runs) are represented in full; long traces are
  represented by 256 evenly-spaced runs spanning their entire length, not just the
  first or last 256 — a long trace's behaviour late in its life is just as visible to
  the model as its behaviour near the start.

```sql
SELECT date, pro_id, path,
       list(sys_call ORDER BY pos) AS seq,
       list(repeat_count ORDER BY pos) AS repeat_counts
FROM picks
JOIN numbered USING (date, pro_id, path, run_pos)
GROUP BY date, pro_id, path
ORDER BY date, pro_id, path
```
Joins the picked `run_pos` values back to `numbered` to recover the actual syscall and
repeat count for each pick, then aggregates into two parallel ordered lists per trace —
`seq` (syscall IDs) and `repeat_counts` — using DuckDB's `list(... ORDER BY pos)`
aggregate, which guarantees the lists come out in the correct sampled order.

### Turning the SQL result into model tensors

```python
d = seqs.merge(meta.reset_index(), on=["date", "pro_id", "path"], how="inner") \
        .sort_values(["date", "pro_id", "path"]).reset_index(drop=True)

Xs = np.zeros((len(d), L), dtype="int32")
Xcnt = np.zeros((len(d), L), dtype="float32")
for i, (seq, counts) in enumerate(zip(d["seq"], d["repeat_counts"])):
    tokens = [vocab.get(int(v), 1) for v in seq]
    Xs[i, :len(tokens)] = tokens
    Xcnt[i, :len(counts)] = np.log1p(np.asarray(counts, dtype="float32"))
```

- Merges the per-trace sequence data (`seqs`, from the SQL above) with the per-trace
  metadata (`meta`, from cell 17, which carries the trace-level label) on the trace key.
- `Xs` and `Xcnt` are pre-allocated as all-zeros — this **is** the padding: any trace
  with fewer than 256 sampled positions simply leaves the tail of its row at 0, and 0 is
  reserved (never a real vocabulary index), so padding is unambiguous from real data.
- `vocab.get(int(v), 1)` — the vocabulary lookup with a default: if the raw syscall ID
  `v` was seen in training, use its assigned index; if not (a syscall that only appears
  on validation/test days), fall back to index `1`, the "unknown" token, rather than
  crashing or dropping the event.
- `np.log1p(counts)` — `log(1 + x)`, applied to every repeat count. `log1p` rather than
  plain `log` because a run can legitimately repeat exactly once (`count = 1`), and
  `log(1) = 0` is fine, but the `+1` also keeps the transform well-behaved and monotonic
  for small counts without a special case.
- Later in the same cell: `n_tr / n_va / n_te` and the boolean masks `trS / vaS / teS`
  (built as `dts <= TRAIN_END`, `dts == VALIDATION_DAY`, `dts == TEST_DAY`) are what
  actually split `Xs`/`Xcnt`/`ys` into the three splits used for training — the split is
  applied to the already-vectorised tensors, not by re-querying DuckDB per split.

---

## Cell 20 — model definition and training

```python
dm, heads, ff = 64, 4, 128
tok_inp = tf.keras.Input((L,), dtype="int32", name="sys_call_token")
cnt_inp = tf.keras.Input((L, 1), dtype="float32", name="repeat_count")
valid = ops.cast(ops.not_equal(tok_inp, 0), "float32")
```
- Two named inputs matching the two arrays built in cell 19: token ids and repeat
  counts, both length-256. `valid` is a per-position float mask (1.0 for real tokens,
  0.0 for padding) derived by comparing the token id to the pad value `0` — reused
  later for both the attention mask and the pooling step.

```python
tok_emb = layers.Embedding(V, dm)(tok_inp)
pos_emb = layers.Embedding(L, dm)(ops.arange(0, L))
cnt_emb = layers.Dense(dm)(ops.log1p(cnt_inp))
h = tok_emb + pos_emb + cnt_emb
```
- `tok_emb`: standard learned lookup, one `dm=64`-length vector per token id, shape
  `(batch, 256, 64)`.
- `pos_emb`: `ops.arange(0, L)` is the plain sequence `[0, 1, ..., 255]` (not
  batch-dependent), embedded through a *second*, separately-learned embedding table
  indexed by position rather than token identity — this is what lets the model tell
  "poll at position 3" apart from "poll at position 200."
- `cnt_emb`: note `ops.log1p(cnt_inp)` is applied *again* here even though `Xcnt` was
  already log1p-transformed in cell 19 — this is not a double-log bug in the sense of
  redundant scaling collapse, but it does mean the actual value fed to the `Dense`
  layer is `log1p(log1p(repeat_count))`, a detail worth knowing if asked to derive the
  count pathway precisely, since it's not simply "log of the raw count."
- All three are plain-added (not concatenated) into one `(batch, 256, 64)` tensor `h` —
  addition works here because all three embeddings live in the same 64-dimensional
  space by construction (`Embedding(..., dm)` and `Dense(dm)` both target width 64).

```python
amask = ops.cast(ops.not_equal(tok_inp, 0), "bool")[:, None, :]
for _ in range(2):
    at = layers.MultiHeadAttention(num_heads=heads, key_dim=dm // heads)(h, h, attention_mask=amask)
    h = layers.LayerNormalization()(h + at)
    fw = layers.Dense(dm)(layers.Dense(ff, activation="relu")(h))
    h = layers.LayerNormalization()(h + fw)
```
- `amask` reshapes the padding mask to `(batch, 1, 256)` so Keras broadcasts it across
  every query position — it tells attention "these key positions don't exist, don't
  attend to them," applied identically at every layer and every head.
- `MultiHeadAttention(num_heads=4, key_dim=16)` — `key_dim = dm // heads = 64 // 4 = 16`,
  i.e. each of the 4 heads works in a 16-dimensional subspace, and their outputs are
  concatenated back to 64 dimensions internally by the Keras layer.
- Called as `(h, h, ...)` — query and key/value are the same tensor `h`, which is what
  makes this *self*-attention rather than cross-attention.
- `h + at` then `LayerNormalization()` is the residual-plus-norm pattern, applied twice
  per block (once around attention, once around the feed-forward pair) — this is the
  standard post-norm Transformer block from Vaswani et al., not a pre-norm variant.
- The feed-forward path is `Dense(64→128, relu)` then `Dense(128→64)` — expand then
  contract, the standard Transformer FFN shape.
- The whole `for _ in range(2)` loop is the "2 encoder blocks" from the paper — both
  blocks share the same code but have independently-learned weights (each `layers.X()`
  call inside the loop instantiates a fresh layer object).

```python
h = h * ops.expand_dims(valid, -1)
pooled = ops.sum(h, axis=1) / ops.maximum(ops.sum(valid, axis=1, keepdims=True), 1.0)
out = layers.Dense(1, activation="sigmoid")(layers.Dropout(0.3)(pooled))
```
- Zeroes out any residual signal at padding positions one more time before pooling
  (belt-and-braces on top of the attention mask), then sums over the sequence axis
  (`axis=1`) and divides by the count of real positions — a manual masked mean, not
  Keras's built-in pooling layer, because no built-in pooling layer takes a variable
  per-example denominator like this directly.
- `ops.maximum(..., 1.0)` guards against division by zero for the (never actually
  occurring, since every trace has at least one event) case of an all-padding trace.
- Final head: dropout at training time only, then a single sigmoid unit — binary
  probability of attack, no separate "normal" logit needed.

```python
cwS = {0: 1.0, 1: float((ys[trS] == 0).sum() / max((ys[trS] == 1).sum(), 1))}
history = transformer.fit(
    {...}, ys[trS],
    validation_data=({...}, ys[vaS]),
    epochs=25, batch_size=128, class_weight=cwS,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=4, restore_best_weights=True)],
)
```
- `cwS`: class 0 (normal) keeps weight 1.0; class 1 (attack) is weighted by the ratio
  of normal-to-attack traces **in the training split only** — computed fresh from
  `ys[trS]`, never from validation or test labels.
- `EarlyStopping(monitor="val_pr_auc", mode="max", patience=4, restore_best_weights=True)`
  — training keeps going up to 25 epochs, but stops early if `val_pr_auc` hasn't
  improved for 4 consecutive epochs, and whichever epoch had the best `val_pr_auc` is
  the one whose weights are kept (not necessarily the final epoch trained).

```python
val_scores = transformer.predict(...)
threshold = choose_threshold(ys[vaS], val_scores, beta=2.0)
test_scores = transformer.predict(...)
test_pred = test_scores >= threshold
```
- The threshold search (`choose_threshold`, defined earlier in the cell) tries every
  unique score value that appears in the validation predictions as a candidate cutoff,
  computes the F2 score at each, and keeps the cutoff with the highest F2 — an
  exhaustive search over actually-occurring values, not a coarse grid.
- Test predictions use that exact validation-derived `threshold` — nothing about the
  threshold is re-derived or adjusted using test data at any point.

---

## Cell 22 — save

Writes three artefacts to `dataset/final_models/trace_transformer/`: the Keras model
itself (`model.keras`), a compressed `.npz` bundle of every array needed to reproduce
evaluation without re-running the SQL pipeline (`eval_data.npz` — validation/test
tensors, trace metadata, category matrices, dates, pids, paths), and `meta.json`
(vocabulary, syscall names, category names, the chosen threshold, and both validation
and test metrics as they stood at save time). Cell 24 (testing) loads all three back in
rather than depending on any in-memory state from cell 20 — that's deliberate, so the
reported numbers come from a specific frozen artefact on disk, not from whatever
happened to be in memory during a particular run.

## Cell 24 — testing / final evaluation

Reloads the saved model and `meta.json`/`eval_data.npz`, recomputes `tscore` via
`tmodel.predict(...)`, applies the frozen threshold, and reproduces every number in the
paper's Model Evaluation section: `binary_metrics()` (recall/precision/PR-AUC/FAR/
accuracy — same formulas as cell 20), the confusion matrix (`sklearn.metrics.
confusion_matrix`, plotted with `imshow`), and the precision-recall curve
(`precision_recall_curve` + `average_precision_score`, which is the exact same
quantity as PR-AUC — the paper's Fig. 4.2 caption labels it "AP" while the metrics
table calls it "PR-AUC"; they are not two different numbers). This cell also computes
`category_recall_table(...)` and prints it — the per-category breakdown that exists in
the notebook's output but was deliberately not carried into the paper (`viva_qna.md` B6).

---

## Cell 26 — explainability helpers

```python
def behaviour_group(name):
    groups = {"network": {...}, "file": {...}, "process": {...}, "memory": {...}, "signal": {...}}
    for g, s in groups.items():
        if name in s: return g
    return "other"

def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def matrix_score(tokens, counts, masks):
    ...
    return tmodel.predict({"sys_call_token": x, "repeat_count": c}, verbose=0).ravel()
```
- `behaviour_group` is a hardcoded lookup table (syscall name → one of six buckets),
  used by both occlusion and attention to summarise at a coarser level than individual
  syscalls.
- `logit()` converts a probability back to log-odds, with `np.clip` guarding against
  `log(0)` or division by zero at the extremes (a probability of exactly 0 or 1 would
  otherwise blow up). This is what lets occlusion report a **drop in log-odds** rather
  than a raw probability difference — log-odds is the more natural scale for "how much
  did removing this evidence move the decision," since probability differences near 0
  or 1 compress non-linearly.
- `matrix_score` is the shared scoring primitive every explanation method calls: given
  a base token/count sequence and a batch of binary masks (one row per ablation/
  perturbation to test), it builds a batch of full-length model inputs (zeroing out
  masked-out positions) and runs one batched `model.predict()` call — batching multiple
  ablations into a single forward pass is why occlusion's cost stays low despite
  testing several behaviour groups and several individual syscalls.

## Cell 28 — the worked case

Hardcodes one specific alert (`CASE` dict: 16 March, `pro_id 2264`,
`/usr/lib/firefox/firefox`, category Exploits, a 22-entry window of syscall/repeat
pairs) rather than sampling one at runtime — this is the same case used across every
occlusion/attention/LIME figure in the paper, chosen once by hand for a clean,
human-legible illustration. Converts the case's syscall names to token IDs via the
saved vocabulary, builds `X_case`/`C_case` (single-example versions of the same tensor
shapes used in training), and scores it once (`base_score`) to report against the
frozen threshold.

## Cell 30 — occlusion

`ablate(labels)` takes an array of per-position labels (either the behaviour-group
array or the raw syscall-name array) and, for each **distinct** label value, builds one
mask that zeroes out every position carrying that label, then scores all such masks in
one batched call via `matrix_score`. `group_occ`/`syscall_occ` then compute
`base_logit - logit(score_after)` per ablation — the log-odds drop reported in Fig.
3.2 — and flag whether removing that group/call would flip the decision below
threshold (`clears_flag`). Two independent ablations are run in this cell: one over
behaviour groups (a handful of masks) and one over individual syscall names (as many
masks as there are distinct syscalls in the 22-position window) — this is why the
occlusion cost in RQ2 is described as "a small, fixed number of extra forward passes":
fixed by the number of *distinct* labels in the window, not by the window length itself.

## Cell 32 — attention

```python
mha_layers = [l for l in tmodel.layers if isinstance(l, tf.keras.layers.MultiHeadAttention)]
for l in mha_layers:
    src = l.input[0] if isinstance(l.input, (list, tuple)) else l.input
    hin = tf.keras.Model(tmodel.inputs, src).predict(x_case, verbose=0)
    _, att = l(hin, hin, return_attention_scores=True, attention_mask=key_mask[:, None, :])
    a = np.asarray(att)[0].mean(axis=0)
    recv += a[:n, :n].mean(axis=0)
recv /= recv.sum()
```
- `mha_layers` finds both `MultiHeadAttention` layer objects inside the already-trained
  model (one per encoder block, so 2 layers found).
- For each one: `tf.keras.Model(tmodel.inputs, src)` builds a **new, throwaway Keras
  model** whose output is that layer's own input tensor — running `.predict()` on it is
  what forces a fresh forward pass just to recover the hidden state feeding into this
  particular attention layer. This is the concrete mechanism behind "attention is not
  literally free" (`viva_qna.md` C3) — it's a full sub-model invocation per layer, not
  just reading a cached value.
- `l(hin, hin, return_attention_scores=True, ...)` calls the trained layer's own
  `__call__` a second time, this time asking it to also return its attention weight
  tensor `att` (shape roughly `(1, heads, 256, 256)` before slicing).
- `.mean(axis=0)` averages over the 4 heads. `a[:n, :n]` slices down to just the real
  (non-padded) `n=22` positions in this case. `.mean(axis=0)` again averages over query
  positions, leaving one attention-received value per key position — "how much
  attention did this position receive, averaged across every position that could have
  attended to it."
- Summed across both layers (`recv += ...` inside the loop), then renormalised
  (`recv /= recv.sum()`) so the final values are a share-of-total-attention percentage,
  which is what's plotted in Fig. 3.3.

## Cell 34 — LIME

```python
explainer_lime = LimeTabularExplainer(
    training_data=lime_background, feature_names=[...], mode="classification",
    discretize_continuous=False, random_state=42,
)
def lime_predict(mask_matrix):
    p = matrix_score(tokens, counts, mask_matrix.astype("float32"))
    return np.c_[1.0 - p, p]

lime_exp = explainer_lime.explain_instance(
    np.ones(n, dtype="float32"), lime_predict, num_features=n, labels=(1,)
)
```
- `lime_background` (built just above, not shown) is 512 random binary vectors used by
  LIME internally to characterise the "distribution" of the tabular feature space
  it's perturbing around — each of the `n=22` positions is treated as one binary
  present/absent tabular feature.
- `lime_predict` is the bridge between LIME's expected interface (a function taking a
  batch of binary masks, returning class probabilities) and this project's actual
  scoring function (`matrix_score`) — it wraps the single-column attack probability
  `p` into a two-column `[P(normal), P(attack)]` array, since LIME's classification
  mode expects one probability per class.
- `explain_instance(np.ones(n), ...)` — the instance being explained is "all positions
  present" (an all-ones mask, i.e. the actual alert as observed); LIME then internally
  generates its own perturbations around that (turning random subsets of positions off)
  by repeatedly calling `lime_predict`, and fits a linear model (Ridge regression, by
  the `lime` package's default) to those results. `num_features=n` asks for a weight on
  every one of the 22 positions rather than a sparse top-k subset.

## Cells 36 & 38 — cost measurement

`timeit(fn, repeats=10)` calls `fn()` once first (a warm-up call, discarded — this
matters on GPU especially, where the first call pays a one-off kernel-compilation/
memory-allocation cost that would otherwise bias the very first timed run), then times
`repeats` further calls with `time.perf_counter()` and averages. `run_occlusion`,
`run_attention`, and `run_lime` each re-run the exact explanation-generation code from
cells 30/32/34 (not a separate lightweight version) — so the RQ2 numbers measure the
real cost of producing the actual figures shown in the paper, not a synthetic proxy.
Cell 38 repeats the identical `cost_table()` call, but first reloads the model inside a
`with tf.device("/CPU:0")` context, which forces every subsequent op (including the
sub-model re-invocations attention needs) onto CPU regardless of GPU availability.
