# Architecture defense — every component, why it's there

A component-by-component walk through the model definition (`XAI_Final_Notebook.ipynb`
cell 20, reproduced line by line in `code_walkthrough.md`). For each piece: what it is,
why it was chosen, what the real alternatives were, and — this matters — whether the
choice was actually swept/tested in this codebase or is a standard architectural
convention applied at a sensible size for this problem. Don't blur that line under
questioning. Most of the individual numbers below (2 blocks, 4 heads, 64 dimensions,
128 feed-forward width, 0.3 dropout) were **not** the output of a hyperparameter search
in this codebase — they're conventional Transformer sizing choices, scaled down
sensibly for a small, 117-token vocabulary and a 256-length sequence. Where something
genuinely *was* chosen by search (the decision threshold), that's called out
explicitly as the one place an actual sweep happened.

```python
dm, heads, ff = 64, 4, 128
L = 256
# 2 encoder blocks (the for-loop below runs twice)
# dropout = 0.3
```

---

## 1. Why a Transformer encoder at all, at the top level

Covered in depth in `decision_defense.md` §3 — the short version: self-attention gives
an intrinsic, model-internal explanation channel (attention-as-explanation) that no
alternative in the shortlist (LSTM, CNN-LSTM, GNN, tree-based) offers without bolting on
a separate method, and the LSTM/CNN-LSTM baselines that were actually run showed
comparable detection performance, so the architecture switch didn't cost accuracy.
Everything below assumes that decision and defends what's *inside* the Transformer.

---

## 2. Why two encoder blocks (not one, not four)

**What it is:** the `for _ in range(2):` loop in cell 20 — the model stacks two
identical self-attention-plus-feed-forward blocks, each with its own independently
learned weights.

**Why two:** one block gives every position a single round of "look at every other
position and update yourself" — enough to pick up direct pairwise relationships (e.g.
"this `clone` follows that `poll`") but not enough to compose those relationships into
higher-order patterns (e.g. "this run of network calls, followed by a `clone`, followed
by another network burst, is the pattern that matters," which needs the *output* of one
attention round to be attended over again). A second block gives the model exactly one
level of that composition. This is a standard architectural intuition (depth lets a
network combine simpler learned relationships into more complex ones), not something
unique to this project.

**Why not more (4, 6, 12, the depths common in large language models):** those depths
are sized for problems with a large vocabulary, long-range dependencies across
thousands of tokens, and enough training data to fit many more parameters without
overfitting. This model has a 117-token vocabulary, a 256-length sequence, and a
training set that, while large in raw event count, produces a comparatively modest
number of *distinct traces*. Going deeper adds parameters and compute cost without a
matching increase in problem complexity to justify it — and every additional block
also means additional attention re-invocations at explanation time (see
`code_walkthrough.md` cell 32), which directly costs RQ2's latency budget for no
demonstrated accuracy benefit.

**Honest note:** no depth sweep (1 vs 2 vs 3 vs 4 blocks) was run in this codebase. Two
is a reasonable, conventional choice for a problem this size, not a tuned result — say
that plainly if asked whether 2 was empirically the best depth.

---

## 3. Why four attention heads

**What it is:** `MultiHeadAttention(num_heads=4, key_dim=dm // heads)` — instead of one
attention calculation per block, four run in parallel, each in its own 16-dimensional
slice of the 64-dimensional space, and their outputs are combined.

**Why multiple heads at all:** a single attention head has to compress "how relevant is
every other position" into one weighting scheme. Multiple heads let the model
potentially learn several *different* notions of relevance in parallel within the same
block — e.g. one head might end up tracking process-creation relationships, another
might track file-access bursts — without forcing a single head to represent all of
that at once. This is standard Transformer motivation, not specific to syscalls.

**Why four specifically, not one, two, or eight:** four divides the 64-dimensional
embedding evenly into 16-dimensional per-head subspaces (`64 / 4 = 16`), which is a
comfortable, round per-head size — plenty of dimensions for a head to represent a
distinct relationship, without the per-head size becoming so small (as it would at,
say, 8 or 16 heads on a 64-dimensional embedding, giving 8- or 4-dimensional heads)
that individual heads become too narrow to represent much. One or two heads would give
the model very little of the "parallel, different relevance patterns" benefit that
motivates multi-head attention in the first place.

**Honest note:** like block depth, the specific number 4 wasn't swept against 2, 6, or
8 in this codebase — it's a conventional choice that divides `dm=64` evenly and sits
in the normal range used for models of this embedding size in the literature.

---

## 4. Why an embedding dimension of 64

**What it is:** `dm = 64` — the width of every internal representation in the model:
the token embedding, the positional embedding, the count projection, and everything
flowing through the encoder blocks.

**Why 64:** it needs to be large enough to give the embedding table room to represent
117 distinct syscalls as meaningfully different vectors (64 dimensions is generously
more than enough capacity for 117 categories — a much smaller dimension, even single
digits, could in principle separate 117 discrete items), while staying small enough to
keep the model cheap to train and cheap to run explanations against repeatedly (every
occlusion ablation and every attention re-invocation is a forward pass through this
whole width). 64 is a conventional "small Transformer" width — a natural-language
Transformer with a 30,000-word vocabulary would need a much wider embedding to give
each word room to be distinguished; a 117-token vocabulary does not.

**Honest note:** not swept against, say, 32 or 128 in this codebase — chosen as a
sensible, round, small-model width given the vocabulary size.

---

## 5. Why the feed-forward width is 128 (not, say, 64 or 256)

**What it is:** inside each encoder block, after attention, the representation is
expanded from 64 to 128 dimensions (`Dense(128, relu)`), then projected back down to 64
(`Dense(64)`).

**Why expand at all:** the attention step is fundamentally about *mixing* information
between positions (a weighted combination of other positions' values) — it doesn't, on
its own, give the model much room to apply a nonlinear transformation to what it just
gathered. The feed-forward block gives each position, independently, a small
two-layer network to further process what attention just handed it, and the expand-
then-contract shape (64 → 128 → 64) is the standard Transformer feed-forward pattern,
giving the ReLU nonlinearity a wider intermediate space to work in before compressing
back to the model's working width.

**Why 128 specifically (a 2× expansion) rather than a larger multiple:** large
Transformers commonly use a 4× expansion (e.g. 512 → 2048); this model uses a smaller
2× ratio, consistent with its generally small overall size — a 4× expansion here (64 →
256) would be disproportionately large relative to a 64-dimensional model and a
117-token vocabulary.

**Honest note:** the ratio wasn't swept — 2× is a scaled-down, reasonable choice given
the model's small width, not an empirically tuned result.

---

## 6. Why token + positional + count embeddings are summed, not concatenated

**What it is:** `h = tok_emb + pos_emb + cnt_emb` — all three are the same 64-dimensional
shape and are added together, not stacked side by side into a wider vector.

**Why sum rather than concatenate:** concatenation would produce a `3 × dm`-wide vector
that then needs an extra projection layer back down to `dm` before it can enter the
encoder blocks (which expect a fixed 64-dimensional input) — summation gets the same
effect (all three signals present in the representation) without that extra learned
projection, and it's the standard way positional information is combined with token
identity in the original Transformer paper (Vaswani et al.) specifically, which this
architecture already follows for positional encoding — extending the same pattern to
the count channel keeps the design consistent.

**Trade-off, stated honestly:** summing assumes the three signals don't need to be kept
strictly separable in the representation — the model has to learn to disentangle "this
is about token identity" from "this is about position" from "this is about repeat
count" within a shared 64-dimensional space, rather than having dedicated dimensions
for each. Concatenation was not tried as an alternative in this codebase.

---

## 7. Why the repeat count goes through `log1p` before its Dense projection

**What it is:** `cnt_emb = layers.Dense(dm)(ops.log1p(cnt_inp))`.

**Why log-transform a count at all:** raw repeat counts are heavily right-skewed — most
runs repeat once or twice, a few repeat hundreds of times — and feeding that raw range
directly into a Dense layer would let a handful of extreme values dominate the learned
weights. A log transform compresses the large values down while leaving small values
close to their original scale, which is the standard treatment for skewed count data.

**Honest detail worth knowing cold:** the repeat count is *already* log1p-transformed
once during feature engineering (`Xcnt` in cell 19, `np.log1p(counts)`), and then
`ops.log1p(cnt_inp)` applies it a second time inside the model. The value actually
reaching the `Dense` layer is `log1p(log1p(repeat_count))`, not simply
`log1p(repeat_count)`. This is a real detail in the implementation, not a design
narrative — if asked to derive the count pathway precisely, say this plainly rather
than describing a single log transform.

---

## 8. Why a learned positional embedding rather than sinusoidal, relative, or rotary encoding

Covered in `decision_defense.md` §6 in the context of sequence length; the core
reasoning restated here: the sequence length is fixed at exactly 256 by construction
(every trace is padded or downsampled to this exact length), so there's no
variable-length or very-long-context generalisation problem that relative or rotary
position schemes are specifically designed to solve. A simple learned absolute
positional embedding — one trainable vector per position index, 0 through 255 — is
sufficient at this fixed length and keeps the model simpler than the alternatives.

**Honest note:** relative and rotary variants were not implemented or compared.

---

## 9. Why the padding mask is applied where it is

**What it is:** `amask` (attention mask) is computed once from `tok_inp != 0` and
passed into `MultiHeadAttention` at both encoder blocks; a second, separate masking
(`h * valid`) happens again just before pooling.

**Why mask attention specifically:** without it, every position — including the
meaningless zero-padding at the end of short traces — would be a valid target for other
positions to attend to, diluting real attention weight onto positions that carry no
information at all.

**Why mask again before pooling, when attention was already masked:** the residual
connections (`h + at`, `h + fw`) mean that even a padding position's own representation
persists and evolves through the network (it's never zeroed by the attention masking
itself, only prevented from being *attended to* by others) — so a second, explicit
masking step is needed right before the mean-pooling sum to make sure those
still-nonzero padding representations don't get averaged into the final trace vector.
This is a real, necessary second masking step, not a redundant belt-and-braces gesture.

---

## 10. Why residual connections around both sub-layers

**What it is:** `h = layers.LayerNormalization()(h + at)` and the equivalent for the
feed-forward step — each sub-layer's output is added back to its own input before
normalising.

**Why:** without a direct input-to-output shortcut, a signal has to pass successfully
through every layer's transformation to reach the end of the network, and gradients
during training have to flow backward through every one of those transformations too —
in deeper networks this makes training slow or unstable. The residual connection gives
the network an easy "do nothing extra" default (if a layer's output were exactly zero,
the residual path alone would carry the input through unchanged) and gives gradients a
direct path backward. Standard Transformer design, not project-specific.

---

## 11. Why layer normalisation, applied after the residual add (post-norm)

**What it is:** `LayerNormalization()` rescales the values at each position (not across
the batch, unlike batch normalisation) to a stable range, applied to `h + at` (i.e.
*after* the residual sum), not before it.

**Why normalise at all:** keeps the numbers flowing through the network in a
consistent, stable range at every layer, which makes training more stable and
generally faster to converge.

**Why post-norm (normalise after the residual add) rather than pre-norm (normalise
before each sub-layer, a common variant in more recent large Transformers):** post-norm
is the original Transformer paper's design and is a perfectly standard, well-understood
choice for a model of this size; pre-norm variants are more often reached for in very
deep networks (dozens of layers) where post-norm can become harder to train stably —
that concern doesn't really apply at 2 blocks.

**Honest note:** pre-norm was not implemented or compared here.

---

## 12. Why masked mean pooling rather than a CLS token, max pooling, or attention pooling

**What it is:** after the encoder blocks, the sequence of per-position vectors is
collapsed to one vector per trace by averaging over the real (non-padded) positions.

**Alternatives available:**
- **CLS-token pooling** (BERT-style): prepend an artificial extra token to the
  sequence whose final representation is used as the whole-sequence summary. Requires
  adding an out-of-vocabulary token and an extra position, and the model has to learn
  to route relevant information into that one specific token via attention.
- **Max pooling:** take the elementwise maximum across positions rather than the mean.
  Picks out the single strongest signal per dimension but discards information about
  how *widespread* a behaviour was across the trace.
- **Attention pooling:** a small additional learned attention layer specifically to
  weight positions for pooling, separate from the encoder's own self-attention.
- **Masked mean pooling** (what was used): a simple, parameter-free average over real
  positions.

**Why mean pooling:** it's the simplest option that requires no extra learned
parameters and no artificial token, and it naturally reflects the trace-level framing
of the whole study — the paper's own attention-as-explanation figures already read
attention *within* the encoder as "what did the model focus on," so adding a second,
separate attention-for-pooling mechanism would blur that story by giving the model two
different attention-like signals to explain, one of which (pooling attention) wouldn't
be the one shown in the paper's figures.

**Honest note:** CLS-token and max-pooling variants were not implemented or compared
here — mean pooling was the first and only pooling approach used in the final model.

---

## 13. Why dropout at 0.3, and only in one place

**What it is:** `layers.Dropout(0.3)` is applied once, to the pooled trace vector,
immediately before the final classification layer — nowhere else in the model.

**Why dropout at all:** randomly zeroing a fraction of values during training forces
the model not to over-rely on any single dimension of the pooled representation,
which helps prevent overfitting — a real concern here given the model sees a training
set with far more repetition of "normal" behaviour than genuinely diverse attack
behaviour.

**Why 0.3 specifically:** a conventional, moderate dropout rate — high enough to have a
real regularising effect, not so high (e.g. 0.5+) that it would meaningfully slow
learning on a model this size.

**Why only before the final layer, not inside the encoder blocks too:** the encoder
blocks already have residual connections and layer normalisation providing some
training stability, and adding dropout inside every attention/feed-forward sub-layer
(a common practice in larger Transformers) increases regularisation pressure that
wasn't judged necessary for a model this compact — one dropout layer, at the point of
highest information compression (right before the single final decision), was
judged sufficient.

**Honest note:** 0.3, and the single-location placement, were not swept against
alternatives (e.g. 0.1, 0.5, or dropout inside the encoder blocks) in this codebase.

---

## 14. Why a single sigmoid output unit, not a two-unit softmax

**What it is:** `layers.Dense(1, activation="sigmoid")` — one output number between 0
and 1, read directly as P(attack).

**Why not softmax over two classes (normal, attack):** for a strictly two-class
problem, a single sigmoid unit and a two-unit softmax are mathematically equivalent in
what they can represent — softmax over two classes reduces to the same decision
boundary as a sigmoid on the log-odds difference between the two classes. Sigmoid is
the simpler, more direct choice for binary classification specifically, and it's what
the paper's own log-odds framing (used throughout the occlusion explanations) builds
on directly — `logit()` in the explainability code converts this single probability
back to log-odds, which wouldn't be as natural a fit if the output were a two-way
softmax.

---

## 15. Why binary cross-entropy loss

**What it is:** `loss="binary_crossentropy"` — the standard loss function for a
single-probability, two-class prediction problem.

**Why:** it directly measures how far the predicted probability is from the true 0/1
label, penalising confident wrong answers more heavily than unsure wrong answers, which
is the correct behaviour for training a probability estimator. It's the standard,
default choice for this exact output shape (single sigmoid unit, binary label) — there
isn't a meaningfully different alternative loss for this specific setup short of
something like focal loss, which is a *modification* of cross-entropy for imbalance
(covered next), not a different loss family entirely.

---

## 16. Why class weighting, and why computed only from the training split

Covered in full in `decision_defense.md` §7. The one detail worth restating precisely
here: `cwS = {0: 1.0, 1: float((ys[trS] == 0).sum() / max((ys[trS] == 1).sum(), 1))}` —
the weight for the attack class is the ratio of normal-to-attack *traces in the
training split specifically*, recomputed fresh from `ys[trS]` each run, never derived
from validation or test labels. This matters if asked precisely how the weight is
computed: it's a data-derived ratio, not a fixed hyperparameter chosen by hand.

---

## 17. Why the Adam optimiser

**What it is:** `optimizer="adam"` — the algorithm that translates the loss gradient
into actual weight updates during training.

**Why Adam over plain stochastic gradient descent (SGD) or other alternatives (RMSprop,
AdaGrad):** Adam adapts its effective step size per parameter based on the recent
history of gradients for that parameter, which generally makes it more forgiving of
suboptimal learning-rate choices and faster to converge than plain SGD, without needing
extensive manual tuning. It's the default, reliable choice for training a Transformer
from scratch at this scale — not a project-specific finding.

**Honest note:** no optimiser comparison (Adam vs. SGD vs. RMSprop) was run here.

---

## 18. Why batch size 128

**What it is:** `batch_size=128` — the number of traces processed together before each
weight update during training.

**Why 128:** a conventional, round batch size that balances two things — too small a
batch (e.g. 8 or 16) makes each weight update noisy and training slower to converge;
too large a batch (e.g. 1024+) needs more memory and can sometimes generalise slightly
worse. 128 is a standard middle-ground default for a dataset of this size, not a
project-specific tuned value.

**Honest note:** not swept against alternatives in this codebase.

---

## 19. Why up to 25 epochs, with early stopping (patience 4) on validation PR-AUC

**What it is:** training runs for a maximum of 25 passes over the training data, but
stops early if `val_pr_auc` hasn't improved for 4 consecutive epochs, restoring
whichever epoch's weights were actually best.

**Why cap at 25 rather than a fixed, smaller number or an unbounded run:** 25 is a
generous upper bound intended to give the model enough opportunity to converge without
needing to guess the exact right number of epochs in advance — early stopping is the
actual mechanism deciding when training really ends, not the 25 figure itself.

**Why monitor `val_pr_auc` specifically, rather than validation loss or validation
accuracy:** PR-AUC is the threshold-independent metric this study treats as the primary
signal of ranking quality on an imbalanced problem (see `decision_defense.md` §9) — so
using it for early stopping keeps the stopping criterion consistent with how the model
is ultimately evaluated, rather than optimising against a different signal (like raw
validation loss, which is less informative on heavily imbalanced data) during training
and then judging it by PR-AUC afterward.

**Why patience 4:** a moderate patience — long enough to ride out a couple of noisy
epochs where PR-AUC dips before recovering, short enough not to waste much compute
training well past the point of real improvement.

**Honest note:** patience 4 and the 25-epoch cap were not swept against alternatives.

---

## 20. The one place an actual search happened: the decision threshold

**What it is:** `choose_threshold()` tries every unique score value that actually
occurs in the validation predictions as a candidate cutoff, computes the F2 score at
each one, and keeps whichever cutoff scores highest.

**Why this is different from everything above:** every other hyperparameter in this
document is a conventional choice, sized sensibly for the problem but not the result of
a search. The threshold is the one number in the whole pipeline that genuinely *is* the
output of an exhaustive search over actually-occurring candidate values — worth drawing
that contrast explicitly if asked "so nothing here was actually tuned" — the threshold
was, precisely and exhaustively, on the validation split only, and then frozen before
touching test data (`viva_qna.md` A4).
