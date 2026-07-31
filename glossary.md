# Glossary — every term used in this project, explained simply

Plain-language definitions for every technical term that shows up across the paper,
the code, and the other prep documents. Each one is written so you could say it out
loud to a non-specialist examiner without sounding like you're reciting a textbook.
For deeper technical detail on any of these, see `code_walkthrough.md` (implementation)
or `viva_qna.md` (defending the choice). Security-specific attack terminology has its
own deeper treatment in `security_concepts.md` — the entries here are the short version.

---

## Machine learning basics

- **Model** — a mathematical function with adjustable internal numbers ("weights")
  that gets tuned on example data until it produces useful outputs. Here, the model
  takes a process's syscall sequence and outputs a probability that it's an attack.
- **Training** — the process of showing the model many labelled examples (syscall
  sequence → attack or normal) and adjusting its internal weights so its predictions
  get closer to the correct answer over time.
- **Epoch** — one full pass through the entire training dataset. This model trains for
  up to 25 epochs.
- **Batch** — training doesn't look at all examples at once; it looks at small groups
  ("batches") at a time and updates the weights after each group. Batch size 128 here
  means 128 traces are processed together before one weight update.
- **Loss function** — a single number that measures how wrong the model's predictions
  currently are. Training is, mechanically, just trying to make this number smaller.
  This project uses **binary cross-entropy**, a standard loss for "yes/no" predictions
  that punishes confident wrong answers more than unsure wrong answers.
- **Overfitting** — when a model gets very good at the exact examples it trained on but
  fails to generalise to new, unseen data — like memorising exam answers instead of
  understanding the subject. Guarded against here by testing on a day (16 March) the
  model never trained on.
- **Class imbalance** — when one outcome (here, "normal") vastly outnumbers the other
  ("attack") in the data. Left unaddressed, a model can score deceptively well just by
  always guessing the majority class. Handled here by "class weighting" (see below).
- **Class weighting** — telling the loss function to care more about getting the rare
  class (attacks) right, by penalising a wrong answer on an attack example more heavily
  than a wrong answer on a normal example, roughly in proportion to how rare attacks are.
- **Optimiser (Adam)** — the algorithm that actually decides how to adjust the model's
  weights at each training step, based on the loss. Adam is a widely-used, generally
  reliable default choice, not something specific to this project.
- **Early stopping** — automatically halting training if performance on a held-out
  validation set stops improving, to avoid wasting time (or overfitting) by training
  past the point of benefit. This model stops if validation PR-AUC hasn't improved for
  4 epochs in a row, and keeps whichever epoch's weights were actually best.
- **Hyperparameter** — a setting chosen by the person building the model (like batch
  size, number of layers, or embedding size) rather than learned automatically from
  data during training.

---

## Neural networks and deep learning

- **Neural network** — a model built from layers of simple mathematical units
  ("neurons") connected together, loosely inspired by how brain cells connect. Each
  connection has a weight; training adjusts those weights.
- **Layer** — one stage in the network's pipeline; data passes through one layer, gets
  transformed, and is passed to the next.
- **Embedding** — a way of converting a discrete, categorical thing (like a specific
  syscall, or a word) into a list of numbers (a "vector") that the model can do math
  with, learned automatically during training so that similar things end up with
  similar number-lists.
- **Activation function** — a small mathematical function applied inside a layer that
  lets the network learn non-simple (non-"straight-line") patterns. **ReLU** (used in
  this model's feed-forward layers) simply outputs zero for negative inputs and passes
  positive inputs through unchanged. **Sigmoid** (used at the very final output) squashes
  any number into a value between 0 and 1, which is exactly what you want for "output a
  probability."
- **Dropout** — a training trick that randomly switches off a fraction of the network's
  units on each training step, forcing the model not to over-rely on any single one of
  them. Used here at rate 0.3 (30% switched off) right before the final decision layer.
- **Residual connection** — adding a layer's input directly back onto its output
  (`output = input + layer(input)`), which makes very deep networks much easier to
  train by giving gradients a direct path backward. A standard, not project-specific,
  Transformer building block.
- **Layer normalisation** — a step that rescales the numbers flowing through the
  network to keep them in a stable, consistent range at every layer, which makes
  training faster and more stable.
- **Gradient / backpropagation** — the mechanism by which the model figures out *which
  direction* to adjust each weight to reduce the loss. You don't need to derive this
  live, but know the word: backpropagation is the standard algorithm every modern
  neural network training loop uses.

---

## Sequences, tokens, and the Transformer

- **Sequence** — an ordered list of items — here, the list of syscalls a process made,
  in the order it made them.
- **Token / tokenisation** — converting each raw item (each distinct syscall) into a
  small whole number ("token ID") the model can look up in its embedding table. This
  project's vocabulary has 117 tokens, built only from syscalls seen during training.
- **Vocabulary** — the fixed list of all distinct tokens the model knows about. Anything
  encountered later that isn't in this list gets mapped to a reserved "unknown" token
  rather than crashing.
- **Padding** — since the model expects every input to be exactly 256 positions long,
  shorter traces are filled out with a reserved "empty" placeholder (token 0) at the
  end, which the model is explicitly told to ignore.
- **Self-attention** — the core mechanism of a Transformer: for every position in the
  sequence, the model looks at every *other* position and decides how relevant each one
  is to understanding this position, then blends information accordingly. This is what
  lets the model connect a syscall near the start of a trace to one near the end
  directly, without having to pass information step-by-step through everything in
  between (which is what older architectures like LSTMs do).
- **Multi-head attention** — running several independent self-attention calculations
  ("heads") in parallel, each potentially picking up on a different kind of pattern,
  then combining their results. This model uses 4 heads.
- **Positional encoding / positional embedding** — since self-attention on its own has
  no built-in sense of order (it treats "first" and "last" the same unless told
  otherwise), a second embedding is added that encodes *where* in the sequence each
  position is, so the model can tell "early in the trace" from "late in the trace."
- **Encoder** — the "read and understand the input" half of a Transformer, as opposed
  to a "decoder," which generates new output step by step (used in things like
  translation or text generation). This project only needs an encoder, because it's
  classifying a whole trace, not generating new sequences.
- **Pooling** — the step that compresses a whole sequence of vectors (one per position)
  down into a single vector representing the entire trace, right before the final
  yes/no decision. This project uses **masked mean pooling** — averaging over only the
  real (non-padding) positions.
- **Softmax** — a function that converts a list of raw numbers into a set of
  probabilities that all add up to 1. Used inside attention to turn "relevance scores"
  into proper weights.

---

## Evaluation metrics

- **Accuracy** — the percentage of all predictions (both attack and normal) that were
  correct. Can be misleading on imbalanced data — a model that always says "normal"
  would still score high accuracy here, since normal traces are the large majority.
- **Confusion matrix** — a simple 2×2 table showing every combination of
  predicted-vs-actual outcome: true positives (correctly caught attacks), true
  negatives (correctly cleared normal traces), false positives (normal traces wrongly
  flagged), false negatives (attacks the model missed).
- **True/false positive/negative** — "positive" means the model predicted "attack";
  "true" means that prediction was correct, "false" means it wasn't. A false negative
  is a missed attack; a false positive is a false alarm.
- **Recall** — of all the *real* attacks, what fraction did the model catch? The metric
  this project prioritises, because missing a real attack is the costlier mistake.
- **Precision** — of all the alerts the model actually raised, what fraction were real
  attacks? Low precision means analysts waste time chasing false alarms.
- **False alarm rate (FAR)** — of all the *real normal* traces, what fraction got
  wrongly flagged as an attack? A different way of asking "how often do you cry wolf."
- **F-beta score (F1, F2)** — a single number that blends precision and recall
  together. F1 weighs them equally; F2 (used for this project's threshold) weighs
  recall twice as heavily, matching the priority of not missing attacks.
- **Threshold** — the cutoff probability above which the model's output counts as
  "attack." The model outputs a number between 0 and 1; something has to decide where
  the line is drawn. This project picks that line by maximising F2 on validation data.
- **ROC curve / ROC-AUC** — a standard way of showing the trade-off between catching
  more attacks and raising more false alarms as the threshold changes, summarised into
  one number (AUC). Not used as the headline metric here because it can look
  artificially good on heavily imbalanced data.
- **PR-AUC (a.k.a. Average Precision)** — the same idea as ROC-AUC, but plotting
  precision against recall instead — the standard, more honest choice when the positive
  class (attacks) is rare, which is why this project reports it instead of ROC-AUC.

---

## Explainability (XAI)

- **Explainability / interpretability** — the general goal of making a model's decision
  understandable to a human, rather than just trusting a number it outputs.
- **Black box** — a model whose internal reasoning isn't visible or understandable from
  the outside — you can see what goes in and what comes out, but not why. Deep learning
  models are the classic example, which is the whole motivation for this project.
- **Log-odds / logit** — an alternative way of expressing a probability that behaves
  more smoothly near the extremes (close to 0 or close to 1). Occlusion in this project
  reports *drops in log-odds* rather than raw probability differences because it makes
  small-but-meaningful shifts near the decision boundary easier to compare.
- **Occlusion** — an explanation method that works by literally removing (blanking out)
  part of the input, then checking how much the model's prediction changes. A large
  change means that part mattered a lot.
- **Attention (as an explanation)** — reading the attention weights a Transformer
  already computes internally as a signal of which parts of the input it focused on.
- **LIME (Local Interpretable Model-agnostic Explanations)** — an explanation method
  that creates many slightly-altered versions of one input, sees how the model's
  prediction changes across all of them, and fits a simple, easy-to-read model to that
  pattern — giving an approximate, local picture of what mattered for that one
  prediction.
- **Surrogate model** — the simple model (in LIME's case, a linear model) fit to
  approximate a more complex model's behaviour in one small neighbourhood, so a human
  can read it directly.
- **SHAP** — a different, more computationally expensive explanation method (not used
  in this project) based on a game-theory concept called Shapley values, used by
  several of the papers this project compares against.
- **Model-agnostic vs. model-specific** — a model-agnostic method (occlusion, LIME)
  works on any model, treating it as a black box you can only query. A model-specific
  method (attention) only works because this particular architecture happens to expose
  something readable internally.
- **Ablation** — removing or disabling one part of something (an input feature, a
  model component) to measure how much that part mattered, by comparing behaviour
  with and without it. Occlusion is a form of ablation applied directly to the input.

---

## Data engineering

- **SQL** — a standard query language for asking questions of structured, table-shaped
  data ("give me all rows where X"). Used throughout this project's data pipeline.
- **DuckDB** — the specific tool used to run SQL queries directly over the dataset
  files, without needing to set up a separate database server.
- **Parquet** — a compressed, column-oriented file format for storing large tables
  efficiently — much smaller and faster to query than the raw CSV files the dataset
  originally came in.
- **Run-length encoding** — compressing a sequence by collapsing consecutive repeats of
  the same value into one (value, count) pair — e.g. "poll, poll, poll" becomes
  "poll ×3" — which is exactly what's done to each trace before feeding it to the model.
- **Downsampling** — reducing the number of items in a sequence when it's longer than
  the model's fixed input size, by selecting a representative subset rather than
  keeping everything.
- **Window function (SQL)** — a type of SQL calculation that looks across a group of
  related rows (e.g. "everything in this same trace, in time order") to compute
  something like a running count or a comparison to the previous row, without
  collapsing those rows into one. This is the SQL feature the run-length-encoding
  pipeline relies on most heavily.

---

## Quick note on security terms

Attack-type definitions (Exploits, DoS, Generic, Backdoors, Shellcode, Worms,
Reconnaissance) and core security vocabulary (vulnerability, exploit, payload,
shellcode, CVE, IDS vs. IPS, signature-based vs. anomaly-based, zero-day, syscall) are
covered in full, with real examples from this dataset, in `security_concepts.md` — not
repeated here to avoid the two documents drifting out of sync with each other.
