# Decision defense — every major choice, its alternatives, and why

For each decision: what was chosen, what else was genuinely on the table, why this one
won, and — since you have two named examiners — which of them is more likely to push on
it and from what angle, based on their actual research backgrounds (see `viva_qna.md`
section H for the full profiles). Healy = deep learning / HCI / evaluation methodology
/ edge-transformer cost. Xu = NLP / CALL / games-VR / plain-language explanation.

Where "why this was chosen" rests on something actually measured in this codebase, that's
stated as fact. Where it's a reasonable design judgement call without an ablation behind
it, that's flagged explicitly — don't upgrade a judgement call into a measured result
under pressure.

---

## 1. Dataset — NGIDS-DS vs. NSL-KDD / UNSW-NB15 / CICIDS2017 / ADFA-LD

**Alternatives available:** NSL-KDD and UNSW-NB15 are network-flow datasets (features
are aggregated connection statistics, not raw behaviour) — most of the XAI-for-IDS
papers in Related Work use one of these two. ADFA-LD is host-based like NGIDS-DS, but
is pure syscall-ID sequences with no timing, no process metadata, and a much smaller,
cleaner attack set (this is the dataset Senoussi et al. [13] used). CICIDS2017 is a
modern network-flow dataset with more attack diversity but no host-level syscall detail
at all.

**Why NGIDS-DS:** The study is specifically about explaining *host-based* detection —
syscalls are legible, causally close to what a process actually did, and let occlusion/
attention operate on individually-meaningful units (a `clone` call means something; a
network-flow feature like "mean packet inter-arrival time" doesn't translate into a
security-analyst-legible explanation nearly as well). NGIDS-DS also ties every attack to
a real CVE and gives host-level granularity ADFA-LD doesn't (per-process, per-syscall,
with timing and category labels), and it lets you position directly against Haider's own
thesis and the Senoussi et al. Transformer paper.

**Healy's angle:** Given his evaluation-frameworks background, expect "is this dataset
still a fair benchmark in 2026, or should you have validated behaviour on something more
current (e.g. CICIDS2018/2021 successors)?" — honest answer: NGIDS-DS is from 2016/2018
and its age is a real limitation, not something to argue away; it's a *reasonable and
literature-consistent* choice for this specific host-syscall-explainability question, not
a claim that it represents 2026 attacker behaviour.

**Xu's angle:** Unlikely to push on dataset currency specifically; more likely to ask
what the syscall "vocabulary" looks like as a sequence-modelling input, which is answered
in `code_walkthrough.md` (cell 19) and `viva_qna.md` X1.

---

## 2. Granularity — trace-level vs. fixed window vs. per-event

**Alternatives available:** Per-event classification (label each syscall individually);
fixed-size sliding windows (e.g. 16 or 64 consecutive calls, as several papers in this
space do, including Haider's own length-5 windows); whole-trace classification (what was
built).

**Why trace-level:** A single event or a short window looks the same whether or not the
process is compromised — cell 11's own analysis shows no individual syscall's attack
rate departs meaningfully from the dataset baseline. The distinguishing signal is
distributional across the whole session. Run-length encoding plus evenly-spaced
downsampling to 256 positions (`code_walkthrough.md`, cell 19) is what makes "whole
trace" computationally tractable even for traces with tens of thousands of events.

**Trade-off, stated honestly:** trace-level means you cannot localise *where inside* a
long trace an attack happened — you get one score per whole process. That's a genuine
limitation, not hidden: it's consistent with treating "did we catch the intrusion" as
the operative question rather than "which exact syscall was the attack."

**Healy's angle:** Multimodal/behavioural-signal background — may ask whether
finer-grained (sub-trace) attention over time would be more informative for an analyst
than one trace-level score, drawing a parallel to how EEG-signal work often needs
sub-window temporal localisation, not just a whole-session label. Fair pushback; the
honest answer is that finer localisation is future work, not attempted here.

**Xu's angle:** Less likely to probe this one directly.

---

## 3. Model architecture — Transformer vs. LSTM / CNN-LSTM / GRU / GNN / tree-based

**Alternatives available and actually tried:** LSTM and CNN-LSTM baselines exist in the
repo (`architecture_baselines.ipynb`) — the original proposal named LSTM, CNN-LSTM, and
GNN as candidate architectures. Tree-based models (Random Forest, XGBoost — used by
several Related Work papers, e.g. [17][18]) and simple frequency/n-gram models were not
implemented as baselines here.

**Why the Transformer, specifically:** Three converging reasons, none of them alone
sufficient: (1) self-attention gives an *intrinsic* explanation channel to compare
against occlusion and LIME — no other architecture in the shortlist offers that without
bolting on a separate method; (2) it's directly comparable to Senoussi et al. [13], the
closest existing paper; (3) the LSTM/CNN-LSTM baselines that were run showed comparable
detection performance, so switching architecture didn't cost detection accuracy — see
`viva_qna.md` A1.

**Why not tree-based models:** Random Forest/XGBoost need hand-engineered features from
a sequence (e.g. call-frequency histograms), which reintroduces the "does a single call
carry signal" problem cell 11 already answered "no" to, and they don't offer a
sequence-position-level explanation the way attention does. This wasn't run as a
baseline — it's excluded by design reasoning, not by an experiment, and that should be
stated as such if asked.

**Healy's angle:** This is his strongest overlap (his own 2025 paper evaluates a hybrid
MobileViT-LSTM transformer for edge deployment). Expect a technically literate question
on the architecture itself — residual/layer-norm placement, why 4 heads specifically,
whether a smaller/cheaper architecture was considered given the RQ2 cost angle. Answered
in `viva_qna.md` A6 and `code_walkthrough.md` cell 20.

**Xu's angle:** More likely to draw the NLP-structural parallel (tokenise-embed-attend-
pool) than to challenge the architecture on ML grounds — see `viva_qna.md` X1/X2/X5.

---

## 4. Excluding program identity from the model input

**Alternatives available:** Include the program path/name as a categorical feature
(embedding), as almost any "normal" IDS feature set would; include it but down-weight it;
exclude it entirely (what was done).

**Why excluded:** Directly measurable in the data — Firefox has essentially zero normal
events in this dataset, and other victim programs are similarly skewed toward attack
windows. A model with access to program identity could reach a high score by memorising
"this is Firefox" rather than reading behaviour, and that shortcut wouldn't generalise
to a program the model hasn't seen attacked before. This is stated directly in the
paper's own methodology text (cell 12's markdown finding) and is a design decision
grounded in evidence, not caution for its own sake.

**Honest limit:** No identity-inclusive version of this exact Transformer was trained for
the final study, so there's no ablation number to quote for "how much would identity have
helped" — see `viva_qna.md` A2, which already covers exactly this follow-up.

**Healy's angle:** Possible angle from his multimodal-behaviour research: is dropping an
entire modality (identity) rather than learning to down-weight it a blunt instrument?
Fair challenge — the honest answer is that a soft down-weighting approach wasn't tried;
hard exclusion was chosen because the shortcut was severe enough (near-total confound)
that a soft solution felt riskier to get wrong than a clean design constraint.

**Xu's angle:** Unlikely angle for him specifically.

---

## 5. Feature representation — run-length encoding vs. raw sequence vs. n-grams / bag-of-calls

**Alternatives available:** Feed the raw, uncompressed syscall sequence directly (no
run-length collapsing); represent a trace as a bag-of-calls / n-gram frequency vector
(loses order entirely); use run-length encoding (what was done).

**Why run-length encoding:** A large share of trace length is literal repetition of the
same call in a row (e.g. repeated `poll` while a process is idle-waiting). Collapsing
those repeats into a (call, count) pair keeps the *order and identity* of behaviour
changes while not spending the fixed 256-position budget on redundant repeats of the
same call — this directly increases how much distinct behaviour fits inside the length
budget compared to feeding the raw sequence.

**Why not bag-of-calls / n-grams:** Cell 11's own analysis already shows individual call
identity and frequency carry very little signal on their own (attack rate close to
baseline for every top call) — an approach that discards order entirely would likely
perform worse, though this wasn't run as a head-to-head experiment; it's excluded by the
same evidence that motivated run-length encoding, not a separate ablation.

**Healy's angle:** Reasonable ML-methodology question: does collapsing repeat counts
into a single log-scaled channel lose timing information that might matter (his own
work on physiological signal timing might make this instinct sharper)? Honest answer:
yes, potentially — the model gets *frequency* of repetition but not the *precise timing*
between runs beyond that; timing-with-repetition is a real, un-explored trade-off.

**Xu's angle:** Could relate this to sequence compression choices in NLP/text
preprocessing generally (analogous to stemming/subword merging) — see `viva_qna.md` X1.

---

## 6. Fixed length 256 with evenly-spaced downsampling vs. truncation / sliding window

**Alternatives available:** Truncate every trace to the first (or last) 256 runs;
sliding-window classification with aggregation across windows; the evenly-spaced pick
across the whole trace that was implemented (`code_walkthrough.md`, cell 19, the `picks`
CTE).

**Why evenly-spaced sampling:** Truncating to the first or last 256 runs would
systematically hide behaviour elsewhere in a long trace (median normal trace length is
much shorter than many attack traces, so long traces are exactly where truncation would
bias the model). The evenly-spaced linear-interpolation pick spreads the 256 sampled
positions across the *entire* trace regardless of its length, so a long trace's
behaviour late in its life is as visible to the model as behaviour near the start.

**Honest limit:** This is a design choice grounded in reasoning about what truncation
would systematically miss — it was not compared head-to-head against a truncation
baseline in this codebase, so don't claim a measured improvement over truncation if
pressed; claim the reasoning.

**Healy's angle:** Could ask whether 256 was chosen by a sweep or by convention. Honest
answer: 256 is a fixed engineering choice for this study (a common sequence length in
Transformer literature), not the output of a length-sweep ablation in this codebase —
say that plainly rather than implying a tuning search happened.

**Xu's angle:** Could relate to fixed-length sequence padding/truncation choices common
in NLP pipelines — same underlying idea (`viva_qna.md` X1/X2 territory).

---

## 7. Loss and imbalance handling — class-weighted BCE vs. focal loss / SMOTE / oversampling

**Alternatives available:** Focal loss (used in an earlier, abandoned pipeline
per `MODEL_DECISIONS.md` — not in the final model); SMOTE/synthetic oversampling of the
minority class; random undersampling of the majority class; class-weighted binary
cross-entropy (what was done).

**Why class weighting:** It's the simplest mechanism that directly addresses the failure
mode (a model that predicts "normal" for everything would otherwise minimise loss, given
attacks are a small minority of traces) without synthesising data or discarding real
majority-class examples. It was sufficient — there was no need to layer on more
machinery once it worked.

**Honest limit:** Focal loss, SMOTE, and undersampling were **not** compared head-to-head
against class weighting in the final codebase (focal loss exists only in the abandoned
earlier pipeline, on a different model entirely). If asked "did you try X and did it do
better," the honest answer for all three is no, not on this final model — don't imply an
ablation exists that doesn't.

**Healy's angle:** Standard, fair ML-methodology question, likely from either examiner
honestly, but more naturally from Healy given his applied-ML background — answer exactly
as above, plainly.

---

## 8. Threshold selection — F2-maximising vs. default 0.5 / Youden's J / cost-based

**Alternatives available:** The default 0.5 cutoff; Youden's J statistic (maximises
`sensitivity + specificity - 1`, a common ROC-based choice); an explicit cost-weighted
threshold if false-negative and false-positive costs were known in dollar/effort terms;
F2-score maximisation on the validation split (what was done).

**Why F2:** In intrusion detection, missing an attack is worse than an extra false
alarm, so recall should be weighted above precision — F2 (β=2) encodes exactly that
preference without needing to specify an explicit cost ratio, which wasn't available.
Chosen on the validation split only, frozen before touching test data (`viva_qna.md`
A4).

**Why not Youden's J:** Youden's J treats false positives and false negatives as equally
costly (it's symmetric), which doesn't match the stated recall-first framing
([recall-first-operating-point] in the project's own memory/framing) — F2 is the more
honest encoding of the actual priority.

**Healy's angle:** Evaluation-methodology background — may ask why F2 specifically
rather than F1 or F3, or whether the β=2 weighting was itself chosen by search or by
convention. Honest answer: β=2 (F2) is a conventional choice for "recall matters roughly
twice as much as precision," not the output of a sweep over β values in this codebase.

---

## 9. Metrics — PR-AUC/recall/precision/FAR/accuracy vs. ROC-AUC / MCC / Cohen's kappa

**Alternatives available:** ROC-AUC (used far more often in the general ML literature);
Matthews Correlation Coefficient (MCC, often recommended specifically for imbalanced
binary classification); Cohen's kappa; the five metrics actually reported.

**Why PR-AUC over ROC-AUC:** With ~1.4% positive class, ROC-AUC is known to look
misleadingly good on imbalanced problems because the false-positive rate axis is
dominated by the huge negative class — PR-AUC is the standard, more honest choice for
rare-positive-class problems, which is exactly this dataset's shape.

**Why not MCC:** A legitimate alternative that wasn't used — MCC gives one summary
number balancing all four confusion-matrix cells, but the paper's choice to report
recall/precision/FAR/accuracy separately, plus PR-AUC as the threshold-independent
summary, gives an examiner more to inspect directly than a single MCC number would.
This is a reasonable-but-not-uniquely-correct choice; if pushed, MCC would be a fair
metric to add, not a reason the current metrics are wrong.

**Healy's angle:** Could plausibly ask this given his evaluation-framework research
strand specifically. Answer as above — don't defend PR-AUC as the *only* correct choice,
defend it as the standard, well-justified one for this class balance.

---

## 10. Split — temporal (day-based) vs. k-fold / random / stratified

**Alternatives available:** Random k-fold cross-validation (what Haider's original
thesis used); a stratified random split preserving attack/normal ratio; the temporal,
day-based split actually used (train 11–14 March, validate 15th, test 16th).

**Why temporal:** Respects time — no event from a later day can leak into training, and
no attack session gets split across train and test. It's also the realistic deployment
scenario (train on the past, detect on the future), and it's argued in `viva_qna.md` B2
to be *more* rigorous than Haider's own random CV protocol, not less.

**Honest limit:** A single held-out day (16 March) is one sample of "the future" — a
different test day might show somewhat different numbers, and this study doesn't
quantify that day-to-day variance (no repeated holdout across multiple different test
days). That's a real, statable limitation.

**Healy's angle:** Evaluation-methodology background makes this a natural target — might
ask exactly the day-to-day variance question above. Answer honestly: not measured here,
and it's a fair thing to flag as future work.

---

## 11. Explainability methods — Occlusion/Attention/LIME vs. SHAP / Integrated Gradients / Grad-CAM / counterfactuals

**Alternatives available:** SHAP (used by three Related Work papers — [16][17][18]);
Integrated Gradients (a gradient-attribution method, common for differentiable models);
Grad-CAM-style methods (mainly for CNNs, less natural fit here); counterfactual
explanations (show the minimal change that would flip the decision). Occlusion,
Attention, and LIME were chosen.

**Why these three specifically:** They produce genuinely different *kinds* of output —
exact ablation measurement, model-internal weights, and a local linear surrogate — which
is what lets the user study measure explanation *content* rather than presentation
style. SHAP was a reasonable fourth candidate but would have added a rating burden to a
17-person survey without necessarily broadening the mechanism diversity much further
(SHAP and LIME are both, at core, local-surrogate/attribution methods; occlusion and
SHAP's Shapley-value approach are both exact-ish measurement methods in spirit, so the
marginal diversity SHAP adds over the chosen three is smaller than it looks at first
glance).

**Why not Integrated Gradients:** IG needs a well-defined "baseline" input to integrate
from (commonly an all-zero input), which is a less natural concept for a sequence of
discrete tokens (an all-pad-token baseline isn't obviously meaningful the way an
all-black-pixel baseline is for images) — a defensible reason to skip it, not a rejection
of its validity in general.

**Healy's angle:** Given his own eye-gaze/attention work (VEAGLE), may specifically
probe the attention-as-explanation choice — this is already covered thoroughly in
`viva_qna.md` C3 and H6 (the "attention isn't causally faithful" critique). Know that
answer cold.

**Xu's angle:** Less likely to challenge the method selection itself; more likely to ask
about the *presentation* of explanations to non-expert users, tying to the "plain
language" open-feedback theme.

---

## 12. LIME formulation — tabular binary-feature-per-position vs. text explainer / custom sequence perturbation

**Alternatives available:** `LimeTextExplainer` (LIME's NLP-oriented variant, built for
word-level text); a fully custom perturbation scheme tailored to syscall sequences;
`LimeTabularExplainer` treating each position as a binary present/absent tabular
feature (what was done).

**Why tabular LIME:** It's the most direct fit for "one weight per position in a
fixed-length vector," which is exactly the model's own input shape, and it produces the
familiar per-feature weight table practitioners already recognise from other tools. This
is a repurposing of a tabular explainer onto a sequence problem — a fair thing to name
plainly if asked, not something to obscure (`viva_qna.md` C2 already covers this
directly).

**Why not LimeTextExplainer:** That variant is built around word/token boundaries in
natural-language text specifically (handling things like stemming and stop-words in a
way that doesn't map onto syscall tokens), and offers no real advantage here over the
simpler tabular formulation given the input is already a fixed-length vector, not raw
text.

**Xu's angle:** This is the one place his own field (NLP tooling specifically) gives him
a genuinely informed, sharp question — expect him to ask exactly this, and to know
enough about `LimeTextExplainer` to notice the tabular choice. Answer as above,
plainly — it's a defensible, reasoned choice, not an oversight.

---

## 13. User study design — rating survey vs. think-aloud / task-based evaluation

**Alternatives available:** A think-aloud protocol (watch participants reason through an
alert live); a task-based evaluation (measure time-to-correct-decision or decision
accuracy with vs. without each explanation); the rating-plus-single-pick survey actually
run.

**Why a survey:** Scales to more participants per unit effort than think-aloud sessions,
and directly answers RQ1 as posed ("which method do practitioners find understandable
and trustworthy") without needing to infer trust/understanding indirectly from behaviour.

**Honest limit — say this proactively if it comes up:** A rating survey measures
*perceived* usefulness/ease/trust, not *demonstrated* task performance — it doesn't
show whether Attention actually leads to faster or more accurate real-world triage
decisions, only that participants rated it more favourably. That's a genuine
methodological ceiling on what RQ1 can claim, and both examiners could reasonably raise
it — Healy from an evaluation-rigour angle, Xu from a "how do you know they actually
learned/understood, not just liked the format" angle close to his own education-tech
background.

**Follow-up you should have ready:** "Would a task-based design have been better?" —
Yes, arguably, for measuring real decision impact; it wasn't run here because it's a
substantially heavier study to design and recruit for, and RQ1 as posed asks about
perceived trust/understanding specifically, which the survey does answer directly.

---

## 14. Sample size and recruitment (n=17)

**Alternatives available:** A larger target sample (the standard HCI guidance for a
within-subject comparative study is often quoted around n≈20–30 for adequate power); a
smaller, deeper qualitative study (n<10 with interviews); n=17 with a mixed
developer/IT/analyst/student pool (what was run).

**Why n=17 wasn't pushed higher:** Practical recruitment constraints within a practicum
timeline — stated plainly as a limitation, not defended as sufficient. The paper's own
language ("no single method was rated significantly higher") already reflects this
honestly.

**Healy's angle:** Most likely examiner to ask for a concrete number here — "what would
adequate power have required, and did you compute it a priori?" Honest answer: no power
analysis was run before recruitment; the sample size was a practical recruitment outcome,
not a target derived from a power calculation. Say that directly rather than retrofitting
a justification.

---

## 15. Tooling — DuckDB vs. pandas / Spark / a relational database

**Alternatives available:** Load everything into pandas directly (infeasible at 10GB
raw / 90M rows without chunking); Apache Spark (proper big-data tooling, but heavy
operational overhead for a single-machine practicum); a conventional RDBMS (Postgres/
MySQL) with the data loaded in; DuckDB (what was used).

**Why DuckDB:** Runs in-process with no server to stand up, executes SQL (including the
window functions the run-length encoding pipeline needs) directly over parquet files
without loading everything into memory first, and compresses the working dataset from
~10GB raw to ~300MB parquet. It's the right-sized tool for a single-machine, single-user
analytical pipeline at this scale — Spark would be over-engineering for data that fits
on one machine's disk; pandas alone would require manual chunking to avoid loading 90M
rows into RAM at once.

---

## 16. Scope — no deployable client-server prototype

**Alternatives available:** Build the originally-proposed Flask/Kafka/ELK deployment
pipeline (client module for event forwarding, server module for detection + alerting);
skip deployment engineering entirely and spend that time on the detection model and
explainability comparison (what happened).

**Why:** Once the project's centre of gravity became "which explanation do
practitioners trust, and what does it cost" (RQ1/RQ2), the deployment stack stopped
being load-bearing for answering either research question — it would have consumed
engineering time without adding evidence toward the actual questions being asked. This
is covered in full in `viva_qna.md` F1; know it as a deliberate reallocation of effort,
not an admission of running out of time.

**Both examiners could raise this** as a general "does the work match the proposal"
question — it's not specific to either one's research background, it's a standard
scope-integrity question any examiner might ask.
