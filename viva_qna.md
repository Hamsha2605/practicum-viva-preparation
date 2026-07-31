# Viva question bank

Rebuilt from the actual submitted paper (`final/Practicum_Final_Paper.pdf`) and the
actual final notebook (`src/notebook/XAI_Final_Notebook.ipynb`), not from the earlier
draft notes in `docs/documentation/viva_prep.md`, which describe an abandoned
CNN-LSTM/window-level/identity-included model and numbers that never made it into the
final work. Do not reuse figures from that file or from `src/notebook/FINDINGS.md` /
`MODEL_DECISIONS.md` — those describe a different, earlier pipeline (window-level
CNN-LSTM with program-identity embeddings) that was replaced by the trace-level
Transformer reported in the final paper. If a number below isn't in the final paper or
isn't something the notebook actually computes, it's marked as such rather than
invented.

Rule for all answers: state the honest position first, don't over-claim, and where
something genuinely is a limitation, say so plainly rather than dressing it up.

---

## A. Detection model design

### A1. Why a Transformer encoder rather than an LSTM or CNN-LSTM?

**Say:** Three reasons. It's an explainability study, and self-attention gives an
intrinsic explanation channel to compare against occlusion and LIME — an LSTM or a
frequency-based model doesn't offer that for free. It's current practice for sequence
modelling, and there's a directly comparable recent paper (Senoussi et al., a
Transformer encoder on ADFA-LD) to position against. And it's an encoder-only design
because the task is classification of a whole trace, not sequence generation — nothing
needs decoding.

**Follow-up — "Your proposal mentioned LSTM, CNN-LSTM and GNN. Isn't the Transformer a
deviation?"** We did run LSTM and CNN-LSTM baselines — they're still in the repo
(`architecture_baselines.ipynb`) — and the detection difference against the Transformer
was small enough that it wasn't worth reporting as a separate contribution. We kept the
Transformer as the one detector in the paper because the study's contribution is the
explanation comparison, and only the Transformer gives us the attention channel that
comparison needs. The supervisor was informed of the direction; we don't treat this as
an unauthorised departure, just a refinement once the actual research question (RQ1/RQ2)
crystallised around explainability rather than raw detection accuracy.

**Do not:** claim the Transformer detects meaningfully better than the baselines — that
specific comparison isn't in the final paper, so don't invent a number for it live.

---

### A2. Why exclude the program name / process identity from the input?

**Say:** Because in this dataset the victim programs appear almost only during attacks —
Firefox has effectively no normal events at all in the logs. A model with access to the
program name could get a high score just from "this is Firefox," without reading any
actual behaviour. Leaving it out forces the model to learn from the syscall sequence
itself, which is the more honest and more generalisable signal.

**Follow-up — "Doesn't that throw away useful context?"** In a deployment with a much
larger and more balanced population of programs, identity could be legitimate context.
Here it's a shortcut created by how the dataset was generated, not a real detection
signal — a model that leans on it would miss an exploit in a program it hasn't seen
attacked before, and would misfire on any ordinary use of Firefox it never observed. So
excluding it is the choice that generalises, even though it's the harder path.

**Follow-up — "Do you have a number for how much identity would have helped?"** Not one
we're prepared to defend. We didn't train an identity-inclusive version of this
Transformer for the final study, so we're not going to quote an ablation figure for it —
the reasoning above is qualitative and dataset-driven, not backed by a specific
head-to-head number in this codebase.

---

### A3. Why trace-level (256 run-length-compressed positions), not raw events or fixed windows?

**Say:** We run-length encode the sequence first, so repeated calls collapse into one
token with a count, and then represent the whole trace at up to 256 positions —
downsampling evenly if a trace has more runs than that, padding if fewer. That keeps the
model looking at an entire process's behaviour rather than an arbitrary short slice of
it, which matters because a single event or a short window in isolation looks the same
whether or not the process is under attack.

**Follow-up — "Why compress with run-length encoding specifically?"** A lot of the trace
length is repetition — the same syscall fired many times in a row. Run-length encoding
keeps that information (as a count) while letting the model spend its 256 positions on
where the behaviour actually changes, rather than on redundant repeats of the same call.

---

### A4. Why the F2-maximising threshold instead of the default 0.5?

**Say:** In intrusion detection, missing an attack is worse than raising an extra false
alarm, so we favour recall. We select the threshold on the validation day as the value
that maximises F2 — which weights recall twice as heavily as precision — and then apply
that exact threshold, unchanged, to the test day. We never touch the threshold after
seeing test results.

**Follow-up — "Isn't that just tuning the result to look better?"** No — the tuning
happens entirely on the validation split, which is a day the model has already trained
on data before but the threshold search never sees test labels. The number we report
(75.0% recall) is what that frozen threshold produces on an unseen day, not a
best-of-many test-set search.

---

### A5. Why class-weighted binary cross-entropy rather than focal loss or resampling?

**Say:** Attack traces are rare — roughly 1.4% of raw events, and a minority of traces
overall — so the loss up-weights the positive class in inverse proportion to its
frequency in the training split. That's a standard, simple way to stop the model from
minimising loss by just predicting "normal" for everything, and it's what's actually
implemented in the final training cell. We didn't use focal loss or synthetic
oversampling in this version of the model — if asked whether those were tried, the
honest answer is that class weighting was sufficient and we didn't need to layer on
more machinery.

---

### A6. Walk me through the architecture in one breath.

**Say:** Two inputs per position — a syscall token and a log-scaled repeat count.
Each is embedded to 64 dimensions, summed with a learned positional embedding, giving
one 64-d vector per position. That goes through two encoder blocks: 4-head self-attention
with padding masked out, residual + layer norm, then a feed-forward layer (64 → 128 →
64), residual + layer norm again. Masked mean pooling over the real (non-padded)
positions gives one vector for the whole trace, then dropout (0.3) and a single sigmoid
unit for P(attack).

---

## B. Data and evaluation methodology

### B1. Why NGIDS-DS?

**Say:** It's a host-based dataset of real system-call traces with attacks tied to real
CVEs, generated on a realistic setup, and it's an established reference point in the
host-IDS literature (it's also what the Senoussi et al. Transformer paper and Haider's
own thesis work from). That fits a host-based, syscall-level study directly.

### B2. Why a temporal (day-based) split instead of random k-fold cross-validation?

**Say:** Training on days 11–14 March, validating on the 15th, testing on the 16th — a
day the model has never trained on — respects time. A random split could put events
from the same attack session, or even the same process trace, into both train and test,
which leaks information and inflates the apparent score. The temporal split is closer to
how the detector would actually be used: trained on the past, tested on the future.

### B3. What do the two cleaning steps actually remove, and why?

**Say:** Duplicate logged events, plus two syscalls — `clock_gettime` and
`gettimeofday` — that together make up 52% of all events in the raw data. Their rate
inside attack traces is the same as their rate everywhere else, so they add volume
without adding any separating signal. Removing both took the dataset from
90,054,239 events to 42,647,584.

### B4. Explain the five reported metrics and why all five, not just accuracy.

**Say:** Accuracy alone is misleading on an imbalanced problem like this — predicting
"normal" for everything would already score over 98%. Recall (TP/(TP+FN)) tells you what
fraction of real attacks you caught. Precision (TP/(TP+FP)) tells you how much you can
trust an alert once it fires. PR-AUC integrates that trade-off across every possible
threshold, so it isn't a property of the one cutoff we picked. FAR (FP/(FP+TN)) is the
operational cost — how often a normal trace gets flagged. Reporting all five together,
rather than just accuracy, is the honest way to describe an imbalanced detector.

### B5. Read me the confusion matrix.

**Say:** 1,548 test traces, 128 of them genuine attacks. 1,394 true negatives, 26 false
positives, 32 false negatives, 96 true positives. That's 75.0% recall (96 of 128 caught),
78.7% precision (96 of the 122 raised alerts were real), 1.8% false alarm rate (26 of
1,420 normal traces), 96.3% accuracy overall, and 0.79 PR-AUC.

### B6. Why did the per-category attack breakdown not make it into the paper, even though the code computes it?

**Say:** The notebook's `category_recall_table` function does compute recall per attack
category internally, and we looked at it during development. We chose not to publish it
as a table because the trace key we use — date, process id, path — can collide between
unrelated process launches on a busy host, since process ids get recycled. That makes a
"this trace had multiple attack categories" reading potentially an artefact of key
collision rather than a genuine multi-stage attack, and we didn't want to publish a
category breakdown built on a trace identity we don't fully trust. It's an honest scope
decision, not an oversight.

### B7. Results vary slightly when you re-run the notebook — why, and is that a bug?

**Say:** It's not a data pipeline bug — we checked that specifically. The DuckDB feature
pipeline uses explicit `ORDER BY` in every windowed/aggregated query, so the input to
the model is deterministic given the same raw data. The variation comes from
TensorFlow/cuDNN: the attention and softmax kernels used on GPU aren't bit-for-bit
deterministic even with a fixed random seed. Given that, we treat one frozen, saved
run — the model and evaluation data actually written to disk
(`final_models/trace_transformer/`) — as the canonical numbers reported in the paper,
rather than chasing exact reproducibility across GPU runs.

### B8. How do you know the worked example (the Firefox trace) is a genuine attack, and can you show the CVE behind it?

**Say:** The label comes from the dataset's own event-level annotation — every one of
that trace's 54 events carries `attack_cat = Exploits`, `attack_subcat = Browser`,
`label = 1` in `host_logs.parquet`, the same file our model trains and is evaluated on.
That's clean and unambiguous, not a partially-diluted trace where only a couple of
events happen to overlap an attack window.

There's a separate file in the dataset, `ground_truth.parquet`, that records individual
CVE-level attack events with their own timestamps and category names. We checked
whether we could pull one specific CVE out of it for this exact trace, and we couldn't
do it cleanly — that file uses a different, finer subcategory vocabulary (e.g.
"Clientside Microsoft Paint" rather than "Browser") and different timestamp granularity
than the host log, so a manual cross-reference didn't land on a clean 1:1 match. We
don't read that as either file being faulty — it's two independently-structured files
that were never guaranteed to join cleanly on category name and timestamp alone — but
we're not going to claim a specific CVE for this trace on the strength of that
cross-check, since it didn't hold up.

**Follow-up — "So can you name the CVE or not?"** No, and we're not going to guess one
live. The result we report doesn't depend on identifying that CVE; it depends on the
event-level attack label in the file the model actually trains and is scored against,
which is solid on its own.

**Do not:** say "the ground truth file was not clean" as a blanket claim — that invites
"not clean how?" and we don't have evidence for a data-quality defect in that file,
only a structural mismatch between two files' naming and timestamp granularity.

---

## C. Explainability methods

### C1. Why these three methods — occlusion, attention, LIME — and not SHAP or Integrated Gradients?

**Say:** We wanted three methods that produce visibly different kinds of output, so the
user study measures the content of an explanation, not its presentation. Occlusion gives
an exact, model-agnostic ablation measurement. Attention gives a free-ish, model-specific
read of what the Transformer itself focused on. LIME gives a locally-fit linear
surrogate, which is the format a lot of practitioners already recognise from other
tools. SHAP and Integrated Gradients would have been reasonable additions, but three was
already enough to get a meaningful comparison without fatiguing a 17-person survey with
a five-way rating exercise, and it kept every explanation directly attributable to a
distinct underlying mechanism (exact ablation vs. model-internal weights vs. local
surrogate).

### C2. Is LIME actually a good choice here, given tabular LIME is built for tabular features, not sequences?

**Say:** We treat each position in the run-length-compressed trace as a binary feature
(present/removed) and use `LimeTabularExplainer` over that representation, generating
perturbed traces by dropping random subsets of positions and scoring each with the
detector. It works, and it produces the familiar per-feature weight table practitioners
expect — but it is a repurposing of a tabular explainer onto a sequence problem, and
that's a fair thing to push on. The honest answer is that it's the standard way LIME
gets applied to sequence/tabular-adjacent problems in this literature, and our own
results (LIME's picks broadly agreeing with occlusion and attention on the same case)
suggest it's producing a sensible signal despite that mismatch, not that the mismatch is
irrelevant.

### C3. Is attention actually free, since it comes from the model's own forward pass?

**Say:** No — and we deliberately don't claim that in the paper. To extract the
attention weights we re-invoke the attention layers with `return_attention_scores=True`,
which means running a sub-model up to each encoder block's input and calling that layer
again outside the normal single forward pass used for prediction. That's why it measures
at roughly 170–200ms in our cost table — comparable to occlusion, not zero. What is true
is that it needs no separate procedure like LIME's perturb-and-refit loop; it reads
something the model structurally already computed, it just isn't literally free to
extract in this implementation.

**Follow-up — "Could you make it actually free?"** In principle, yes — redesigning the
model to return attention scores as a second output on the same forward pass used for
prediction would avoid the extra invocation. We didn't do that for this study; it wasn't
worth restructuring the model for a latency difference that doesn't change the
qualitative RQ2 conclusion (Attention and Occlusion are both cheap, LIME is the outlier).

### C4. Why does "clone" dominate the attention and occlusion examples?

**Say:** `clone` is a process-creation call — in the case we present (a Firefox exploit
trace), the spikes in both attention and occlusion land on the same positions, which is
a reassuring cross-check between two independently-computed methods, not something we
engineered. Process creation is a natural place for an exploit chain to leave a
footprint, since exploitation typically culminates in spawning a new process.

---

## D. RQ1 — user study

### D1. Who did you survey, and how many?

**Say:** 17 participants: 6 software developers, 4 IT professionals, 3 security
analysts, 4 students, spanning no cybersecurity/deep-learning background up to expert.
They rated all three methods on usefulness, ease, and trust, then picked the single most
effective method overall.

### D2. Is 17 a large enough sample?

**Say:** It's a modest sample, and we don't pretend otherwise. It's why the paper's own
language is careful — "no single method was rated significantly higher... though
Attention was chosen as the most effective by the majority" — rather than claiming a
strong statistically significant winner. With three repeated-measures conditions and
n=17, we treated the numbers as an honest descriptive comparison (means, and a
majority-preference count) rather than leaning on a significance test that would have
very limited power at this sample size to make a real claim either way.

### D3. What did the numbers actually show?

**Say:** Attention scored highest on usefulness and ease (7.29 each), occlusion lowest
on those two (6.24, 6.59). Trust was close across all three, 6.88 to 7.00 — meaning
people trusted the underlying model's decision about the same amount regardless of which
explanation they saw, which tells us the explanation format changed legibility, not
belief in the model itself. On the single-pick question, 53% chose Attention, 29%
Occlusion, 18% LIME.

### D4. How did you avoid bias in which explanation "looks" best?

**Say:** All three methods were shown for the same alert, in the same card format, so
participants were judging content rather than which one had nicer formatting or more
polish.

### D5. What did the open feedback say, and what does it change?

**Say:** Three recurring asks: a plain-language explanation alongside the visuals for
less technical users, a clearer baseline or threshold to judge the raw numbers against,
and a similarity score against previously seen attacks. We didn't build any of those
into this study — they're explicitly framed as future work, not retrofitted into the
current results.

---

## E. RQ2 — cost of explainability

### E1. What exactly did you measure?

**Say:** Wall-clock latency, averaged over 10 runs, for a baseline prediction alone
versus prediction plus each explanation method, on both GPU (RTX 3090) and CPU (same
machine, GPU execution disabled).

### E2. Summarise the result.

**Say:** On GPU everything is fast — baseline 86.9ms, Attention +188.4ms, Occlusion
+171.1ms, LIME +431.8ms, all comfortably under a second total. On CPU, Attention and
Occlusion barely change (196.9ms and 195.3ms added) because both need only a small,
fixed number of extra forward passes. LIME jumps to 11,583.3ms — over 11 seconds — 
because it scores many perturbed samples individually to fit its surrogate, so its cost
scales with sample count rather than staying fixed, and it depends heavily on GPU
parallelism to stay fast.

### E3. Does this change which method you'd recommend deploying?

**Say:** It's a real factor, not just an academic footnote. If explanations need to run
at alert volume on commodity CPU hardware, an 11-second-per-alert method is a genuine
deployment blocker in a way a sub-300ms method isn't. Combined with RQ1 — where
Attention was also the user-preferred method — the two research questions point the same
direction rather than trading off against each other, which is a reasonably clean result
to land on.

---

## F. Scope, limitations, and proposal alignment

### F1. Your original proposal described a deployable client-server prototype (Flask, Kafka, ELK). Where is it?

**Say:** It wasn't built, and we're upfront about that rather than implying otherwise.
The practicum's actual engineering time went into the detection model iteration and the
explainability comparison — the two research questions this study answers — rather than
a deployment stack that wouldn't itself have answered either RQ1 or RQ2. Building a
production client-server pipeline was a reasonable ambition at proposal stage, but once
the project's centre of gravity became "which explanation do practitioners actually
trust, and what does it cost," that infrastructure stopped being load-bearing for the
research questions, so we prioritised depth on those over breadth into deployment
engineering.

### F2. What would you flag as this study's main limitations?

**Say:** Three, plainly: the user study is 17 participants, which limits how strongly we
can claim a "winning" explanation method; the explanation comparisons are built on
individual flagged-trace case studies rather than an aggregate evaluation across every
alert type; and everything is evaluated on one dataset (NGIDS-DS) with one architecture,
so generalising the specific numbers to other host telemetry or other models is future
work, not a claim made here.

### F3. Did you consider deep learning survey/background papers when framing "black box" concerns, or is that a throwaway line?

**Say:** It's grounded — the paper cites three sources together for that claim ([14],
[16], [17]), including a 2025 systematic review on explainable AI-based IDS for
Industry 5.0. It's a well-supported framing claim, not an unsupported aside.

### F4. Your proposal's RQ2 asked how explainability affects "accuracy and response time." The final paper only reports latency. Did explainability change accuracy — and why did the RQ wording change?

**Say, directly:** No, it didn't change accuracy, and that's not an oversight — it's a
finding worth stating plainly. Explainability is applied post-hoc, after the detector
has already made its prediction; occlusion, attention, and LIME all explain a decision
the model already reached, they don't feed back into or alter it. So there's no
accuracy number to report "with vs without explainability" because the detector's
output is identical either way — the only thing explainability adds is time. That's
exactly why the final RQ2 narrows to computational cost specifically: it's not a
retreat from the original question, it's the honest answer to it. "How does
explainability affect accuracy" turned out to have a one-word answer — it doesn't —
so the research question sharpened to the part that actually had a measurable, variable
answer: latency, on GPU versus CPU, per method.

**Also be ready for — "your proposal's RQ1 said 'best suited' and named SHAP/LIME as
example methods; the final paper doesn't use SHAP at all and doesn't crown a single
'best' method."** Two honest, separate answers: first, the proposal's "how will you
explore this" section named SHAP and LIME only as *examples* ("eg: SHAP, LIME, etc"),
not a fixed commitment — the final method set (Occlusion, Attention, LIME) was chosen
instead because it gives three explanations with visibly different underlying
mechanisms (exact ablation, model-internal weights, local surrogate) rather than two
methods (SHAP, LIME) that are both, at core, local-attribution/surrogate techniques —
see `decision_defense.md` §11 for the full reasoning. Second, "best suited" in the
proposal implied a single winner; the final study deliberately reports a majority
preference (53% Attention) rather than declaring one method objectively "best," because
with n=17 and no significant difference in the rating scores, declaring a definitive
winner would overstate what the data supports. The RQ narrowed in wording, but the
underlying question — which method do practitioners prefer — is the same one being
answered, just with a more careful, evidence-appropriate claim attached to the answer.

**Do not:** get caught implying the RQ2 wording change was hidden or convenient — say
directly that accuracy doesn't change under a post-hoc explainability layer, and that
this is *why* the question sharpened, not something worked around.

---

## G. Conceptual / "explain it like I'm not in this field" questions

### G1. In plain terms, what does self-attention do?

**Say:** Every position in the sequence is compared against every other position: each
position is turned into a query, a key, and a value, and the similarity between a
query and all the keys decides how much each position's value contributes to that
position's output. Unlike an LSTM, which processes the sequence step by step, attention
relates distant positions directly and in parallel — which is also what makes the
weights themselves readable afterward as an explanation.

### G2. What's the difference between occlusion and LIME, in one sentence?

**Say:** Occlusion measures the model exactly, by actually removing input and
re-scoring; LIME approximates the model, by fitting a simple linear surrogate to how it
behaves on a batch of perturbed inputs around one prediction.

### G3. What is masked mean pooling, and why mask?

**Say:** Traces shorter than 256 positions are padded with zeros to reach a fixed
length. Masked mean pooling averages only over the real, non-padded positions when
collapsing the sequence to one vector, and the same mask is used inside attention so
padding positions can't be attended to at all — otherwise the padding would dilute both
the attention weights and the pooled representation with meaningless zeros.

### G4. Why DuckDB instead of pandas/SQL-on-something-else for the data pipeline?

**Say:** The raw dataset is around 10GB across 99 CSV files; DuckDB lets us run
SQL directly over that, including window functions for the run-length encoding and
trace-level aggregation, without loading everything into memory as a pandas frame first.
It's also why the cleaned dataset compresses down to roughly 300MB as parquet.

---

## H. Examiner-specific questions — Graham Healy & Liang (Charlie) Xu

Researched from public DCU staff pages, DBLP, and Google Scholar. **Caveat up front:**
neither examiner has a public research history in intrusion detection, cybersecurity, or
XAI specifically — this reads as a general-computing panel pairing, not domain
specialists. Treat everything below as "their background makes this angle more likely
than average," not "they will definitely ask this." Healy's match is closer than a
generic evaluation-methodology overlap, though — see below.

### Graham Healy — profile

Assistant Professor, DCU School of Computing; Programme Chair, BSc Computer Science;
previously Research Fellow at the Insight Centre for Data Analytics. PhD in
brain-computer interfaces; research area is machine learning and multimodal modelling
of human behaviour (EEG, eye-tracking, physiological/behavioural signals), plus
**evaluation frameworks and benchmark/dataset methodology** as an explicit research
strand. This makes the user-study half of your work (RQ1) a natural target — evaluation
rigour is closer to his own research than intrusion detection is.

More specifically: his recent (2025) publication record includes real, verified
transformer-architecture work, not just generic ML — co-author on **"Efficient
Transformer-Based Drowsiness Detection on the Edge using a Hybrid MobileViT-LSTM
Architecture"** (with Shams Ur Rahman and Noel E. O'Connor), which evaluates a
transformer-based model specifically for resource-constrained/edge deployment cost, and
**VEAGLE: Eye Gaze-Assisted Guidance for Video Browser Showdown** (MMM 2025), which is
eye-gaze/attention-signal work. Both are closer parallels to this project than a generic
"he does evaluation" framing suggests: one is literally about measuring a transformer's
deployment cost on constrained hardware (your RQ2), the other is about reading human
gaze/attention as a signal (adjacent to your attention-as-explanation channel, RQ1/C3).
Treat him as the examiner most likely to engage with your architecture and cost
measurements technically, not just your survey methodology.

### H1. Was your rating scale a validated instrument, or one you designed yourselves? What was the actual question wording for usefulness/ease/trust?

**This one is genuinely open — you need to answer it from what you actually built, not
from this document.** I don't have the underlying Google Form's exact wording or
whether "usefulness/ease/trust" map onto an established construct (e.g. something in
the trust-in-automation or explanation-satisfaction literature) or were defined
in-house for this study. Before the viva, check your own form and be ready to say
plainly which it was — an honest "we defined these three dimensions ourselves, informed
by [whatever you actually read]" is a fine answer; claiming a validated scale you can't
name if pressed is not.

### H2. Was the order in which participants saw Occlusion/Attention/LIME randomised or counterbalanced?

**Also open — confirm this yourself before the viva.** The final paper doesn't state
this either way, and I'm not going to assert "yes, it was counterbalanced" on your
behalf without you confirming it against the actual survey design — an earlier stale
draft (`docs/documentation/viva_prep.md`) claimed this, but that file has already turned
out to be unreliable on other numbers, so don't repeat its claims without checking. If
order wasn't controlled, the honest answer is that it wasn't, and the mitigation is that
all three were shown in the same visual format for the same alert, which controls for
presentation bias even if not order bias.

### H3. With n=17 and three repeated-measures conditions, do you consider this result conclusive or exploratory?

**Say:** Exploratory / pilot-scale, and the paper's own wording reflects that — "no
single method was rated significantly higher than the others." We report descriptive
statistics (means, majority-preference count) rather than leaning on a significance
test that would have very limited power to detect a real difference at this sample
size. A moderate, non-dramatic preference for Attention is the honest reading, not a
proven superiority claim.

### H4. General ML grounding he may probe given his own applied-ML background

Be ready for plain "explain the mechanism" questions independent of the security domain
— residual connections and layer norm (why both, what breaks without them), what the
class-weighting actually does to the loss surface, why PR-AUC rather than ROC-AUC on an
imbalanced problem. These are covered in sections A and G above; just expect him to ask
them more precisely/technically than a generalist examiner might.

### H5. You measure LIME at 11.6 seconds on CPU and call it a deployment blocker. Given work like efficient/edge transformer deployment, wouldn't the real fix be an architectural or sampling change, not just reporting the cost as a finding?

**Say:** That's a fair push, and the honest answer is we treated RQ2 as a measurement
study, not an optimisation study — the goal was to characterise the cost each method
actually has today, not to engineer LIME down to a lower number. There are real,
known ways to cut it: fewer perturbation samples in the LIME call (`num_features`/
sample count is a direct dial), or restructuring the model so attention scores come out
on the same forward pass used for prediction instead of the extra sub-model invocation
we currently do (see C3). We didn't do either here, because the point of RQ2 was to
report the honest cost of the methods as commonly implemented, not the cost of a
version we'd specifically engineered to look cheap. If deployment were the actual next
step, that engineering work is exactly where we'd go next.

### H6. Attention weights aren't the same as causal importance — there's a body of work arguing attention doesn't reliably explain a model's decision. How do you defend using it as an explanation channel at all?

**Say:** We don't claim attention is a ground-truth causal explanation — we present it
as one of three different lenses, precisely because none of the three is claimed to be
the single correct explanation. What we do have is a cross-check: on the worked example,
attention's highest-weighted positions (the `clone` calls) agree with what occlusion
independently measures as the largest exact log-odds driver, using a completely
different mechanism. That agreement is evidence the attention signal is tracking
something real on this case, not proof that attention is causally faithful in general.
We're reporting what practitioners find useful and trustworthy about each method (RQ1),
not asserting that any one of them is the objectively correct account of the model's
reasoning — that's a fair distinction to draw if pushed on the attention-as-explanation
critique specifically.

---

### Liang (Charlie) Xu — profile

Lecturer/Assistant Professor, DCU School of Computing. Research area is NLP and
computer-assisted language learning — digital game-based language learning, an
Irish-language learning game project ("Cipher"), VR/games for language education. His ML
exposure is NLP-flavoured, not security. The natural overlap with your project isn't the
security domain at all — it's that your detector is structurally a small NLP-style
sequence model (tokenise, embed, self-attend, pool) applied to syscalls instead of
words, and your RQ1 open feedback (users wanting "plain-language explanation") sits
close to his own interest in how people learn from system output.

### X1. Your syscall vocabulary and tokenisation looks like NLP preprocessing. How do you handle a syscall at test time that never appeared in training?

**Say:** The vocabulary is built only from syscalls seen in the training days —
concretely, 117 distinct syscalls survive after cleaning (out of 347 defined in the
32-bit Linux syscall table used by this dataset), plus a reserved pad token (0) and an
unknown token (1). Any syscall encountered at validation or test time that wasn't in
that training vocabulary maps to the unknown token rather than crashing or being
dropped. Confirmed directly in the code: `vocab.get(int(v), 1)`.

### X2. Why masked mean pooling rather than a [CLS]-style pooling token, which is more standard in BERT-family NLP transformers?

**Say:** Mean pooling over the real (non-padded) positions treats every position
equally and doesn't require adding an artificial token to the sequence or vocabulary —
it keeps the representation purely a summary of the behaviour observed. A learned
[CLS]-style pooling token is a reasonable alternative and is standard in NLP, but it
wasn't tried in this study, so I won't claim a comparison we didn't run. If pushed on
"would it help," the honest answer is: possibly, it's untested here.

### X3. Your positional embeddings are learned/absolute rather than relative or rotary, which is more common in recent NLP transformers. Why?

**Say:** The sequence length is fixed at 256 by construction (pad/downsample), so
there's no variable-length or very-long-context problem that relative or rotary
position schemes are usually solving. A simple learned absolute positional embedding is
sufficient at this fixed length and keeps the model simple; we didn't test relative or
rotary variants, so this is a design-simplicity choice, not a result of an ablation.

### X4. Your survey found users wanted more plain-language explanation. Did you consider generating natural-language text alongside the visual explanations?

**Say:** No — we didn't build or test a natural-language explanation layer in this
study. It's explicitly named as future work in the conclusion, prompted directly by
that open-feedback theme, but nothing in the current paper implements it.

### X5. Is this fundamentally an NLP model applied to a security domain — syscalls as "words," traces as "sentences"? What's actually different?

**Say:** Structurally, yes, it borrows the tokenise-embed-self-attend-pool playbook
from NLP. Two things are genuinely different: each position carries a second numeric
channel (the log-scaled repeat count) fused into the embedding, which isn't part of
standard word tokenisation; and the vocabulary is small and closed — 117 syscalls
versus tens of thousands of words — so there's no subword tokenisation or smoothing
needed, just a single explicit unknown-token fallback.
