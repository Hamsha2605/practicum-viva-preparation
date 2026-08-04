# Presentation script — 15 minutes, two speakers

Speakers: **Harrish** covers the deck through the explainability overview (slides
1–8). **Hamsha** takes it from the occlusion example onward (slides 9–17). One
handoff, not a back-and-forth. Times are per-slide targets, ~15 minutes total with a
little slack — practice it once end to end with a timer before the real thing, it
always runs long the first time.

---

## Harrish's part — Title through Explainability overview (slides 1–8, ~6 min 50s)

### Slide 1 — Title (~15s)

Good [morning/afternoon]. I'm Harrish, and this is Hamsha, we're presenting our
practicum, Explainable AI in Anomaly Detection, supervised by Dr. Irina Tal. We built
an intrusion detector, then spent most of the project on the part most papers skip:
explaining its decisions, and actually asking security people whether those
explanations are any good.

### Slide 2 — Motivation (~60s)

So why does this matter. CrowdStrike's 2025 threat report found that 79% of the
intrusions they caught in 2024 were malware free, hands on keyboard attacks, not
something a signature database would recognise. Rule based intrusion detection is
built entirely around signatures, so it's structurally behind that shift.

The usual fix is machine learning and deep learning, model normal behaviour, flag
deviations. That part works reasonably well. But it creates a second problem. These
models are black boxes. A security analyst gets an alert with no reasoning attached,
and in a security context, an alert you can't verify is an alert you can't act on with
confidence.

That's the gap this project sits in. We don't just build a detector. We build one,
then apply three different explanation methods to its alerts, and compare them
directly against each other with real practitioners.

### Slide 3 — Research questions (~40s)

That comparison is organised around two research questions, straight from our
proposal. RQ1: which Explainable AI method is best suited for anomaly detection, and
which one is most transparent and understandable for security analysts. RQ2: how does
adding explainability affect the model's performance, such as accuracy and response
time.

On RQ2, I'll flag this now so it doesn't sound like an oversight later: explainability
here is applied after the model has already made its decision, so accuracy never
actually changes. What we found is that the real effect is on response time, and
that's what RQ2 ends up being about.

### Slide 4 — Related work gap (~45s)

Quickly on related work. Deep learning for intrusion detection is a well covered area.
DeepLog uses an LSTM over log sequences, later work adds CNN and attention layers on
top of that, others use ensembles or federated learning, and one recent paper puts a
Transformer encoder directly on syscall traces, which is close to what we do.

Separately, there's a growing body of XAI for IDS work. But look closely and most of
it applies SHAP or LIME after the fact, to tree models or CNNs, on network flow data,
and almost none of it asks the actual intended user, a security analyst, whether the
explanation helps them. That's the gap. Nobody compares explanation methods head to
head, on host level syscall data, with a real practitioner survey and a cost
measurement attached. That's what we did.

### Slide 5 — Dataset (~60s)

The dataset is NGIDS-DS, from Haider's 2018 UNSW thesis. It's real system call traces
off a Linux host, with attacks tied to actual CVEs, simulated over six days, ninety
million events total, about one point four percent of them attack events, spanning
seven attack categories from exploits to reconnaissance.

We split it by day, not randomly. Train on the eleventh through the fourteenth of
March, validate on the fifteenth, test on the sixteenth, a day the model has never
seen. That's deliberate. It simulates the real deployment situation, train on the
past, detect on the future, and it avoids any leakage you'd get from a random split
cutting across the same attack session.

### Slide 6 — Cleaning and feature engineering (~60s)

Before that data reaches the model, two things happen. First, cleaning. We found
duplicate logged events, and we found that two syscalls, clock_gettime and
gettimeofday, make up fifty two percent of all events on their own, and occur at
exactly the same rate in attacks as in normal activity. They carry zero signal and a
lot of noise, so we drop them. That takes us from ninety million events down to about
forty two point six million.

Second, feature engineering. We run length encode the trace, so repeated calls
collapse into one entry with a count, fix every trace to two hundred fifty six
positions, and tokenise each call using a vocabulary built only from the training
days.

One deliberate choice here. We leave the program name out entirely. In this dataset,
the victim programs barely show up outside attack windows, Firefox has essentially no
normal events at all. If we kept the program name, the model would just learn "this is
Firefox, therefore attack," which is memorising the dataset's construction, not
learning attacker behaviour. Dropping it forces the model to actually read the syscall
behaviour.

### Slide 7 — Model architecture (~90s)

So how does the detector actually work. It's a Transformer encoder. At each of the two
hundred fifty six positions we combine three things into one sixty four dimensional
vector, an embedding of the call itself, a learned positional embedding, and the log
of how many times that call repeated. Those get summed, so every position carries what
it is, where it is, and how much it repeated.

That sequence goes through two identical encoder blocks. Each one is multi head self
attention with four heads, padding masked out so it can't attend to nothing, then a
feed forward layer, with residual connections and layer norm around both. After the
second block we take a masked mean over the real positions to get one vector per
trace, pass it through dropout and a single sigmoid unit, and that's our probability
of attack.

On training, Adam optimiser, batch size one twenty eight, up to twenty five epochs
with early stopping on validation PR-AUC. Because attacks are rare, the positive class
is up weighted in the loss. And one important detail, we don't use the default zero
point five cutoff. We pick the threshold on the validation day by maximising F2, which
weights recall above precision, because in intrusion detection missing an attack is
worse than a false alarm. That threshold is then frozen and applied unchanged to the
test day, so we're never tuning on the data we report results on.

That's the detector. Now, how do we explain it.

### Slide 8 — Explainability overview (~40s)

For explainability, we apply three methods to the same alerts. Occlusion removes part
of the input and measures how much the score moves, it's exact and doesn't care what
model it's explaining. Attention reads the weights the Transformer already computed as
part of its own forward pass. LIME fits a small linear model to how the detector
behaves on perturbed versions of the trace, locally around one prediction.

We deliberately show all three in the same visual format for the same alert, so when
we ask users to compare them, they're comparing the content of the explanation, not
which one has nicer formatting.

I'll hand it over to Hamsha to walk through what each of these actually looks like on
a real alert, and what we found.

---

## Hamsha's part — Occlusion through close (slides 9–17, ~8 min 50s)

### Slide 9 — Occlusion example (~55s)

Thanks Harrish.

Here's occlusion on a real case, a Firefox exploit trace from the test day. On the
left, we remove whole behaviour groups, network, file, process, one at a time and
watch the log odds drop. Network calls and process calls are the biggest drivers here,
removing the file calls actually pushes the score up, meaning file activity was
working against the alert, not for it.

On the right, same idea at the individual syscall level, poll and clone are the two
calls doing the most work, removing either one drops the score by nearly a full log
odds point. That's occlusion's strength, it's a direct, exact measurement, not an
approximation.

### Slide 10 — Attention example (~55s)

"Same trace, different lens. This time we're not removing anything — we're reading the weight the model's own attention layers put on each position, averaged across heads and layers.

The dashed line is our baseline: what uniform attention looks like if the model isn't focusing on anything in particular.

Three clear spikes rise above that line — all clone calls, process creation events — three to four times the uniform share.

And here's the key part: we didn't run any extra procedure to get this. It's already sitting inside the model's forward pass. That's exactly why attention is cheap — which matters in a minute when we get to cost."

The advantage here is we're not running any extra procedure to get this, it's already
sitting inside the model's forward pass. That's also exactly why it's cheap, which
matters for RQ2 in a minute.

### Slide 11 — LIME example (~45s)

LIME takes a different approach. We generate perturbed versions of the trace by
randomly dropping subsets of calls, score every one with the detector, and fit a
linear surrogate to those results. The weight on each position in that surrogate
becomes its contribution to the alert.

You can see it here on the same trace, clone again comes out as the strongest positive
contributor, consistent with what occlusion and attention both showed, while a couple
of write and read calls actually pull the score down slightly. LIME doesn't measure
the model exactly the way occlusion does, it approximates it locally, but it produces
a format a lot of practitioners already recognise from other tools.

### Slide 12 — Model evaluation results (~85s)

Stepping back from individual explanations to the detector itself. On the test day,
fifteen forty eight traces, one twenty eight of them genuine attacks, we catch ninety
six and miss thirty two, with twenty six false alarms out of fourteen twenty normal
traces. That gives seventy five percent recall, seventy eight point seven percent
precision, a one point eight percent false alarm rate, and ninety six point three
percent accuracy overall.

The precision recall curve on the right tells us this isn't a fragile result tied to
one lucky threshold. Precision stays close to one point zero while recall climbs past
roughly zero point five, and only trades off as we push recall further, right where
you'd expect given we optimised for recall with F2. The area under that curve, the
PR-AUC, is zero point seven nine, the model is ranking attack traces above normal ones
reliably across most of the range, not just right at the cutoff we picked.

### Slide 13 — RQ1 results (~85s)

Now the part that actually answers RQ1. We surveyed seventeen people, software
developers, IT professionals, security analysts, and students, spanning no
cybersecurity background up to expert, and had them rate all three explanation methods
on usefulness, ease, and trust.

Attention comes out on top on usefulness and ease, both at seven point two nine, with
occlusion lowest on those two. Trust is close across all three, between six point
eight eight and seven point zero zero, people trusted the model's decision about the
same amount regardless of which explanation they were shown, which itself is a
finding, the explanation changed how easy the alert was to read, not how much they
believed it.

When we asked users to just pick the single most effective method, fifty three
percent picked Attention, twenty nine percent Occlusion, eighteen percent LIME. That
lines up with the average scores. It's a majority, not a landslide, no method won
overwhelmingly.

The open feedback converged on three things. People want plain language text alongside
the visuals, a clearer baseline to judge the numbers against, and a similarity score
against previously seen attacks. All three feed directly into our future work.

### Slide 14 — RQ2 results (~70s)

RQ2, what does each of these cost. We measured latency with and without explanation,
on both GPU and CPU, averaged over ten runs.

On GPU, everything is fast. Baseline prediction is eighty seven milliseconds,
Attention and Occlusion add another one seventy to one ninety milliseconds each, LIME
adds four thirty two. All comfortably under a second.

CPU is where it splits. Attention and Occlusion barely move, still under three hundred
milliseconds total, because both only need a small, fixed number of extra forward
passes. LIME, though, needs many perturbed samples scored individually to fit its
surrogate, so it depends on parallel hardware to stay fast. Take that away and it
jumps to eleven point six seconds per alert. That's not a rounding difference, that's
the difference between usable at alert volume and not.

### Slide 15 — Discussion (~60s)

Pulling RQ1 and RQ2 together. The literature we built on flagged two standing gaps in
XAI research generally, a lack of quantified impact, meaning most work claims an
explanation is useful without measuring it against real users, and a lack of
standardisation in how explanations get presented or compared. This study speaks
directly to both. We didn't just assert Attention was better, we measured it against
seventeen practitioners, on a common format, and found a real but moderate preference,
not a landslide.

The cost side matters just as much in practice as the usefulness side. Attention and
Occlusion are close to free. LIME's near twelve second CPU cost is exactly the kind of
thing that decides whether an explanation method survives contact with a real, high
volume alerting pipeline, and that's the kind of number this field doesn't report
often enough.

What we're not claiming, this is one dataset, one architecture, and the qualitative
comparisons are built on individual flagged trace case studies, not an explanation for
every alert type. That scope is deliberate, not an oversight, and it's exactly where
the future work picks up.

### Slide 16 — Conclusion and future work (~60s)

To close. We built a Transformer based host intrusion detector that reads behaviour
alone, with no program identity, and reaches seventy five percent recall and seventy
eight point seven percent precision on a genuinely held out test day. We applied three
explanation methods to its alerts, compared them directly with seventeen real
practitioners, and found Attention preferred by a majority, with Occlusion and
Attention both cheap and LIME expensive without a GPU.

For future work, the survey itself pointed the way, a plain language layer to sit
alongside the visual explanations, a similarity score against known attacks so an
analyst has something to anchor a new alert against, and broadening the evaluation
past the single trace case studies we used here to a wider range of attacks.

### Slide 17 — Thank you (both, ~15s)

**Hamsha:** That's our talk.

**Harrish:** Thank you, happy to take questions.

---

## Timing summary

| # | Slide | Speaker | Target |
|---|---|---|---|
| 1 | Title | Harrish | 15s |
| 2 | Motivation | Harrish | 60s |
| 3 | Research questions | Harrish | 40s |
| 4 | Related work gap | Harrish | 45s |
| 5 | Dataset | Harrish | 60s |
| 6 | Cleaning & feature engineering | Harrish | 60s |
| 7 | Model architecture | Harrish | 90s |
| 8 | Explainability overview | Harrish | 40s |
| 9 | Occlusion example | Hamsha | 55s |
| 10 | Attention example | Hamsha | 55s |
| 11 | LIME example | Hamsha | 45s |
| 12 | Model evaluation results | Hamsha | 85s |
| 13 | RQ1 results | Hamsha | 85s |
| 14 | RQ2 results | Hamsha | 70s |
| 15 | Discussion | Hamsha | 60s |
| 16 | Conclusion & future work | Hamsha | 60s |
| 17 | Thank you | Both | 15s |

Harrish's block runs about 6 minutes 50 seconds, Hamsha's about 8 minutes 50 seconds.
Still Hamsha-heavy, mainly because the two biggest single chunks in the whole deck,
the RQ1 and RQ2 results slides at 85 seconds each, both sit on his side and there's no
way to move them without breaking up the results section.

## Delivery notes

- Don't read the slide text. The slides carry the numbers and structure, say the
  reasoning in your own words, which is what this script models.
- Each person owns their block start to finish. You don't need to re-explain who you
  are mid-block, just talk like you're walking a colleague through your own work.
- On the one handoff (end of slide 7 to start of slide 8), make eye contact as Harrish
  finishes his last line, it reads as rehearsed but natural, not stiff.
- If a figure gets a reaction (the attention spikes, the LIME table), pause half a
  second and point before explaining it. Don't talk over the moment someone is still
  reading.
- Practice out loud at least once with a timer running. This script reads faster than
  it will actually come out under nerves, budget for that, don't just trust the numbers
  above.
