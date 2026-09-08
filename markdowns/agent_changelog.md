# Agent change log — every change to the agent, in plain English

This is the running, non-technical record of **every change made to the QC agent**: its
system prompt, the tools it can call, the rules it must follow, the evidence it is given,
and how its answers are judged. It exists so the whole project can be talked through from
one file — what was broken, what was changed, and what the change actually bought.

**Read it newest-first.** Each entry is self-contained; you should not need to open any
code to understand one.

**Rules for adding to this file are in `CLAUDE.md` §14.** Every agent-affecting change
gets an entry, in the same format, in the same commit as the change.

**Vocabulary, once, so the entries can stay short:**

| Term | Plain meaning |
| --- | --- |
| the agent | the LLM that inspects a sensor record, runs quality-control checks, and decides what to do about each problem |
| system prompt | the standing instructions the agent is given at the start of every run; versioned `v0.x` |
| detector | a statistical check that points at suspicious rows. It raises *candidates*, it does not decide |
| verdict | the agent's claim about what a stretch of data **is** (a real anomaly, or genuine water) |
| action | what the agent does with the values (delete, correct, keep, fill in) |
| flag log | the per-row record of what was found, what was decided, and why |
| run | one end-to-end pass of the agent over one dataset, labelled by letter (run K, run P, …) |
| precision / recall | how often the agent's accusations are right / how many of the real problems it catches |
| FNU | the unit turbidity (water cloudiness) is measured in |

---

## 2026-09-08 — Measuring how a point was *approached*, not just how wide it is

**Prompt version:** — · **Area:** new measurement
**Runs involved:** the 2026-09-02 run on 01467200

**Before:** the width of a suspicious excursion was measured at **half its own height**,
which sits well up the cone of a peaked feature — so it described the summit rather than the
foot. Nothing measured whether the series *climbed steadily* into the point over the couple of
hours before it, either: the two existing shape measurements look back 45 and 90 minutes, and
the run's own notes record that nothing looked further back than that.

**Changed:** the excursion is now measured at two heights — near the top and near the base —
and the ratio between them describes whether it is a narrow spire or a broad mound. Alongside
it, a count of how many consecutive samples the series rose into the point and fell away from
it, looked for over two hours either side.

**Design decision that is load-bearing:** a *flat* step does not count as "still rising". A
synthetic fault is injected as a rectangular displacement of 1–3 samples, so counting flat as
continuing would walk the measurement straight across an artifact's own flat top and hand it
the exact shape this exists to find on real water.

**Evidence:** treating flat as continuing **inverted the result** — it covered 55.4% of the
run's wrong deletions against 90.0% of its correct ones, i.e. it pointed at the true faults.
Made strict, a run of 3 or more covers **30.8% of wrong deletions against 10.0% of correct
ones** — a 3× separation, and it catches 15 of the 58 wrongly-deleted points that the width
measurement had called one sample wide.

**Honest caveat recorded:** it reaches under a third of the false accusations. One piece of
evidence, never a decider — and the hours-scale version of the same idea is weaker still
(23.1% at 4.8%).

**Interview angle:** the same measurement taken at a second height, and over a longer window,
found points that the original was structurally blind to. Also: the sign of a result flipping
when a definition changes by one word is worth measuring rather than arguing about.

---

## 2026-09-02 — Spike sensitivity is an absolute standard, not a fixed quota

**Prompt version:** v0.23 · **Area:** detector settings, tool limits
**Runs involved:** measured across all 18 datasets

**Before:** the spike detector was tuned to always hand the agent about the same number of
suspects (~175) on every record, no matter how many real problems were present.

**Changed:** the detector now applies a fixed *standard* of unusualness instead of a fixed
*headcount*, so a clean record produces few suspects and a badly contaminated one produces
many. The agent was also allowed to examine up to 1,000 points in detail per call, up from
300, so it can actually look at everything the detector raised.

**Why it matters:** the quota was quietly capping the score in both directions — on a clean
record it forced 168 suspects when only 21 problems existed (guaranteeing false accusations),
and on a dirty one it capped the catch rate at 45% before the agent had judged anything.

**Evidence:** correlation between "number of suspects raised" and "number of real problems
present" went from **−0.29 (backwards) to +0.72**. Catch rate on the dirtiest dataset went
**0.454 → 0.970**; suspects on the cleanest went 174 → 88.

**Interview angle:** a metric that looks stable across datasets can be stable *because* it
has stopped measuring anything. Normalising is right when the units differ between records
and wrong when the quantity itself is what you want to know.

---

## 2026-08-31 — Confidence is derived from the evidence, not self-reported

**Prompt version:** v0.22 · **Area:** decision quality, flag log
**Runs involved:** run P

**Before:** the agent labelled each of its own decisions "clear-cut" or "a judgement call",
and nothing checked that label. Since the review queue is built from exactly that label, the
agent could mark a hard call easy and it would never be reviewed.

**Changed:** the system now cross-checks the measurements behind each point. If the evidence
disagrees with itself — the shape says "sensor error" but the surroundings say "the water was
genuinely busy", or the rainfall record says "it was storming" — the point is *forced* to be
treated as a judgement call, must get its own individual decision, and must carry written
reasoning. The export refuses the file otherwise.

**Why it matters:** it converts an honesty requirement into an enforced one. It also stops a
single sweeping "these are all fine" decision from quietly absorbing points the agent had
personally measured and found alarming.

**Evidence:** points whose evidence conflicted were decided **wrong 69% of the time versus
17%** when the evidence agreed — a 4.2× signal, and the sharpest predictor of error in the run.
Honest limit recorded at the time: conflicted points hold about **half** the run's errors, so
this fixes the record-keeping and roughly half the problem, not the accuracy overall.

**Interview angle:** self-reported confidence is unfalsifiable and therefore useless as a
routing signal. Deriving it from whether the instruments agree makes it auditable.

---

## 2026-08-31 — Rainfall is a weighting, not a trump card

**Prompt version:** v0.21 · **Area:** system prompt, use of outside evidence
**Runs involved:** run P

**Before:** the prompt said that if it had rained near the sensor, the agent should default
to keeping the suspicious reading (on the theory that rain explains a genuine spike in
cloudiness).

**Changed:** rain is now one piece of evidence to weigh, explicitly *not* a default.
A spike-shaped point that also has rain behind it is defined as a judgement call, because the
two instruments genuinely disagree and neither settles it.

**Why it matters:** the previous wording was too strong and was costing real detections.

**Evidence:** measured over **1,471** spike candidates, rain moves the odds that a point is
real water from **44% to 59%** — real, but weak. Run P kept 22 points on the strength of rain
and **45.5% of them were injected faults**, almost exactly what those odds predict.

**Interview angle:** quantify how much a piece of evidence is worth *before* writing an
instruction that acts on it. "Rain implies real water" felt obviously true and was only worth
15 percentage points.

---

## 2026-08-25 — Spike thresholds are measured from the record, not guessed

**Prompt version:** v0.20 · **Area:** detector settings, tool output

**Before:** the agent picked spike-detector sensitivity from remembered rules of thumb.

**Changed:** the initial inspection now measures the record and hands the agent the settings
that fit it, for **both** spike detectors, and the agent is told to run both. Also narrowed
the detector's notion of "neighbourhood" from 20 samples to 10.

**Why it matters:** the two detectors disagree usefully — each catches spikes the other
misses — and the neighbourhood size had silently become wrong when the data moved from
15-minute to 5-minute sampling (20 samples used to mean 5 hours, then meant 100 minutes).

**Evidence:** no summary statistic of a record predicts the right threshold (the best
predictor is how contaminated the record is, which the agent cannot know). The narrower
neighbourhood won on **8 of 9 datasets** and tied on the ninth.

**Interview angle:** any constant expressed in "samples" is a latent bug the moment the
sampling rate changes. Four separate constants in this project had that bug.

---

## 2026-08-25 — One rule for filling holes: straight-line, up to an hour, everything

**Prompt version:** v0.19 · **Area:** repair behaviour
**Runs involved:** run N

**Before:** gaps were filled with a rolling median, and rows the agent *deleted* were left as
permanent holes because deletion happened at the very end, after the filling tool had run.

**Changed:** one rule now covers every hole — natural gaps, injected gaps, and holes created
by the agent's own deletions: fill anything an hour or shorter by drawing a straight line
across it, leave anything longer as an honest hole. The line is anchored on the *median* of
the readings either side, not on the two individual readings.

**Why it matters:** deleting a 10-minute spike and leaving a permanent hole is not a repair —
run N left **218** such holes. The old method also half-filled gaps, repairing the ends and
leaving the middle (23 of 27, 23 of 35, 23 of 55 rows on three of four test gaps).

**Evidence:** straight-line filling beat the rolling median on both accuracy and coverage
(100% coverage at 1.659 error vs 62% coverage at 1.765). The median anchor costs ~4% accuracy
on clean data and is **32× more robust** when a single bad reading sits at the edge of a gap —
which is exactly where bad readings cluster.

**Interview angle:** the simpler method won because it estimates the right thing (the trend
across the gap) rather than the wrong thing (the level around it).

---

## 2026-08-25 — A level shift is corrected, not deleted

**Prompt version:** v0.18 · **Area:** repair behaviour, new tool
**Runs involved:** run L

**Before:** when the sensor stepped to a wrong baseline for a stretch, the agent deleted the
whole stretch.

**Changed:** a new tool measures the size of the step at both ends of the stretch and shifts
the values back by it. The agent is told to correct rather than delete.

**Why it matters:** a level shift is an *offset* — the sensor reported the wrong number, but
the water underneath still moved normally, so the shape inside the window is real data at the
wrong height. Deleting it throws away recoverable record.

**Evidence:** on the test dataset the corrected stretch went from **6.95 to 0.78 FNU** away
from the truth, 117 rows recovered, nothing outside the window touched. Run L had deleted
those same 117 rows, which was most of why its wrongly-deleted count doubled.

**Also recorded:** the off-the-shelf library function for this was tested and rejected — it
rewrote 2,324 rows of genuine storm data and never touched the actual fault.

**Interview angle:** matching the repair to the *physical* failure mode, rather than treating
every anomaly as something to remove.

---

## 2026-08-25 — A level shift is settled by both of its edges

**Prompt version:** v0.17 · **Area:** system prompt
**Runs involved:** run J

**Before:** the agent judged a suspected level shift from one edge and from how much the data
wobbled inside the window.

**Changed:** the rule is now "two sharp edges around a held level is a rectangle, and rivers
do not make rectangles". Wobble inside the window is explicitly declared *not* evidence. A
contradictory leftover line ("default action: keep unless clearly erroneous") sitting directly
above the new rule was removed.

**Why it matters:** closing one escape route without closing the other just moves the failure.
Run J stopped using the previously-removed loophole and took the other exit instead —
declaring the artifact to be genuine water, on unsound grounds.

**Interview angle:** prompts fail like software does. Fixing one branch sends the behaviour
down the next one, and contradictory instructions left adjacent to each other get obeyed
selectively.

---

## 2026-08-25 — The rainfall check is enforced in code (and had never once run)

**Prompt version:** v0.16 · **Area:** enforcement, bug
**Runs involved:** every run since v0.13

**Before:** the prompt had *required* the agent to check every spike against nearby rainfall
since v0.13, twelve days earlier.

**Changed:** two things. First, the bug: the code that routes tool calls had no branch for the
rainfall module, so every single call came back "unknown tool", the agent worked around the
error, and the run finished normally. Nothing looked wrong from any angle. Second, the
requirement is now enforced by the export step, which refuses to delete a spike that was never
checked against rainfall.

**Why it matters:** this is the sharpest lesson in the project — an instruction the agent
cannot physically comply with produces runs that look fine. A test now asserts that every tool
the agent is offered is actually *reachable*, not merely defined.

**Design note kept deliberately:** the check records that the agent **looked**, not that it got
an answer. Requiring an answer would make any point outside the rain gauge's coverage
permanently undeletable, leaving the run no way out but to ship values it believes are wrong.

**Interview angle:** the gap between "the prompt says so" and "the system does so" is where
this project's most expensive bugs lived. Enforce in code what matters; test reachability, not
just existence.

---

## 2026-08-25 — Stop deliberating over which points to hand a batch tool

**Prompt version:** v0.15 · **Area:** cost, system prompt

**Before:** the agent agonised over which subset of timestamps to pass to tools that accept
many at once.

**Changed:** told to pass them all. Reasoning effort was also lowered for that class of call.

**Evidence:** one such deliberation cost **18,290 characters of reasoning to produce a 74-character
result**.

**Related failure it prevents:** on another run the agent hand-picked 76 of 145 candidate
timestamps, its subset happened to exclude the real fault, and the run concluded the record had
no level shifts — with nothing about the result looking wrong. The underlying tool was also
changed to always read the full list itself, so a caller's selection can only *add* to it.

**Interview angle:** letting the model choose a subset introduced a silent, luck-dependent
failure. Removing the choice removed the class of error rather than warning about it.

---

## 2026-08-24 — A jump is an edge; a level shift is a window

**Prompt version:** v0.14 · **Area:** system prompt, scoring
**Runs involved:** run E

**Before:** the jump detector marks the two *edges* of a step, but a level shift is the whole
stretch in between. The agent's decisions therefore only ever covered a couple of rows, and
every run scored near-zero on level shifts no matter what the detector did.

**Changed:** three parts. A tool now pairs jump edges into candidate windows; the agent is
required to write one decision covering the whole window; and a decision that says "this is an
anomaly" now claims **every row it covers**, flagged or not.

**Also changed — scoring:** level shifts are now scored by whether the *event* was found, not
by what fraction of its rows were flagged. Judging an edge detector by row coverage is close to
a category error.

**Evidence:** the same flag log scores **0.009 by row and 1.000 by event**; the headline
score moved 0.651 → 0.772 on an identical run. **Recorded explicitly as a measurement
correction, not an improvement** — the agent did not get better, the ruler was wrong.

**Interview angle:** knowing which of your improvements are real and which are measurement
artefacts, and writing that distinction down where it cannot be quietly forgotten.

---

## 2026-08-20 — Rainfall as outside evidence, and a silently wrong time window

**Prompt version:** v0.13 · **Area:** new evidence source, bug

**Before:** every measurement the agent had was derived from the sensor's own trace. Nothing
could confirm from outside that the water had genuinely moved.

**Changed:** added nearby-rainfall lookup as a tool, and required the agent to check every
spike verdict against it — including the ones that look obvious, since those are the ones
outside evidence can overturn. Added a measurement of the *multi-hour* shape leading into a
point (nothing else looked back further than 90 minutes).

**Bug fixed at the same time:** a tuned 45-minute measurement window was written as "3 samples"
back when data was 15-minutely. When the project moved to 5-minute data it silently became 15
minutes — and this particular measurement *reverses its meaning* at the wrong width. A gradual
2½-hour rise read as "sensor artifact" at 15 minutes and "genuine storm" at 45.

**Honest caveat recorded:** the nearest rain gauge to one site is 31 km away, and summer storms
are often smaller than that. So rain *present* is decent evidence; rain *absent* proves nothing,
and "no data" must never be allowed to read as "no rain".

**Interview angle:** bringing in an independent data source to break ties that the primary
signal structurally cannot.

---

## 2026-08-20 — The jump threshold's unit was wrong, not just its number

**Prompt version:** v0.12 · **Area:** detector settings, tool output

**Before:** the agent chose the jump threshold from a remembered range, reasoning from the
record's average and standard deviation.

**Changed:** the initial inspection now measures the actual quantity the jump detector
thresholds, on this record, and hands the agent the number. The prompt also tells the agent to
reason from robust statistics (median, robust spread) rather than average and standard
deviation.

**Why it matters:** on a storm-driven river the standard deviation is dominated by a handful of
huge peaks. One gauge reports a standard deviation of 28 against a typical reading of 1.9 —
so "half a standard deviation" sits at twenty times the water's ordinary movement.

**Evidence:** the workable threshold is **6.2 / 13.8 / 75.7** on the three rivers — a 12×
spread that no multiple of any single summary statistic can span. A run at 3.0 flagged 2,865
rows, correctly identified them as storm limbs, kept all of them, and therefore reported zero
level shifts on a dataset containing three. The new approach produces a stable **131–162**
candidates on every gauge.

**Interview angle:** the agent's stated reasoning was internally sound and led to a
catastrophically wrong number, because it was reasoning about the wrong statistic. That is
diagnosable only by reading the run's own explanation.

---

## 2026-08-20 — Measure every flagged point; require a confidence label

**Prompt version:** v0.11 · **Area:** tool limits, decision quality
**Headline result:** spike precision **0.227 → 0.460** with no loss of catch rate

**Before:** the tool that inspects points in detail was capped at 20 points per call. A run
therefore measured **45 of 10,463 decided rows (0.43%)** and deleted 367 spikes having
personally looked at 33 of them.

**Changed, in two separately-measured parts:**

1. **The tool limit.** Cap raised 20 → 100 per call. The run then measured **all 263 flagged
   points in two calls**. Removed 90 false accusations on its own.
2. **The decision layer.** The "busy stretch" guard was retuned for the new data cadence; the
   agent is now given an upfront map of which stretches of the record are unusually noisy; and
   the confidence label on each decision was made **mandatory with no default**.

**Evidence:** false accusations 194 → 104 → **67**. Catch rate unchanged at 0.905 throughout,
so both fixes are pure precision. The confidence label had defaulted to "clear-cut", so **145
of 147** decisions read as easy — and those deletions were wrong **37.8%** of the time. Made
mandatory, 12.1% of decisions were marked hard, and those were wrong 66.7% vs 27.8% — a 2.4×
separation, which is what makes a review queue worth having.

**Diagnostic worth remembering:** after the fix, **100% of the remaining false accusations had
been measured**. The "deleted a point it never looked at" failure was gone; what was left was
genuine misjudgement. That is how the two causes were told apart.

**Interview angle:** the biggest single accuracy win in the project came from raising a tool's
output limit. A default that fabricates a confidence claim nobody made is worse than no field
at all.

---

## 2026-08-13 — Judging a point against its own neighbourhood, not the whole record

**Prompt version:** v0.10 · **Area:** new measurement

**Before:** every measurement scored a point against the *record-wide* scale. So a point read
as extreme whether its neighbours were flat or thrashing, and in a busy stretch the detectors
fired on dozens of points that each looked extreme. Runs were deleting them as dozens of
separate sensor failures. They are one stretch, and the right output is one decision about the
stretch.

**Changed:** a new measurement characterises the *surroundings* — how busy this stretch is
compared to a typical stretch of the same record, and how much the point stands out **within**
it. Both numbers ride along in every point description, because this failure is only visible
across a cluster; no single row shows it.

**Evidence:** the measurement the agent already had reads **18.7 on a real spike and 16.7 on a
false alarm** — it cannot tell them apart at all. Rescaling the *same* movement against the
local neighbourhood gives **17.1 vs 1.6**. The information was always there; the denominator
was wrong. A rule built on it spares 60% of false alarms at a cost of 4% of real spikes.

**Also distinguished:** a stretch can be busy because the sensor is thrashing or because the
water is genuinely moving fast. Those need different decisions, and how often the signal
reverses direction separates them.

**Interview angle:** the fix was not a new signal but a corrected reference frame for a signal
already in hand.

---

## 2026-08-13 — Splitting "what is it?" from "what do I do about it?"

**Prompt version:** v0.9 · **Area:** decision contract, scoring

**Before:** one field carried both. Whether the agent *claimed* a stretch was faulty was
inferred from whether it had done something destructive to the values.

**Changed:** every decision now states two separate things — a **verdict** (is this an anomaly,
and of what kind) and an **action** (delete, correct, keep, fill). Detection is scored on the
verdict alone. Consistency is enforced one way only: calling something genuine water and then
deleting it is refused; calling something faulty and keeping it is allowed for gaps, with a
written reason.

**Why it matters:** conflating them cost real catch rate — a gap correctly identified but too
long to fill defensibly was recorded as "keep", and scored as the agent claiming the data was
fine.

**Interview angle:** the metric was measuring the treatment and calling it a diagnosis. Two
questions, two fields.

---

## 2026-08-13 — Three reliability fixes worth more than any accuracy work

**Prompt version:** — · **Area:** robustness

A run is 15+ sequential API calls, so one failure anywhere throws away everything before it.
Three guards were added over several weeks, each for a *different* failure that had actually
happened:

- A request that fails **before** the response starts — retried up to 8 times. (A server
  overload at step 15 of 15, one call before the export, discarded a complete run.)
- A failure **during** the response stream — the SDK cannot retry these, so the code re-asks
  the whole question itself. (A read timeout while the long final response was streaming
  ended a process with 14 completed steps unexported.)
- A server error delivered **inside** an open stream, which is a different exception type than
  the previous case and slipped past the handler written for exactly this situation.
  (Run K: eight completed steps and ~$0.90 discarded.)

**Also fixed here:** the imputation tool was silently overwriting **1,887 real readings**
because the underlying library treats already-flagged rows as missing. It corrupted three
separate measurements before anyone noticed, because an overwritten row is recorded as a
successful gap fill.

**Interview angle:** long agent runs make ordinary transient errors expensive, and the failure
modes are distinguishable only by exactly where in the request lifecycle they occur.

---

## 2026-08-11 — Decisions must show their working, and sweeps are labelled as sweeps

**Prompt version:** v0.8 · **Area:** flag log, explainability

**Before:** a carefully-reasoned decision about one point and a blanket "everything else is
fine" decision covering hundreds of rows produced **identically-shaped** entries in the log.
Every run so far had had a blanket absorb points the agent had actually measured, and the log
could not show it.

**Changed:** every row in the log now records where its reasoning came from — a deliberated
judgement, a specific clear-cut call, a blanket sweep, or a code-level decision the agent never
made. A decision marked as a hard call **must** carry written deliberation; the export refuses
it otherwise. Each row also names *which* decision claimed it and how wide that decision
reached.

**Why it matters:** this is the project's core promise — that the tool explains itself —
made enforceable rather than aspirational. An auditor needs to tell a judgement from an
absorption.

**Interview angle:** the honest failure was not the sweeping decision itself, it was that
nothing in the output could distinguish it from a considered one.

---

## 2026-08-11 — The rise-versus-fall shape works, at exactly one window width

**Prompt version:** v0.7 · **Area:** measurement, system prompt

**This entry corrects an earlier conclusion, deliberately.** The project had previously
measured this shape signal, found it useless, and written down "do not rediscover this".

**What was wrong with that:** the original test compared injected faults against big *storm
peaks* over a 2-hour window. Both choices hid the effect. Re-measured against the population
that actually causes false accusations — small flush events — with the window swept:

| window | real flush | sensor artifact | separation |
| --- | ---: | ---: | ---: |
| 30 min | 0.58 | 0.99 | 0.56 |
| **45 min** | **0.75** | **0.99** | **0.84** |
| 90 min | 1.11 | 0.98 | 0.42 |
| 180 min | 1.43 | 0.93 | 0.27 |

(0.5 is a coin flip.) A real flush rises in one sample and decays over three to five, so at 45
minutes its fall is visibly gentler than its rise. Past 90 minutes the decay is over, the
window fills with flat surroundings, and **the ordering flips** — which is exactly what makes
this look like noise if you only sample a couple of wide windows.

**Changed:** window set to 45 minutes, exposed to the agent as its own tool, and the prompt now
treats a decaying fall as evidence for keeping the value.

**Interview angle:** a negative result that was a measurement-design failure, caught and
reversed. The tell was that the earlier test had used the wrong comparison group.

---

## 2026-08-10 — The most specific decision wins

**Prompt version:** v0.6 · **Area:** decision contract

**Changed:** when two of the agent's decisions overlap, the narrower one wins, so a broad
catch-all can no longer override a specific verdict the agent reasoned about individually.

**Interview angle:** small structural rule, removes a whole class of self-inflicted error.

---

## 2026-08-10 — Phases replace the script; the agent reads its results instead of executing a plan

**Prompt version:** v0.5 · **Area:** system prompt, cost

**Before:** the prompt laid out steps 1–8 with a fixed detector order and demanded a full plan
up front. Since the agent cannot see the data before it starts, that plan is a guess dressed up
as a decision — and it produced runs that executed their plan instead of reading their results.

**Changed:** the prompt now describes **phases**, not a script. Only three orderings are
genuinely forced, and each for a technical reason: inspect first; screen impossible values
before the spike detectors (so they don't distort the neighbourhood statistics those detectors
rely on); find gaps before filling them. Everything else is a menu the agent sequences from
evidence, and it is told explicitly that running a detector it has no reason to expect anything
from is a wasted call.

**Cost work in the same change:** the run had been costing $4.91, of which **88% was input
tokens re-sent at full price on every one of 15 steps**. Turned on prompt caching (verified);
also found that **34% of the payload was raw lists of timestamps**, which are now sampled
evenly across the record rather than sent whole. Recorded finding: turning off the agent's
reasoning would save little (12% of the bill) and would cost the entire audit trail.

**Interview angle:** the fix for "the agent ignores its results" was structural — remove the
plan it was following. And the cost lever was payload shape, not model choice.

---

## 2026-08-01 — Turn on the agent's reasoning trace, and fix a silently wrong model

**Prompt version:** — · **Area:** agent loop, observability

**Changed:** enabled adaptive extended thinking, which also lets the agent reason *between*
tool calls — that per-step trace is what makes a run reviewable afterwards. Raised the response
ceiling substantially (the final export step is the longest of the run; at too low a ceiling
the agent spent its whole budget deliberating and never emitted the export, ending a 15-step
run with nothing saved).

**Bug found:** the model identifier had drifted to a version predating the reasoning feature
entirely. It now lives in exactly one place, tied to a project-level rule.

**Interview angle:** an agent that cannot show its reasoning cannot be audited, and the
observability tooling is worthless without it.

---

## 2026-08-01 — One crash-prone detector can no longer sink a run

**Prompt version:** — · **Area:** robustness, tool output

**Changed:** wrapped the one detector known to crash on certain inputs so it returns an empty
result instead of an exception, and made every detector report both what it added and the
running total.

**Why the totals matter:** flagging is additive — a stricter second run of a detector takes
nothing back and reports only what it newly added. So an agent that re-tunes a detector and
reads the new count as "what this setting would find" is reading it wrong. It must start strict
and loosen.

**Interview angle:** the tool's return value had to be redesigned to prevent a specific
misreading, not just to report more.

---

## 2026-07-31 — The agent itself: the reasoning loop

**Prompt version:** v0.2–v0.4 · **Area:** foundation

**Built:** the ReAct loop — the agent inspects, reasons, calls one tool, reads the result,
decides the next call, and finishes by exporting its decisions and writing a plain-language
report. Hard cap of 25 tool calls enforced in code. Every API call logged to disk with inputs,
outputs and token counts, so any run can be replayed and any claim in this file can be traced
back to the run that produced it.

**Interview angle:** the logging is what makes every other entry in this document possible.

---

## 2026-07-30 — Giving the agent a way to tell a storm from a sensor fault

**Prompt version:** v0.2 · **Area:** new measurements

**Before:** detectors say **where** a rule fired. They cannot say **whether it is real water**,
which is the decision that actually matters.

**Built:** a set of measurements that describe the shape around any single timestamp — how wide
the excursion is, how long it took to recover, how sharp the step is — plus a summary tool that
returns all of them at once for a batch of points.

**Evidence that this is the right axis:** measured across three rivers, a sensor spike is 2–3
samples wide and recovers in 2 samples; a genuine storm peak is 7–25 samples wide and takes
12–30 samples to recover.

**Interview angle:** the project's central idea in one change — quality control is not anomaly
detection. Detection raises candidates; deciding what they are is a separate problem needing
different instruments.

---

## Backfill note

Entries dated before 2026-09-08 were reconstructed from the version history kept in
`src/agent.py`, the dated decision records in `CLAUDE.md`, and the git log. They are accurate
as to substance and measurement, and the dates are the date of the change rather than of this
write-up. Everything from 2026-09-08 onward is written at the time of the change.
