# main.tex edits required by the repaired theory section

**Version.** These edits are against the restructured `paper/main.tex`, md5 `152d8f961752c43b1939e615f3ceabdf` (saved 2026-10-20 00:52). Line numbers refer to that version. main.tex was not edited.

**Quoting.** Each "Current" block quotes the LaTeX source exactly, with line breaks joined by single spaces. Every block was machine-checked against the file.

**What the repaired `theory.tex` now says (summary):**
- *Context.* X is everything the agent is shown, including the reactions displayed in its history (in v2fb, the raw reactions of its last 6 turns).
- *No-feedback posterior.* `\eta_\varnothing = P(L|X,C)`. In the eps-family `\eta_\varepsilon`, eps = 0 means *perfect* feedback, so "eta_0" must not be used for a feedback-free quantity anywhere in the paper.
- *Theorem 1 (within-context form).* Given (X,C), a context-and-sample score is independent of the user, so among sessions sharing (X,C) its AUROC is exactly 1/2. Across contexts it is bounded by `\eta_\varnothing`. It does **not** put such detectors at chance, and `\eta_\varnothing` is not feedback-free.
- *Corollary 1.* Scoped to resample-dispersion scores, with a sharp form: if s* is least suspicious, `|AUROC - 1/2 - (e1-e0)/2| <= e0 e1/2`.
- *Theorem 2.* Titled "Less noisy feedback cannot hurt" (weak dominance). It holds only when the law of (u,X,C) does not depend on eps, i.e. for reactions regenerated for the detector. It does not show that feedback "breaks the symmetry".
- *Proposition 1.* Sequential (A1)-(A2) assumptions, with a proof. It is "Bayes-optimal under (A1)-(A2) with the true priors and channel; the implementation plugs in estimates". It covers the restricted label L_A (A fixed by X) and any memoryless channel P(F|w), which includes V6.
- *Proposition 2.* Titled "Priors are identified where feedback exists": n counts reactions only; attributes without reactions are not identified; with eps unknown, an interval of eps fits.
- *Deleted.* The scoring-rule paragraph.

---

## 1. References and labels pointing to the deleted scoring-rule paragraph

- **No `\ref` or `\label` points to it.** The paragraph never had a label. It cited nothing: `gneiting2007strictly` was never cited and stays in refs.bib.
- **main.tex line 15, `\newtheorem{lemma}{Lemma}`.** No lemma environment remains in theory.tex. Delete this line unless another section uses a lemma; currently none does.
- **Double-blind risk (W14), not main.tex.** Before any repository link, scrub the repo files `pge/scoring.py` and `tests/test_asymmetric_scoring.py`, which cite an "ICML'26 reviewer".

## 2. Sentences whose claim must change

### Abstract

**AB1** (lines 40-43)
```current
We show that sampling-consistency detectors cannot see these errors: conditional on the conversation, the model's resamples carry no information about which user it is wrong about, and an agent that fabricates consistently drives their AUROC to $1/2$.
```
Replace with:
> We show that sampling-consistency detectors cannot tell users apart: conditional on the conversation, the model's resamples carry no information about which user it is wrong about, and the more consistently an agent fabricates, the closer their AUROC is to $1/2$.

Why:
- Theorem 1 is a within-context statement. Across contexts, sample-based scores can carry base-rate signal: dev 8B per-commitment disagreement AUROC is 0.536, and its Cor. 1 cap is 0.884.
- Corollary 1 bounds |AUROC - 1/2| by (1 - (1-e0)(1-e1))/2. That bound shrinks as resampling becomes deterministic, and AUROC is exactly 1/2 only at a point mass.

**AB2** (lines 48-50)
```current
In a simulator where the truth is known, exact Bayes predictors split what is detectable into population risk, available from the commitments alone, and individual evidence, available only from the user's reactions.
```
Replace with:
> In a simulator where the truth is known, exact Bayes predictors split what is detectable into population risk, the chance that these commitments are wrong for a user with the same intake, and individual evidence from the user's reactions.

Why: the agent is shown the reactions, so its later commitments (and resamples) can carry part of the individual evidence. "Available only from the user's reactions" is therefore not true of the protocol.

**AB3** (lines 52-56)
```current
\chk{sampling detectors score \val{C/L_U/pooled/H1: AUROC(SelfCheckGPT)} AUROC on user-specific hallucination, detectors on PGE observables reach up to \pending{max}, and reactions add \val{C/RA/pooled/H5: etaF (Bayes, feedback) - eta0 (Bayes, no feedback)} AUROC beyond the best possible context-only predictor}.
```
Replace the final clause:
> ..., and reactions add \val{C/RA/pooled/H5: etaF (Bayes, feedback) - eta0 (Bayes, no feedback)} AUROC beyond the population risk of the same commitments}.

Why: the e11 quantity "eta0" is not the best possible context-only predictor when the agent is shown reactions. See ED1.

### Introduction

**IN1** (lines 82-85)
```current
More fundamentally, two users who give identical answers to the intake are indistinguishable to the model, so nothing computed from the model and its context can tell which of them it is wrong about (Theorem~\ref{thm:bound}).
```
Replace with:
> More fundamentally, two users who share a conversation (the same intake answers and, where the agent is shown them, the same reactions) are indistinguishable to the model, so nothing computed from the model and its context can tell which of them it is wrong about (Theorem~\ref{thm:bound}).

Why: PGE shows the agent the reactions. Two users with the same intake but different reactions are *not* indistinguishable to the model. Theorem 1 conditions on the whole context X.

**IN2** (lines 88-89)
```current
Sampling-consistency scores do not even do that (Corollary~\ref{cor:degenerate}).
```
Replace with:
> Sampling-consistency scores of a self-consistent agent do not even do that (Corollary~\ref{cor:degenerate}).

Why: Corollary 1 pins dispersion near 1/2 only when e0 and e1 are small. For the less self-consistent 8B agent the cap is loose (dev 0.884).

**IN3** (lines 94-97)
```current
Reactions are noisy and partial, but they are correlated with the truth for \emph{this} user, and that breaks the symmetry that defeats resampling (Theorem~\ref{thm:feedback}).
```
Replace with:
> Reactions are noisy and partial, but they depend on the truth for \emph{this} user, which no resample does: under a simple model the posterior of the hallucination event uses them exactly (Proposition~\ref{prop:posterior}), and less noisy reactions cannot hurt (Theorem~\ref{thm:feedback}).

Why: Theorem 2 gives weak dominance, only with the agent's trajectories fixed. It does not show that feedback breaks the symmetry. What does: reactions depend on w_t (Theorem 1 contrast), and under (A1) they are the only user-specific term in Proposition 1.

### Contributions

**CO2** (lines 106-109)
```current
\item \textbf{A decomposition of detectability.} Exact Bayes predictors in a simulator with known truth separate the population risk visible in the commitments from the individual evidence in the reactions, and bound what any detector of either kind can achieve.
```
Replace with (option A, no new analysis):
> \item \textbf{A decomposition of detectability.} Exact Bayes predictors in a simulator with known truth separate the population risk of the commitments from the individual evidence in the reactions; the posterior given commitments and reactions bounds what any detector of both, with or without resamples, can achieve.

Option B, if `\eta_\varnothing` is computed (see section 3, item 2): append "and the posterior given the conversation bounds every sampling detector (Theorem~\ref{thm:bound})".

Why: eta_F bounds every detector of (intake, C, F, samples), by NP and Theorem 1. The population-risk predictor does *not* bound context-only detectors when the agent's commitments depend on the reactions, so "of either kind" is false.

**CO3** (lines 110-113)
```current
\item \textbf{A protocol and a posterior.} PGE's commitments and reactions make user-specific hallucination detectable; its label-event posterior gives calibrated per-commitment risk, which we use to choose what to ask the user, measured with the agent re-reasoning after each answer.
```
Replace "gives calibrated per-commitment risk" with:
> gives per-commitment risk that is exact under a conditional-independence model (estimated by plug-in)

Why: Proposition 1's exactness and optimality hold only under (A1)-(A2) with the true priors and channel. On the dev data (A1) is mildly violated (proof_check §4).

### Problem, protocol, and labels

**PR1** (lines 177-179)
```current
(iii)~\emph{Reaction}: the user's implicit reaction $F_t\in\{-1,0,+1\}$ (contradicts, no reaction, accepts) is recorded and shown to the agent in later turns.
```
Replace with:
> (iii)~\emph{Reaction}: the user's implicit reaction $F_t\in\{-1,0,+1\}$ (contradicts, no reaction, accepts) is recorded and shown to the agent for its next six turns.

Why: Section 4 defines the context X as what the agent is shown, and both `\eta_\varnothing` and the scope of Theorem 2 depend on it. The prompt window is `max_turns=6`, so each reaction is visible for the next six turns.

**LB1** (lines 183-185, minor)
```current
With confident set $H=\{t:p_t\ge\tau\}$, $\tau=0.6$, the \emph{composite} label $L$ (Eq.~\ref{eq:label}) marks a session in which at least half of the confident commitments are wrong.
```
Replace with:
> With confident set $H=\{t:p_t\ge\tau\}$, $\tau=0.6$, the \emph{composite} label $L$ (Eq.~\eqref{eq:label}) marks a session in which at least half of the confident commitments are wrong, and is $0$ if there are none.

Why: Eq. (1) now requires |H| >= 1, which matches the code (`eval_label_v2`).

### Detectors

**DT1** (lines 203-204)
```current
\paragraph{PGE posterior.} Proposition~\ref{prop:posterior} gives the score.
```
Replace with:
> \paragraph{PGE posterior.} The score is a plug-in estimate of the posterior of Proposition~\ref{prop:posterior}.

**DT2** (lines 206-208)
```current
Cell priors are estimated from training reactions by the method of moments (Proposition~\ref{prop:identify}), shrunk toward attribute-level and bucket-level estimates (strength $k=5$).
```
Replace with:
> Cell priors are estimated from training reactions by the method of moments (Proposition~\ref{prop:identify}), shrunk toward attribute-level and bucket-level estimates (strength $k=5$); cells on the two attributes that never receive reactions are not identified and get the bucket-level prior pooled over the other attributes.

**DT3** (lines 213-214). Insert after the sentence that ends "...the special case $r_0=g_1=\varepsilon$, $r_1=g_0=1-\varepsilon$.":
> Proposition~\ref{prop:posterior} covers this channel, since its proof uses only $P(F_t\mid w_t)$; the channel scale is chosen on training labels because reactions alone do not identify it (Proposition~\ref{prop:identify}).

**DT4** (lines 215-217). Recommended addition after the existing sentence:
```current
For $\LU$ the label event is restricted to commitments on attributes that depend on a hidden field anywhere in the schema, a set the detector can know without the clinic's rules.
```
Add:
> V6 is therefore the posterior of a proxy event ($L_A$ of Section~\ref{sec:theory} with $A$ the schema-level set), not of $\LU$.

See section 3, item 1, for the size of the mismatch.

### Experimental design

**ED1** (lines 288-295)
```current
Because the simulator's rules and hidden-field prior are known, we can compute exactly $\eta_0=P(\LU=1\mid\text{intake},C)$, the best any detector that sees the intake and the commitments can do without reactions, and $\eta_F=P(\LU=1\mid\text{intake},C,F)$ with the true channel.
```
Replace with:
> Because the simulator's rules and hidden-field prior are known, we can compute exactly the \emph{population risk} $\bar\eta=\sum_{h}P(h)\,\LU(C,h)$, the probability that the session's commitments would be wrong for a random user with the same intake (hidden fields $h$), and $\eta_F=P(\LU=1\mid\text{intake},C,F)$ with the true channel. No detector that sees the commitments and reactions, with or without resamples, can beat $\eta_F$ (Theorem~\ref{thm:bound} and NP). $\bar\eta$ would be the best context-only predictor for an agent whose commitments ignore the reactions; ours are shown them, so a context-only detector can in principle exceed it.

Why (three separate problems):
1. *Notation clash.* In theory.tex, `\eta_0` is the perfect-feedback posterior (eps = 0), and the no-feedback posterior is `\eta_\varnothing`. Rename main.tex's eta_0 everywhere: here, in Table `tab:decomp` (header, caption, pooled row) and in §7.3.
2. *Wrong claim.* `e11_confirmatory.bayes_LU(traj, setup)` with `eps=None` computes sum_h p(h) L_U(C,h). This equals P(L_U | intake, C) only if C is independent of the hidden fields given the intake. In v2fb the agent is shown the raw reactions and adapts: on the dev data, P(next p >= 0.6) is 0.42 after a contradiction vs 0.68 after an acceptance. So C carries reaction information, and "the best any detector that sees the intake and the commitments can do" is false.
3. *Wrong bound for sampling.* Theorem 1's bound for sampling detectors is `\eta_\varnothing = P(L_U | X, C)`, which conditions on the reactions the agent was shown. It *is* computable exactly in the simulator (section 3, item 2).

`\eta_F` is correct as stated. It equals P(L_U | X, C, F), because the agent's policy factors do not involve the hidden fields and cancel (proof of Proposition 1, which needs no (A1) for this). It is the true ceiling for every detector of (X, C, F, samples).

### Results

**RS1** (lines 325-329)
```current
\chk{On commitments whose truth depends on undisclosed facts, agents that are self-consistent repeat the same answer whether it is right or wrong ($e_1\approx e_0$), so disagreement carries little information, while one reaction with 25\% noise carries a great deal.}
```
Replace "($e_1\approx e_0$)" with:
> ($e_0$ and $e_1$ small and close, so by Corollary~\ref{cor:degenerate} disagreement's AUROC lies within $\tfrac12e_0e_1$ of $\tfrac12+\tfrac12(e_1-e_0)$)

Why: e1 ≈ e0 alone does not make disagreement uninformative when both are large. For dev 8B, e1 = 0.538 and e0 = 0.496, and the sharp interval is [0.388, 0.655].

Suggestion: add cap and sharp-interval columns to Table `tab:percommit`. numbers.tex already has `D/PC/*/cor1_bound`; `make_numbers.py` would need `C/PC/*/cor1_bound` and the interval endpoints.

**RS2** (lines 431-436)
```current
\chk{The best possible context-only predictor $\eta_0$ is far above every sampling detector: much of $\LU$ is population risk, visible in \emph{which} values an agent commits to. The user's reactions add information no context-only detector can have ($\eta_F-\eta_0$, H5), and this is the part sampling cannot reach even in principle.}
```
Replace with:
> \chk{The population risk $\bar\eta$ is far above every sampling detector: much of $\LU$ is visible in \emph{which} values an agent commits to. The user's reactions add $\eta_F-\bar\eta$ beyond it (H5). Because the agent is shown earlier reactions, a context-only or sampling detector could in principle recover part of this through the agent's later commitments; what sampling cannot reach even in principle is what $\eta_F$ adds to $\eta_\varnothing=P(\LU=1\mid X,C)$ (Theorem~\ref{thm:bound}).}

If `\eta_\varnothing` is computed, report `\eta_F - \eta_\varnothing` next to H5. That is the theorem's quantity: the last reaction plus anything the 6-turn display hid.

**RS3** (lines 473-476, precision)
```current
Structure-based detection does not help: graphon likelihoods score at or below chance on every corpus, and a latent error model over the dependency graph is no better than independence ($0.732$ vs.\ $0.734$ without value conditioning).
```
Replace "a latent error model over the dependency graph" with:
> exact inference in a simple cascade model over the dependency graph (errors depend on attribute, bucket, and whether any parent is wrong)

Why: `DagPosterior` is exact inference in an approximate cascade model (proof_check §4). theory.tex now cites this subsection (`sec:exp-robust`) for "a simple cascade model does not help".

### Limitations (add; the old "Theory" limitation was dropped in the restructure)

Insert, e.g. after the "Supervision" item (line 491):
> \emph{Theory.} Proposition~\ref{prop:posterior} is exact only under (A1)--(A2) with the true priors and channel, and the detector is a plug-in estimate; Theorem~\ref{thm:feedback} covers reactions regenerated for the detector with the agent's trajectories fixed, not the protocol's own noise, because the agent conditions on the reactions it is shown.

### Conclusion

**CC1** (lines 497-499)
```current
When an agent invents a fact about its user, its own samples cannot tell which user it is wrong about. Some of the risk is visible in what the agent commits to; the rest is visible only to the user.
```
Replace with:
> When an agent invents a fact about its user, its own samples cannot tell which of the users sharing its context it is wrong about. Some of the risk is visible in what the agent commits to; the rest is visible only through the user's reactions.

### Checked, no change needed
- Lines 100-105 (Contribution 1: "A within-context impossibility result and a dispersion bound"): consistent. Optionally write "a two-sided dispersion bound".
- Lines 127-132 (Related Work: "...are all functions of the model given its context, so Theorem~\ref{thm:bound} applies to them"): consistent. Search results are a u-free kernel.
- Lines 437-441 (V6 targets "only a proxy for the label's support"): consistent with Proposition 1's L_A remark.
- Lines 468-469 (V6 "recovers the channel from data"): consistent. The channel scale comes from labels (Proposition 2).
- Lines 488-493 ("the posterior can run without labels when the channel is known"; white-box probes "share the within-context limit, but ... could exploit population risk"): consistent.

## 3. Facts the new text should reflect

1. **Size of the V6 proxy mismatch (dev corpora, seed 42, reconstructed as in `e11_confirmatory.load`).**
   - `USER_DEPENDENT_SCHEMA` has 7 attributes. `user_dependent_attrs(intake)` has 0-5 and is always a subset.
   - `transport_assistance` is in the schema set but is determined by the intake in all 162 intake combinations. So "attributes that depend on a hidden field anywhere in the schema" (7, line 216) and "Six attributes can depend on a hidden field" (line 244) both hold, but they describe different sets. Say so.
   - The confident event sets differ in 539/600 sessions (89.8%, 70B) and 156/200 (78.0%, 8B).
   - In 158 (70B) and 78 (8B) sessions the schema event is non-empty while L_U's is empty.
   - The schema-event label disagrees with L_U on 27.5% (70B) and 20.0% (8B) of sessions.
   - This is the gap that `V6-session` closes; it supports the "\pending{gap}" sentence at line 439.
2. **Computing `\eta_\varnothing` exactly (optional, cheap).**
   - `\eta_\varnothing = P(L_U | X, C)` is `bayes_LU(traj, setup, eps)` with the reaction likelihood restricted to the reactions the agent was shown. With the 6-turn window that is every turn except the last: each reaction is displayed in the next turn's prompt.
   - Then `eta_F - \eta_\varnothing` is the information sampling cannot reach (Theorem 1), and `\eta_\varnothing - \bar\eta` is reaction information that reaches the context.
3. **Feedback increment (H2).**
   - Under (A1), V6-prior scores the tail over the cell priors (population base rates). V6 minus V6-prior isolates the reactions' likelihood ratios.
   - V6-prior and ctx-LR are not reaction-free in v2fb, because the agent picks later commitments after seeing reactions. Word H2 as the prereg does: "beyond population base rates".
4. **Theorem 2 and the development noise sweep.** If the dev noise sweep (old Fig. 2) returns, describe it as reactions regenerated for the detector with trajectories fixed. That is the only setting Theorem 2 covers (counterexample: AUROC 0.50 at eps = 0 vs 0.71 at eps = 0.25 when the agent conditions on the reactions).

## 4. Dependencies of theory.tex on main.tex

- **Labels.** theory.tex refers to `sec:exp-dissociation` (§7.1) and `sec:exp-robust` (§7.6). Keep both. The old `sec:exp-ablation` no longer exists, and its reference was retargeted to `sec:exp-robust`, where the cascade result now lives.
- **Macros.** theory.tex uses `\LU` and `\val{...}` from main.tex's preamble. The old `theory_check.tex` wrapper no longer compiles standalone without them.
- **numbers.tex keys used by theory.tex** (all present):
  - `D/PC/70B/e1` = 0.079
  - `D/PC/70B/e0` = 0.078
  - `D/PC/70B/cor1_bound` = 0.576
  - `D/PC/70B/auroc_disagreement` = 0.500
  - `D/PC/70B/auroc_feedback_bit` = 0.764
  - `D/PC/8B/cor1_bound` = 0.884

  These replace the old e9 numbers (0.103/0.053, 0.575, 0.525, 0.871). The e9 numbers came from the five resampled schema attributes and would contradict §7.1, which uses the per-session user-dependent commitments.
- **Titles renamed in theory.tex:**
  - §4: "What Resampling and Feedback Can Reveal"
  - Corollary 1: "Dispersion under systematic fabrication"
  - Theorem 2: "Less noisy feedback cannot hurt"
  - Proposition 2: "Priors are identified where feedback exists"
- **Rendering.** Use `\varnothing`, not `\emptyset`: under mathptmx, `\emptyset` renders as a slashed zero. `\mathbb{1}` renders as "⊮" with amssymb, so theory.tex now uses `\mathbf{1}`.

## 5. Not theory (for completeness)

- Code nit (read-only here): the `ChannelPosterior` docstring cites "Proposition 4". The paper's number is Proposition 2.
