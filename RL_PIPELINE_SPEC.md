# GRPO-Based RL Pipeline Spec for HTML PPT/Document Generation

## 1. Purpose

This document specifies a reinforcement learning pipeline for training a model to generate HTML-based slides or documents.

The core goal is not just to produce plausible HTML text, but to train a policy that reliably generates HTML that:

1. renders successfully in a browser environment,
2. avoids runtime and layout failures,
3. produces visually high-quality screenshots,
4. satisfies the original document-generation intent.

The primary learning algorithm for v1 is **GRPO**, because the task naturally provides execution-grounded rewards from browser rendering and downstream visual evaluation.

---

## 2. Problem Framing

### 2.1 Task Definition

Given a prompt such as:

- "Create a 1-page startup pitch slide"
- "Create a one-page research summary"
- "Create a clean HTML report page for quarterly metrics"

the policy generates HTML (or an intermediate layout representation compiled to HTML), which is then evaluated by an external environment.

### 2.2 Why GRPO Instead of DPO

This task is better framed as **environment-grounded policy optimization** than pure preference alignment.

Reasons:

1. **Render success/failure is executable feedback** rather than only pairwise human preference.
2. **Browser errors, timeouts, and layout breakage** are direct reward signals.
3. **Screenshot quality** can be evaluated after rollout completion.
4. The same prompt can generate multiple candidates, making grouped relative optimization natural.

DPO may still be used later as a supporting method, but the main optimization target in v1 is GRPO.

---

## 3. Scope

## 3.1 In Scope for v1

- single-page or single-slide rollout unit,
- HTML-based generation,
- browser-based rendering and screenshot capture,
- render-failure reward,
- screenshot-based quality reward,
- prompt-fidelity reward,
- grouped candidate sampling for GRPO,
- offline or batched RL training loop.

## 3.2 Out of Scope for v1

- full multi-page document coherence optimization,
- complex interactive JavaScript applications,
- arbitrary external asset fetching without controls,
- human-in-the-loop real-time annotation UI,
- full production orchestration and autoscaling.

---

## 4. Rollout Unit

### 4.1 Recommended v1 Unit

The rollout unit for v1 is **one page / one slide per episode**.

### 4.2 Rationale

- simpler credit assignment,
- easier debugging of render failures,
- lower reward variance,
- better throughput per GRPO group,
- easier benchmark design.

### 4.3 Future Extension

In v2, the unit can be expanded to full multi-page documents or decks, with additional rewards for style consistency and narrative flow.

---

## 5. Output Representation

### 5.1 v1 Recommendation

There are two possible policy output modes:

1. **Raw HTML/CSS output**
2. **Intermediate layout DSL / JSON spec compiled into HTML**

### 5.2 Preferred Option

Prefer **intermediate structured representation** when possible.

Example:

```json
{
  "title": "Q4 Growth",
  "layout": "hero_chart",
  "theme": "clean_blue",
  "elements": [
    {"type": "heading", "text": "Q4 Growth"},
    {"type": "chart", "kind": "bar"},
    {"type": "bullets", "items": ["Revenue +32%", "Margin +8%"]}
  ]
}
```

### 5.3 Why Structured Output Is Preferred

- reduces invalid HTML generation,
- reduces action-space complexity,
- improves reward stability,
- simplifies layout validation,
- makes downstream rendering more deterministic.

If raw HTML is used in v1, it should still be sandboxed and validated aggressively.

---

## 6. End-to-End Architecture

```text
Task Sampler
  -> Policy Sampler (K candidates per task)
  -> Render Worker
      -> fail path: traceback/logs/failure reward
      -> success path: Screenshot Worker
            -> Judge Worker
            -> Reward Aggregator
  -> Grouped Rollout Store
  -> GRPO Trainer
  -> Evaluation Pipeline
```

### 6.1 Core Components

1. **Task Sampler**
   - produces prompts and constraints
2. **Policy Sampler**
   - generates K candidate outputs per task
3. **Render Worker**
   - runs browser rendering and collects execution signals
4. **Screenshot Worker**
   - captures render output artifacts
5. **Judge Worker**
   - scores visual quality using a multimodal LLM
6. **Reward Aggregator**
   - merges render, visual, and fidelity rewards
7. **GRPO Trainer**
   - performs grouped policy optimization
8. **Evaluation Pipeline**
   - tracks benchmark quality over time

---

## 7. Task Sampler

### 7.1 Responsibilities

- sample training prompts,
- attach document type,
- attach style/layout constraints,
- define page count and target format,
- assign split metadata.

### 7.2 Example Task Schema

```json
{
  "task_id": "task_0001",
  "prompt": "Create a clean one-page startup pitch slide for an AI education company.",
  "doc_type": "slide",
  "constraints": {
    "page_count": 1,
    "aspect_ratio": "16:9",
    "theme": "minimal",
    "must_include": ["title", "problem", "solution", "traction"]
  },
  "split": "train"
}
```

---

## 8. Policy Sampler

### 8.1 Responsibilities

- take a task as input,
- sample K candidate outputs,
- record sampling metadata,
- store raw generation artifacts.

### 8.2 Candidate Count

Recommended v1 group size:

- **K = 4 to 8** candidates per task.

If judge cost is high, start with **K = 4**.

### 8.3 Candidate Schema

```json
{
  "candidate_id": "cand_0001",
  "task_id": "task_0001",
  "model_version": "policy_v0",
  "sampling_params": {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 4096
  },
  "output_format": "html",
  "html": "<html>...</html>"
}
```

---

## 9. Render Worker

### 9.1 Recommended Runtime

Use **Playwright with headless Chromium** for v1.

### 9.2 Responsibilities

- load HTML in a controlled browser environment,
- wait for render completion,
- capture runtime errors,
- detect timeouts,
- detect blank renders,
- detect severe overflow/clipping,
- persist logs and failure signals.

### 9.3 Failure Taxonomy

The render worker must classify failures into a fixed taxonomy:

- `timeout`
- `js_exception`
- `asset_missing`
- `blank_render`
- `severe_overflow`
- `invalid_html`
- `unknown_render_failure`

### 9.4 Render Result Schema

```json
{
  "candidate_id": "cand_0001",
  "render_status": "success",
  "error_type": null,
  "traceback": null,
  "console_errors": [],
  "timeout": false,
  "blank_page": false,
  "overflow_score": 0.12,
  "render_time_ms": 1430,
  "screenshot_ready": true
}
```

### 9.5 Render Traceback Signal

The render layer should return both:

1. raw execution artifacts,
2. normalized failure labels.

This enables:

- direct reward assignment,
- training-time filtering,
- future self-repair supervision,
- failure-mode analytics.

---

## 10. Screenshot Worker

### 10.1 Responsibilities

For successful renders:

- capture full-page screenshot,
- optionally capture viewport-specific screenshot,
- extract visible text,
- optionally store DOM snapshot.

### 10.2 Screenshot Artifact Schema

```json
{
  "candidate_id": "cand_0001",
  "screenshot_path": "artifacts/task_0001/cand_0001.png",
  "dom_path": "artifacts/task_0001/cand_0001.html",
  "extracted_text": "Q4 Growth Revenue +32% Margin +8%"
}
```

### 10.3 Why Text Extraction Matters

Screenshot-only judging is insufficient for content fidelity. The judge should ideally receive:

- original prompt,
- screenshot,
- extracted visible text,
- optional structured constraints.

---

## 11. Judge Worker

### 11.1 Role

The judge worker evaluates **visual quality** and optionally **content fidelity** using a commercial multimodal LLM.

### 11.2 Preferred Evaluation Mode

For v1, use **pairwise comparisons** among successful candidates from the same task.

Pairwise mode is preferred over absolute scalar scoring because it:

- reduces calibration drift,
- matches grouped sampling naturally,
- is more robust for subjective visual judgments,
- produces clearer relative preferences.

### 11.3 Judge Inputs

The judge should receive:

- task prompt,
- screenshot A,
- screenshot B,
- extracted text A,
- extracted text B,
- rubric instructions.

### 11.4 Judge Rubric

The rubric should cover:

- readability,
- alignment and spacing,
- visual hierarchy,
- clipping / overflow absence,
- professionalism,
- prompt fidelity,
- content completeness.

### 11.5 Judge Output Schema

```json
{
  "judge_id": "judge_0001",
  "task_id": "task_0001",
  "candidate_a": "cand_0001",
  "candidate_b": "cand_0002",
  "winner": "cand_0002",
  "confidence": 0.81,
  "rubric_scores": {
    "readability": {"a": 6, "b": 8},
    "layout_balance": {"a": 5, "b": 8},
    "visual_hierarchy": {"a": 6, "b": 9},
    "professionalism": {"a": 6, "b": 8}
  },
  "reason": "Candidate B has clearer hierarchy and less crowding."
}
```

### 11.6 Judge Reliability Controls

To reduce reward noise:

- use fixed prompts,
- periodically re-score the same examples,
- maintain a human-audited calibration set,
- monitor disagreement rates over time.

---

## 12. Reward Design

Reward should be separated into three components:

1. **render reward**
2. **visual reward**
3. **fidelity reward**

### 12.1 Render Reward

Render reward should dominate early training.

Example mapping:

```text
success                    +0.3
timeout                    -1.0
js_exception               -0.9
blank_render               -0.8
asset_missing              -0.6
severe_overflow            -0.5
minor_overflow             -0.2
```

### 12.2 Visual Reward

Visual reward is derived from judge results and normalized into `[0, 1]`.

Possible sources:

- pairwise win rate within group,
- rubric subscore average,
- tournament ranking.

### 12.3 Fidelity Reward

Fidelity reward measures whether the output satisfies prompt requirements.

Examples:

- required sections included,
- requested title present,
- requested page count respected,
- required content elements present,
- requested chart/table included.

Normalize into `[0, 1]`.

### 12.4 v1 Reward Formula

Recommended simple formula:

```python
if render_failed:
    total_reward = -1.0
else:
    total_reward = 0.3 + 0.5 * visual_reward + 0.2 * fidelity_reward
```

### 12.5 Alternative Formula

```python
total_reward = (
    render_base_reward
    - overflow_penalty
    - js_error_penalty
    + 0.45 * visual_reward
    + 0.25 * fidelity_reward
)
```

### 12.6 Reward Prioritization Strategy

Training should emphasize:

1. **first:** renderability,
2. **second:** visual quality,
3. **third:** prompt fidelity sophistication.

This order is important because the visual judge is only meaningful once browser execution becomes stable.

---

## 13. GRPO Group Construction

### 13.1 Group Definition

Each GRPO group consists of K candidates sampled from the same task prompt.

Example:

```text
task_0001
  cand_1 -> reward -1.0
  cand_2 -> reward  0.2
  cand_3 -> reward  0.7
  cand_4 -> reward  0.5
```

### 13.2 Why This Fits GRPO

- same task prompt,
- multiple candidate rollouts,
- relative ranking inside group,
- execution-derived rewards,
- easy grouped advantage estimation.

### 13.3 Stored Group Schema

```json
{
  "group_id": "group_0001",
  "task_id": "task_0001",
  "candidate_ids": ["cand_1", "cand_2", "cand_3", "cand_4"],
  "rewards": [-1.0, 0.2, 0.7, 0.5]
}
```

---

## 14. Training Curriculum

### 14.1 Bootstrap Requirement

Do not start with RL-only training from scratch.

The policy should first receive **SFT warm-starting** using:

- clean HTML examples,
- template-based document examples,
- human-written or curated slide/page layouts,
- synthetic but validated examples.

### 14.2 Curriculum Stages

#### Phase 0: SFT Bootstrap

Goal:

- learn basic HTML/page structure,
- avoid immediate catastrophic render failure.

#### Phase 1: Render-Focused GRPO

Goal:

- maximize successful render rate,
- reduce runtime and layout failures,
- stabilize environment interaction.

At this stage, visual reward should be low-weight or absent.

#### Phase 2: Visual Reward Integration

Goal:

- improve layout quality,
- reduce crowding,
- improve readability and hierarchy.

#### Phase 3: Fidelity and Harder Tasks

Goal:

- improve adherence to prompt requirements,
- introduce denser content,
- handle charts, tables, and richer layouts.

#### Phase 4: Multi-Page Extension (Future)

Goal:

- learn deck-level consistency,
- style continuity,
- narrative progression.

---

## 15. Data Model

### 15.1 tasks

```json
{
  "task_id": "task_0001",
  "prompt": "Create a clean startup pitch slide.",
  "doc_type": "slide",
  "constraints": {
    "page_count": 1,
    "aspect_ratio": "16:9"
  },
  "split": "train"
}
```

### 15.2 candidates

```json
{
  "candidate_id": "cand_0001",
  "task_id": "task_0001",
  "model_version": "policy_v0",
  "sampling_params": {
    "temperature": 0.8,
    "top_p": 0.95
  },
  "output_format": "html",
  "html": "<html>...</html>"
}
```

### 15.3 render_results

```json
{
  "candidate_id": "cand_0001",
  "status": "success",
  "error_type": null,
  "traceback": null,
  "console_errors": [],
  "overflow_score": 0.12,
  "blank_page": false
}
```

### 15.4 artifacts

```json
{
  "candidate_id": "cand_0001",
  "screenshot_path": "artifacts/task_0001/cand_0001.png",
  "dom_path": "artifacts/task_0001/cand_0001.html",
  "extracted_text": "Q4 Growth Revenue +32% Margin +8%"
}
```

### 15.5 judge_results

```json
{
  "judge_id": "judge_0001",
  "task_id": "task_0001",
  "mode": "pairwise",
  "candidate_a": "cand_0001",
  "candidate_b": "cand_0002",
  "winner": "cand_0002",
  "confidence": 0.81,
  "rubric_scores": {
    "readability": {"a": 6, "b": 8}
  },
  "reason": "Candidate B is clearer and less crowded."
}
```

### 15.6 rewards

```json
{
  "candidate_id": "cand_0001",
  "render_reward": 0.3,
  "visual_reward": 0.72,
  "fidelity_reward": 0.66,
  "total_reward": 0.79
}
```

### 15.7 trajectory_groups

```json
{
  "group_id": "group_0001",
  "task_id": "task_0001",
  "candidate_ids": ["cand_1", "cand_2", "cand_3", "cand_4"],
  "rewards": [-1.0, 0.2, 0.7, 0.5]
}
```

---

## 16. Training Loop

### 16.1 High-Level Loop

```text
sample task batch
  -> generate K candidates per task
  -> render all candidates
  -> screenshot successful candidates
  -> judge successful candidates
  -> aggregate rewards
  -> form GRPO groups
  -> update policy
  -> evaluate on fixed benchmark
```

### 16.2 Step-by-Step Description

1. sample a batch of tasks,
2. generate grouped candidates,
3. evaluate renderability,
4. collect render failures and logs,
5. judge successful outputs visually,
6. compute fidelity metrics,
7. aggregate final rewards,
8. compute grouped relative advantages,
9. run GRPO update,
10. periodically run benchmark evaluation.

---

## 17. Evaluation Plan

### 17.1 Fixed Benchmark Set

Maintain a held-out benchmark set of prompts spanning:

- simple title + bullets slides,
- dense report pages,
- chart-heavy slides,
- image-heavy layouts,
- multilingual text,
- long-text overflow cases.

### 17.2 Core Metrics

#### Render Metrics

- render success rate,
- hard failure rate,
- timeout rate,
- JS exception rate,
- blank render rate,
- overflow incidence.

#### Quality Metrics

- judge win rate,
- judge confidence distribution,
- human agreement on calibration set,
- readability score,
- layout quality score.

#### Fidelity Metrics

- required-section inclusion rate,
- prompt compliance rate,
- requested format compliance.

### 17.3 Human Validation

Maintain a small human-reviewed set to detect:

- judge drift,
- reward hacking,
- content-quality regressions,
- visually attractive but semantically wrong outputs.

---

## 18. Risks and Mitigations

### 18.1 Reward Noise from Judge

**Risk:** multimodal LLM judgments may vary across repeated evaluations.

**Mitigations:**

- use pairwise comparison,
- fix judge prompts,
- re-score sampled items periodically,
- track human agreement.

### 18.2 Reward Hacking

**Risk:** policy may generate outputs that look superficially attractive but fail to satisfy the prompt.

**Mitigations:**

- include prompt and extracted text in judging,
- add fidelity reward,
- add explicit content checks,
- keep a human-audited benchmark.

### 18.3 Early Training Collapse

**Risk:** if initial policy produces mostly invalid HTML, learning signal becomes too sparse.

**Mitigations:**

- SFT warm start,
- simple-task curriculum,
- render-focused early training,
- optional structured DSL output.

### 18.4 Action Space Complexity

**Risk:** raw HTML token generation is too unconstrained.

**Mitigations:**

- use intermediate schema,
- template-constrained layouts,
- compiler-based rendering path,
- lint and validation before browser render.

### 18.5 Non-Deterministic Rendering

**Risk:** asset loading, fonts, or environment differences create unstable rewards.

**Mitigations:**

- pin browser version,
- pin fonts and assets,
- sandbox execution,
- keep rendering environment deterministic.

---

## 19. Implementation Priorities

### Priority 1

- define task schema,
- implement grouped candidate generation,
- implement Playwright render worker,
- implement failure taxonomy,
- persist render artifacts.

### Priority 2

- implement screenshot capture,
- implement extracted-text pipeline,
- implement judge worker,
- implement reward aggregation.

### Priority 3

- implement GRPO training data export,
- implement benchmark evaluation,
- add calibration set and drift monitoring.

### Priority 4

- move from raw HTML to structured DSL,
- add self-repair loop,
- extend to multi-page reward.

---

## 20. Acceptance Criteria for v1

The v1 pipeline is considered functionally complete when:

1. a task prompt can produce K grouped candidates,
2. every candidate can be rendered in a controlled browser environment,
3. render failures are classified into the fixed failure taxonomy,
4. successful renders produce screenshots and extracted text,
5. successful candidates can be pairwise judged by a multimodal LLM,
6. render, visual, and fidelity rewards are aggregated into a per-candidate total reward,
7. grouped rewards are exportable in a format usable by a GRPO trainer,
8. a held-out benchmark can report render and quality metrics over time.

---

## 21. Recommended v1 Summary

The recommended v1 design is:

- **single-page / single-slide rollout unit**,
- **K-sample grouped candidate generation**,
- **Playwright-based render evaluation**,
- **render failure as primary negative signal**,
- **multimodal screenshot judge for visual quality**,
- **prompt fidelity reward for content correctness**,
- **GRPO as the main optimization method**,
- **SFT warm start before RL**,
- **structured output representation preferred over raw HTML where possible**.

This design prioritizes execution-grounded learning first, then visual quality, then richer document semantics.
