# MuJoCo Starter Template

Use this scaffold for robotics simulation tasks. It is intentionally
minimal; for a complete working reference, read
`examples/mujoco-pendulum/`.

After creating a task, update:

- `instruction.md` with the model, policy, or control objective.
- `task.toml` with resources, timeouts, and required `/tmp/output/...`
  artifacts.
- `scorer/compute_score.py` with `TASK = RubricTask(...)` and a pure evaluator;
  use `RegularFileArtifact` plus `context.policy(...)` for submitted Python.
- `scorer/evaluation.plan.json`: sealed plan refreshed from `TASK` by harness reference/ground-truth (commit; never hand-edit).
- `data/` with public examples, starter assets, or environment clients.
- `scorer/data/` with private test fixtures, seeds, XML assets, or target
  specifications.
- `solution/solve.sh` with a reference solution when possible.

Common additions:

- MuJoCo XML assets under `scorer/data/assets/`.
- Hidden seeds under `scorer/data/seeds.json`.
- Public environment clients or examples under `data/`.
- GPU requirements in `task.toml` when training or evaluation needs CUDA.

If `scorer/compute_score.py` imports MuJoCo or another package that is installed
only in the task image, set `in_container = true` under `[ground_truth]`. Commit
the generated `scorer/evaluation.plan.json`; host CI validates that sealed file
without importing the scorer, then binds it back to `TASK.evaluation_plan`
inside the task image.

Before submitting, run:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```
