# Project working rules

## Scope and authorization

- Work only on the stage/substep explicitly authorized by the user. Complete its checks, report facts and limitations, then stop. Do not start Benchmark, reliability, fusion, training, UI or Stage 7A work without explicit authorization.
- Learned matching and robust SfM initialization are a candidate research direction, not permission to implement them. Do not expand the heuristic fusion route without new authorization.
- Use an independent `codex/` branch for approved engineering changes. Protect main, user edits, untracked learning experiments and existing local artifacts.
- Before deleting, archiving or condensing historical material, provide exact paths, reasons, dependencies and expected effects, and wait for approval. Preserve recoverable evidence and old reproduction paths. Do not rewrite public Git history, force push, reset --hard, remove remote branches/Releases or choose a project license without separate authorization.

## Changes, commits and publication

- Only propose commits that improve engineering, algorithms, evaluation, testing, reproducibility or a reviewer's understanding of the project. Do not create many commits merely to record individual runs or debugging attempts.
- Keep raw training/inference/debug logs, temporary results, machine-specific environment dumps, full prediction caches, raw data, weights, vendor checkouts and large generated artifacts outside tracked files by default. Publish concise metrics, configuration, provenance and representative failures when they support a technical conclusion.
- Never hide negative results to improve presentation. Preserve auditable evidence of OOMs, degradation, no benefit, failed checks and reproduction limits; archival must retain a recovery route.
- Before committing, show the full Diff, actual test results (including skip reasons) and proposed file list. Check for private paths, secrets, binary caches, redundant reports and accidental generated files.
- Do not automatically commit or push after a task. Obtain user confirmation before commit and push; use clear `feat:`, `fix:`, `test:`, `refactor:` or `docs:` commit prefixes. Do not automatically squash or rewrite published commits.

## Experiments, resources and reporting

- Preserve actual failures, OOMs and all comparable runs. Do not silently change image counts, resolution or algorithm settings. Explicitly identify resource-profile changes and upstream capabilities.
- Local GPU use is authorized within the 12 GB VRAM budget. Prefer GPU acceleration where supported and faster; measure actual memory. This does not authorize paid/cloud resources or bypass stage boundaries.
- Distinguish implemented capabilities, evidence-supported findings and unproven novelty/performance. Acceptance, extra matches and lower fitted residuals alone do not establish accuracy gains. Include a brief cumulative contribution summary with project results.
- Keep known-intrinsics teaching runs separate from estimated-intrinsics baselines, independent evaluation separate from inference, and unknown evidence separate from known failure. Preserve data splits and evaluation protocols.
- New general-purpose entry points should accept explicit data, source, weight and output locations and refuse to overwrite prior results; changes to existing entry points need the authorized maintenance scope.
