# Project working rules

- Work only on the stage/substep explicitly authorized by the user. Complete its checks, document facts and limitations, push when requested, then stop. Do not start Benchmark, reliability, fusion, training or UI work without authorization.
- User preference (2026-10-16): GPU use is authorized for this project within the local 12 GB VRAM budget. Prefer GPU acceleration where supported and faster. Measure actual memory; this is not permission for paid/cloud resources or an exemption from stage boundaries.
- Preserve actual failures, OOMs and all comparable runs. Do not silently change image counts, resolution or algorithm settings. Explicitly identify resource-profile changes and upstream capabilities.
- Keep raw data, weights, vendor checkouts and large generated artifacts out of Git. Preserve existing local learning experiments.
