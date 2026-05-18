# Task 05 — Edge MiniCPM-V Backend

Bring MiniCPM-V (4-bit) up locally so the same `VLMBackend.predict()` works
without a network. This is the macOS dev path for the iPhone story.

## Deliverables

- [ ] `backends/edge/minicpm_llamacpp.py` — wraps `llama-cpp-python` with a
      Q4_K_M GGUF. CPU on Intel, Metal on Apple Silicon.
- [ ] `backends/edge/minicpm_mlx.py` — wraps `mlx-vlm` on Apple Silicon.
- [ ] Both backends share one prompt path and one image-preprocessing step.
- [ ] `scripts/download_minicpm_gguf.sh` filled in.
- [ ] `scripts/run_edge_benchmark.py` produces first-token latency + tok/s.
- [ ] `tests/test_backends_edge.py` — smoke test against a tiny fixture GGUF.
      Marked `@pytest.mark.edge`.

## Done when

`python scripts/run_vlm_eval.py --backend edge_minicpm_llamacpp --dataset mvtec
--category bottle --limit 5` returns five predictions with no network.
