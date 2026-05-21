# Task 12 — iOS Edge App (VLMAnomalyEdge)

Ship the SwiftUI app that runs MiniCPM-V on-device and reproduces the
iPhone 16 Pro Max benchmark numbers from §1.

## Deliverables

- [x] Xcode project under `ios/VLMAnomalyEdge/` (iOS 17+, Xcode 16+).
- [x] `MiniCPMRunner.swift` — wraps llama.cpp (primary) or mlx-swift.
- [x] `PromptBuilder.swift` — loads the bundled copy of `prompts/*.yaml`,
      byte-identical to the Python toolkit.
- [x] `AnomalyResponseParser.swift` — behaviour parity with the Python
      `utils/json_parsing.py` (golden test files in both).
- [x] `CaptureView` → `ResultView` → `BenchmarkView` user flow.
- [x] `BenchmarkView` records first-token ms, tok/s, peak MB into a local
      SQLite store the user can inspect.
- [x] Airplane-mode test: app fully functional with the radio off.

## Done when

A clean install on an iPhone 16 Pro Max in airplane mode produces an
anomaly report in ≤ ~2 s first-token latency at ≥ ~17 tok/s, and the
benchmark log shows the run.
