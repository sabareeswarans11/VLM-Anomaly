# VLMAnomalyEdge iOS App

SwiftUI app that runs MiniCPM-V (4-bit GGUF) fully on-device via llama.cpp.
No network. No cloud. Works in airplane mode.

## Requirements

- Xcode 16+
- iOS 17+ target
- iPhone 15 Pro or later recommended (Neural Engine + 8 GB RAM)
- ~2.5 GB free storage (model download on first launch)

## Setup in Xcode

1. Open `VLMAnomalyEdge.xcodeproj` (create it below if missing).
2. Add the llama.cpp Swift Package:
   - File → Add Package Dependencies
   - URL: `https://github.com/ggerganov/llama.cpp`
   - Branch: `master` (or latest tag ≥ b3500)
3. Add `libllama.a` and `llama.h` to the Compile Sources and Link phases.
4. Set the Bundle Resources to include `Resources/prompts/`.
5. Select your iPhone target and hit Run.

## Creating the Xcode project (first time)

```bash
cd ios/VLMAnomalyEdge
# In Xcode: File → New → Project → iOS App
# Product Name: VLMAnomalyEdge
# Bundle ID: com.yourname.vlmanomalyedge
# Language: Swift, Interface: SwiftUI
# Then drag the Sources/ and Resources/ folders into the project navigator.
```

## Source layout

```
Sources/
├── VLMAnomalyEdgeApp.swift     App entry point + ModelStore
├── Views/
│   ├── CaptureView.swift       Photo picker → inference → result
│   ├── ResultView.swift        Anomaly verdict + metrics display
│   └── BenchmarkView.swift     Benchmark log (SwiftData SQLite)
├── Inference/
│   ├── MiniCPMRunner.swift     llama.cpp wrapper + download manager
│   └── PromptBuilder.swift     YAML prompt library (byte-identical to Python)
└── Parsing/
    └── AnomalyResponseParser.swift  5-step JSON fallback (mirrors json_parsing.py)

Resources/
└── prompts/                    Copy of repo prompts/*.yaml — bundled read-only
```

## Validated benchmark target

| Metric | Target |
|---|---|
| First-token latency | ≤ 2,000 ms |
| Throughput | ≥ 17.9 tok/s |
| Network required | None (airplane mode) |
| Device | iPhone 16 Pro Max |

Run the **Benchmark** tab after each analysis to log your numbers.

## Airplane-mode test

1. Enable airplane mode on the device.
2. Open the app — it should show "Model ready" (model already downloaded).
3. Pick a photo → Analyze.
4. Confirm a result appears without any network error.

## Parser parity

`AnomalyResponseParser.swift` mirrors the 5-step fallback chain in
`src/vlm_anomaly/utils/json_parsing.py`. Golden test cases live in
`tests/fixtures/` — add matching XCTest cases under `Tests/` to keep
both parsers in sync.
