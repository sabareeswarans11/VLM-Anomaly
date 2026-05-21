import Foundation
import UIKit

// MARK: - MiniCPMRunner
// Wraps llama.cpp (primary path) for on-device MiniCPM-V inference.
// The llama.cpp iOS framework is added as a Swift Package dependency in Xcode:
//   https://github.com/ggerganov/llama.cpp  (tag: b3500+)
//
// When MLX path is preferred (Apple Silicon Mac dev), swap the backend
// to mlx-vlm via a compiler flag: -DUSE_MLX

/// Singleton runner managing the loaded GGUF model.
@MainActor
final class MiniCPMRunner {
    static let shared = MiniCPMRunner()

    private(set) var isModelLoaded = false
    private var modelPath: URL?

    // llama.cpp opaque pointers — declared as `Any?` so the file compiles
    // without the llama.cpp Swift package present.  In the real Xcode project
    // these are typed `OpaquePointer?`.
    private var llamaContext: Any?

    // ── Model location ──────────────────────────────────────────────────────

    private var modelURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("Models/minicpm-v-2.6-q4_k_m.gguf")
    }

    private let remoteURL = URL(string:
        "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/minicpm-v-2.6-q4_k_m.gguf"
    )!

    // ── Download & load ─────────────────────────────────────────────────────

    func downloadAndLoad(onProgress: @escaping (Double) -> Void) async throws {
        if modelURL.checkFileExists() {
            try await load()
            return
        }

        try FileManager.default.createDirectory(
            at: modelURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let (localURL, _) = try await URLSession.shared.download(
            from: remoteURL,
            delegate: ProgressDelegate(callback: onProgress)
        )
        try FileManager.default.moveItem(at: localURL, to: modelURL)
        try await load()
    }

    private func load() async throws {
        // In the real build, this calls llama_load_model_from_file() and
        // llama_new_context_with_model().  The mock below lets the app compile
        // and navigate without the llama.cpp package.
        isModelLoaded = true
    }

    // ── Inference ───────────────────────────────────────────────────────────

    /// Run MiniCPM-V on `image` with `prompt`.  Returns an `AnomalyResult`.
    func predict(image: UIImage, prompt: String) async -> AnomalyResult {
        guard isModelLoaded else {
            return AnomalyResult(isAnomalous: false, confidence: 0,
                                 description: "Model not loaded.",
                                 defectType: nil, latencyMs: 0,
                                 tokensPerSecond: 0, parseError: true)
        }

        let startWall = Date()
        var firstTokenDate: Date?
        var tokenCount = 0

        // In the real build:
        // 1. Encode `image` to JPEG bytes and build the llama.cpp multimodal
        //    embedding via clip_image_encode().
        // 2. Tokenise `prompt` and prepend the image token.
        // 3. llama_decode() in a loop, recording the first token time.
        // 4. Collect the full text output.
        // 5. Hand to AnomalyResponseParser.

        // ── Mock path (replace with real llama.cpp calls in Xcode) ──────────
        try? await Task.sleep(nanoseconds: 1_500_000_000)  // ~1.5 s simulated
        firstTokenDate = startWall.addingTimeInterval(1.8)
        tokenCount = 75
        let rawResponse = """
        {"is_anomalous": false, "confidence": 0.12,
         "description": "Surface appears uniform with no visible defects.",
         "defect_type": null}
        """
        // ────────────────────────────────────────────────────────────────────

        let totalMs = Date().timeIntervalSince(startWall) * 1000
        let firstTokenMs = firstTokenDate.map { $0.timeIntervalSince(startWall) * 1000 } ?? totalMs
        let elapsed = max(Date().timeIntervalSince(firstTokenDate ?? startWall), 0.001)
        let tokPerSec = Double(tokenCount) / elapsed

        let (parsed, parseError) = AnomalyResponseParser.shared.parse(rawResponse)

        return AnomalyResult(
            isAnomalous: parsed["is_anomalous"] as? Bool ?? false,
            confidence:  parsed["confidence"]  as? Double ?? 0,
            description: parsed["description"] as? String ?? "",
            defectType:  parsed["defect_type"] as? String,
            latencyMs:   firstTokenMs,
            tokensPerSecond: tokPerSec,
            parseError:  parseError
        )
    }
}

// MARK: - Helpers

private extension URL {
    func checkFileExists() -> Bool {
        FileManager.default.fileExists(atPath: path)
    }
}

private final class ProgressDelegate: NSObject, URLSessionDownloadDelegate {
    let callback: (Double) -> Void
    init(callback: @escaping (Double) -> Void) { self.callback = callback }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        callback(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {}
}
