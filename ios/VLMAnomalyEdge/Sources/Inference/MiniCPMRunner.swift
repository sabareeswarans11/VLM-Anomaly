import Foundation
import UIKit
import llama  // llama.cpp Swift Package — add via project.yml packages: section

// MARK: - Public types

enum RunnerError: Error, LocalizedError {
    case modelLoadFailed
    case contextCreateFailed

    var errorDescription: String? {
        switch self {
        case .modelLoadFailed:    return "Failed to load GGUF model."
        case .contextCreateFailed: return "Failed to create llama context."
        }
    }
}

// Sendable container for inference results — crosses actor boundary safely.
struct InferenceOutput: Sendable {
    var text: String
    var firstTokenMs: Double
    var tokensPerSecond: Double
    var tokenCount: Int
}

// MARK: - LlamaEngine
// @unchecked Sendable: pointer fields are never shared concurrently —
// MiniCPMRunner serialises all access through Task.detached.

private final class LlamaEngine: @unchecked Sendable {

    static let modelFilename  = "MiniCPM-V-4.6-Q4_K_M.gguf"
    static let mmprojFilename = "mmproj-MiniCPM-V-4.6-Q8_0.gguf"

    var llamaModel:   OpaquePointer?   // llama_model *
    var llamaContext: OpaquePointer?   // llama_context *

    // Vision encoder — only available when LLAVA_AVAILABLE is defined.
    // Enable by adding a bridging header (see LlavaSupport.h).
    var clipContext:  OpaquePointer?   // clip_ctx *

    // MARK: Load

    func load(langPath: String, mmprojPath: String?) throws {
        llama_backend_init()

        // ── Language model ───────────────────────────────────────────────────
        var mparams = llama_model_default_params()
        mparams.n_gpu_layers = 99     // offload all layers to Metal on iPhone
        mparams.use_mmap     = true

        guard let m = llama_model_load_from_file(langPath, mparams) else {
            throw RunnerError.modelLoadFailed
        }

        var cparams = llama_context_default_params()
        cparams.n_ctx               = 4096
        cparams.n_batch             = 512
        let nThreads                = UInt32(ProcessInfo.processInfo.processorCount)
        cparams.n_threads           = nThreads
        cparams.n_threads_batch     = nThreads

        guard let c = llama_new_context_with_model(m, cparams) else {
            llama_model_free(m)
            throw RunnerError.contextCreateFailed
        }

        llamaModel   = m
        llamaContext = c

        // ── Vision encoder (mmproj) ──────────────────────────────────────────
        // clip_model_load is declared in common/clip.h.
        // Activate by defining LLAVA_AVAILABLE in Build Settings and adding
        // LlavaSupport.h as the Objective-C Bridging Header.
        #if LLAVA_AVAILABLE
        if let mmp = mmprojPath, FileManager.default.fileExists(atPath: mmp) {
            clipContext = clip_model_load(mmp, 0)
        }
        #endif
    }

    // MARK: Inference

    func run(image: UIImage, prompt: String) -> InferenceOutput {
        guard let ctx = llamaContext, let model = llamaModel else {
            return InferenceOutput(text: "", firstTokenMs: 0,
                                   tokensPerSecond: 0, tokenCount: 0)
        }

        let wallStart = Date()
        var firstTokenDate: Date?
        var tokenCount = 0

        llama_memory_clear(llama_get_memory(ctx), true)
        var nPast: Int32 = 0

        // ── 1. Image embedding (vision path) ─────────────────────────────────
        // Resize to 448×448 — MiniCPM-V expected input resolution.
        let hasVision = clipContext != nil
        #if LLAVA_AVAILABLE
        if let clip = clipContext,
           let pixels = image.resizedJpeg(side: 448) {
            pixels.withUnsafeBytes { raw in
                guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self)
                else { return }
                let embed = llava_image_embed_make_with_bytes(
                    clip,
                    Int32(ProcessInfo.processInfo.processorCount),
                    ptr,
                    Int32(pixels.count)
                )
                if let embed = embed {
                    // Inserts image tokens into KV cache; advances nPast.
                    llava_eval_image_embed(ctx, embed, 512, &nPast)
                    llava_image_embed_free(embed)
                }
            }
        }
        #endif

        // ── 2. Build prompt ───────────────────────────────────────────────────
        // MiniCPM-V 4.6 chat template — image embeddings are already in the KV
        // cache at positions 0..nPast, so we only tokenise the text portions.
        let (prefixText, suffixText) = chatPrompt(userPrompt: prompt, hasVision: hasVision)
        let fullText = hasVision ? suffixText : prefixText + suffixText
        let tokensToEval: [[llama_token]] = hasVision
            ? [tokenise(model: model, text: prefixText, addBOS: true),
               tokenise(model: model, text: suffixText, addBOS: false)]
            : [tokenise(model: model, text: fullText, addBOS: true)]

        // ── 3. Eval prompt tokens ─────────────────────────────────────────────
        var batch = llama_batch_init(512, 0, 1)
        defer { llama_batch_free(batch) }

        for (chunkIdx, tokenChunk) in tokensToEval.enumerated() {
            batchClear(&batch)
            let isLastChunk = chunkIdx == tokensToEval.count - 1
            for (i, tok) in tokenChunk.enumerated() {
                let isLast = isLastChunk && (i == tokenChunk.count - 1)
                batchAdd(&batch, token: tok, pos: nPast, seqId: 0, logit: isLast)
                nPast += 1
            }
            if batch.n_tokens > 0 {
                llama_decode(ctx, batch)
            }
        }

        // ── 4. Sampler — greedy (deterministic JSON output) ──────────────────
        let chainParams = llama_sampler_chain_default_params()
        let sampler     = llama_sampler_chain_init(chainParams)
        defer { llama_sampler_free(sampler) }
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy())

        let vocab    = llama_model_get_vocab(model)
        let eosToken = llama_vocab_eos(vocab)
        var generated = ""

        // ── 5. Decode loop ────────────────────────────────────────────────────
        for _ in 0..<256 {
            let newToken = llama_sampler_sample(sampler, ctx, -1)
            if newToken == eosToken { break }

            if firstTokenDate == nil { firstTokenDate = Date() }
            tokenCount += 1

            // Token → UTF-8 piece
            var piece = [CChar](repeating: 0, count: 128)
            let pieceLen = llama_token_to_piece(vocab, newToken, &piece, 128, 0, false)
            if pieceLen > 0 {
                generated += String(
                    bytes: piece[0..<Int(pieceLen)].map { UInt8(bitPattern: $0) },
                    encoding: .utf8
                ) ?? ""
            }

            // Advance KV cache
            batchClear(&batch)
            batchAdd(&batch, token: newToken, pos: nPast, seqId: 0, logit: true)
            llama_decode(ctx, batch)
            nPast += 1
        }

        // ── 6. Metrics ────────────────────────────────────────────────────────
        let now         = Date()
        let firstTokenMs = (firstTokenDate ?? now).timeIntervalSince(wallStart) * 1_000
        let genSec       = now.timeIntervalSince(firstTokenDate ?? wallStart)
        let tokPerSec    = Double(max(tokenCount, 1)) / max(genSec, 1e-3)

        return InferenceOutput(text: generated, firstTokenMs: firstTokenMs,
                               tokensPerSecond: tokPerSec, tokenCount: tokenCount)
    }

    // MARK: Private helpers

    /// MiniCPM-V 4.6 ChatML template.
    /// Returns (prefix, suffix) split around the image slot.
    private func chatPrompt(userPrompt: String, hasVision: Bool) -> (String, String) {
        let sys = "You are an industrial quality inspector. Reply ONLY with valid JSON."
        let prefix = "<|im_start|>system\n\(sys)<|im_end|>\n<|im_start|>user\n"
        let suffix = (hasVision ? "\n" : "") + "\(userPrompt)<|im_end|>\n<|im_start|>assistant\n"
        return (prefix, suffix)
    }

    private func tokenise(model: OpaquePointer, text: String, addBOS: Bool) -> [llama_token] {
        let vocab   = llama_model_get_vocab(model)
        let utf8    = text.cString(using: .utf8)!
        let maxTok  = text.utf8.count + 64
        var tokens  = [llama_token](repeating: 0, count: maxTok)
        let n = llama_tokenize(vocab, utf8, Int32(utf8.count - 1),
                               &tokens, Int32(maxTok), addBOS, true)
        return n > 0 ? Array(tokens[0..<Int(n)]) : []
    }
}

// MARK: - Batch helpers (mirrors llama.swiftui example)

private func batchAdd(_ batch: inout llama_batch,
                      token: llama_token, pos: llama_pos,
                      seqId: llama_seq_id, logit: Bool) {
    let i = Int(batch.n_tokens)
    batch.token[i]     = token
    batch.pos[i]       = pos
    batch.n_seq_id[i]  = 1
    batch.seq_id[i]![0] = seqId
    batch.logits[i]    = logit ? 1 : 0
    batch.n_tokens    += 1
}

private func batchClear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

// MARK: - MiniCPMRunner

@MainActor
final class MiniCPMRunner {
    static let shared = MiniCPMRunner()

    private(set) var isModelLoaded = false
    private let engine = LlamaEngine()

    // MARK: File locations

    static let remoteModelURL = URL(string:
        "https://huggingface.co/ggml-org/MiniCPM-V-4.6-GGUF/resolve/main/MiniCPM-V-4.6-Q4_K_M.gguf"
    )!
    static let remoteMmprojURL = URL(string:
        "https://huggingface.co/ggml-org/MiniCPM-V-4.6-GGUF/resolve/main/mmproj-MiniCPM-V-4.6-Q8_0.gguf"
    )!

    private func fileURL(_ filename: String) -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("Models/\(filename)")
    }

    // MARK: Download & load

    func downloadAndLoad(onProgress: @escaping (Double) -> Void) async throws {
        let langDest   = fileURL(LlamaEngine.modelFilename)
        let mmprojDest = fileURL(LlamaEngine.mmprojFilename)

        try FileManager.default.createDirectory(
            at: langDest.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        // Main model (~2.6 GB) accounts for 85 % of progress.
        if !langDest.checkFileExists() {
            let (tmp, _) = try await URLSession.shared.download(
                from: Self.remoteModelURL,
                delegate: ProgressDelegate { onProgress($0 * 0.85) }
            )
            try FileManager.default.moveItem(at: tmp, to: langDest)
        }

        // mmproj vision encoder (~0.5 GB) accounts for the last 15 %.
        if !mmprojDest.checkFileExists() {
            let (tmp, _) = try await URLSession.shared.download(
                from: Self.remoteMmprojURL,
                delegate: ProgressDelegate { onProgress(0.85 + $0 * 0.15) }
            )
            try FileManager.default.moveItem(at: tmp, to: mmprojDest)
        }

        onProgress(1.0)
        try await load()
    }

    private func load() async throws {
        let langPath   = fileURL(LlamaEngine.modelFilename).path
        let mmprojPath = fileURL(LlamaEngine.mmprojFilename).path
        let eng        = engine          // capture for detached task

        try await Task.detached(priority: .userInitiated) {
            try eng.load(langPath: langPath,
                         mmprojPath: FileManager.default.fileExists(atPath: mmprojPath)
                                        ? mmprojPath : nil)
        }.value

        isModelLoaded = true
    }

    // MARK: Inference

    /// Run MiniCPM-V on `image` with `prompt`.
    /// Executes on a detached task so the main thread is not blocked.
    func predict(image: UIImage, prompt: String) async -> AnomalyResult {
        guard isModelLoaded else {
            return .failed("Model not loaded.")
        }

        let eng = engine
        let out = await Task.detached(priority: .userInitiated) {
            eng.run(image: image, prompt: prompt)
        }.value

        let (parsed, parseError) = AnomalyResponseParser.shared.parse(out.text)
        return AnomalyResult(
            isAnomalous:     parsed["is_anomalous"] as? Bool   ?? false,
            confidence:      parsed["confidence"]   as? Double ?? 0,
            description:     parsed["description"]  as? String ?? out.text,
            defectType:      parsed["defect_type"]  as? String,
            latencyMs:       out.firstTokenMs,
            tokensPerSecond: out.tokensPerSecond,
            parseError:      parseError
        )
    }
}

// MARK: - AnomalyResult convenience

extension AnomalyResult {
    static func failed(_ message: String) -> AnomalyResult {
        AnomalyResult(isAnomalous: false, confidence: 0, description: message,
                      defectType: nil, latencyMs: 0, tokensPerSecond: 0, parseError: true)
    }
}

// MARK: - Private helpers

private extension URL {
    func checkFileExists() -> Bool { FileManager.default.fileExists(atPath: path) }
}

private extension UIImage {
    /// Returns JPEG data of the image resized to `side`×`side` pixels.
    func resizedJpeg(side: CGFloat) -> Data? {
        let size = CGSize(width: side, height: side)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        let resized = renderer.image { _ in self.draw(in: CGRect(origin: .zero, size: size)) }
        return resized.jpegData(compressionQuality: 0.9)
    }
}

private final class ProgressDelegate: NSObject, URLSessionDownloadDelegate {
    let callback: (Double) -> Void
    init(_ callback: @escaping (Double) -> Void) { self.callback = callback }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite expected: Int64) {
        guard expected > 0 else { return }
        callback(Double(totalBytesWritten) / Double(expected))
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {}
}
