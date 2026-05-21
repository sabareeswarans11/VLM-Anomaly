import Foundation

/// Loads prompts/*.yaml from the app bundle and renders by ``name.variant`` key.
/// Byte-identical to the Python ``PromptLibrary`` in evaluators/prompt_library.py.
final class PromptBuilder {
    static let shared = PromptBuilder()

    private var prompts: [String: [String: String]] = [:]

    private init() {
        loadAll()
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Render the prompt for `key` (format: ``"<name>.<variant>"``).
    /// Returns a fallback string if the key is not found.
    func render(key: String) -> String {
        let parts = key.split(separator: ".", maxSplits: 1).map(String.init)
        guard parts.count == 2 else {
            return "Is there a defect? Reply with JSON: {\"is_anomalous\": bool, \"confidence\": float}"
        }
        let name = parts[0], variant = parts[1]
        return prompts[name]?[variant]?.trimmingCharacters(in: .whitespacesAndNewlines)
            ?? "Is there a defect in this image? Reply with JSON."
    }

    func availableKeys() -> [String] {
        prompts.sorted { $0.key < $1.key }.flatMap { name, variants in
            variants.keys.sorted().map { "\(name).\($0)" }
        }
    }

    // ── YAML loader ─────────────────────────────────────────────────────────
    // Uses a minimal hand-rolled YAML parser for the simple key: value |
    // block-scalar structure used in prompts/*.yaml — no external dependency.

    private func loadAll() {
        let bundle = Bundle.main
        guard let urls = bundle.urls(forResourcesWithExtension: "yaml", subdirectory: "prompts")
        else { return }

        for url in urls.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            if let text = try? String(contentsOf: url, encoding: .utf8) {
                parseYAML(text)
            }
        }
    }

    private func parseYAML(_ text: String) {
        var name: String?
        var variants: [String: String] = [:]
        var currentVariant: String?
        var currentLines: [String] = []
        var inVariants = false

        func flush() {
            guard let v = currentVariant else { return }
            variants[v] = currentLines.joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            currentVariant = nil
            currentLines = []
        }

        for rawLine in text.components(separatedBy: "\n") {
            let line = rawLine

            // Top-level name:
            if !inVariants, let m = line.firstMatch(of: /^name:\s*(.+)/) {
                name = String(m.output.1).trimmingCharacters(in: .whitespaces)
                continue
            }

            // variants: section marker
            if line.hasPrefix("variants:") {
                inVariants = true
                continue
            }

            guard inVariants else { continue }

            // variant key (2-space indent, ends with |)
            if let m = line.firstMatch(of: /^  (\w+):\s*\|?\s*$/) {
                flush()
                currentVariant = String(m.output.1)
                continue
            }

            // block scalar line (4-space indent)
            if line.hasPrefix("    ") {
                currentLines.append(String(line.dropFirst(4)))
            } else if line.hasPrefix("  ") && currentVariant != nil {
                // continuation at 2-space (some editors trim trailing spaces)
                currentLines.append(String(line.dropFirst(2)))
            }
        }
        flush()

        if let n = name, !variants.isEmpty {
            prompts[n] = variants
        }
    }
}
