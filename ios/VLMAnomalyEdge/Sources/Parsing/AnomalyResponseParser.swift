import Foundation

/// Behaviour-parity Swift implementation of Python's ``utils/json_parsing.py``.
///
/// Five-step fallback chain (mirrors CLAUDE.md §7.3):
/// 1. JSONSerialization on the full response.
/// 2. Extract first balanced ``{…}`` block.
/// 3. Strip markdown code fences.
/// 4. Regex extraction of known keys.
/// 5. Return parseError = true.
final class AnomalyResponseParser {
    static let shared = AnomalyResponseParser()

    private init() {}

    /// Parse a raw VLM response string.
    /// - Returns: ``(dict, parseError)`` — dict is empty when parseError is true.
    func parse(_ text: String) -> ([String: Any], Bool) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Step 1 — direct JSON
        if let d = try? JSONSerialization.jsonObject(with: Data(t.utf8)) as? [String: Any] {
            return (normalise(d), false)
        }

        // Step 2 — balanced braces
        if let block = extractBalancedBraces(t),
           let d = try? JSONSerialization.jsonObject(with: Data(block.utf8)) as? [String: Any] {
            return (normalise(d), false)
        }

        // Step 3 — strip code fences
        let stripped = stripCodeFences(t)
        if stripped != t {
            if let d = try? JSONSerialization.jsonObject(with: Data(stripped.utf8)) as? [String: Any] {
                return (normalise(d), false)
            }
            if let block = extractBalancedBraces(stripped),
               let d = try? JSONSerialization.jsonObject(with: Data(block.utf8)) as? [String: Any] {
                return (normalise(d), false)
            }
        }

        // Step 4 — regex extraction
        if let d = regexExtract(t) {
            return (normalise(d), false)
        }

        // Step 5 — give up
        return ([:], true)
    }

    // MARK: - Normalisation

    private func normalise(_ raw: [String: Any]) -> [String: Any] {
        var out: [String: Any] = [:]

        // is_anomalous — coerce string and int
        switch raw["is_anomalous"] {
        case let b as Bool:   out["is_anomalous"] = b
        case let s as String: out["is_anomalous"] = ["true","yes","1"].contains(s.lowercased())
        case let i as Int:    out["is_anomalous"] = i != 0
        default: return [:]  // required field missing
        }

        // confidence — clamp to [0,1]
        let isAnom = out["is_anomalous"] as? Bool ?? false
        switch raw["confidence"] {
        case let d as Double: out["confidence"] = max(0, min(1, d))
        case let s as String: out["confidence"] = Double(s).map { max(0, min(1, $0)) } ?? (isAnom ? 1.0 : 0.0)
        default:              out["confidence"] = isAnom ? 1.0 : 0.0
        }

        if let desc = raw["description"] as? String { out["description"] = desc }
        if let dt   = raw["defect_type"] as? String { out["defect_type"]  = dt }
        if let regs = raw["regions"]     as? [[String: Any]] { out["regions"] = regs }

        return out
    }

    // MARK: - Helpers

    private func extractBalancedBraces(_ text: String) -> String? {
        guard let start = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inString = false
        var prevChar: Character = " "
        for idx in text[start...].indices {
            let ch = text[idx]
            if ch == "\"" && prevChar != "\\" { inString.toggle() }
            if !inString {
                if ch == "{" { depth += 1 }
                else if ch == "}" {
                    depth -= 1
                    if depth == 0 { return String(text[start...idx]) }
                }
            }
            prevChar = ch
        }
        return nil
    }

    private func stripCodeFences(_ text: String) -> String {
        let pattern = /```(?:json)?\s*\n?([\s\S]*?)\n?```/
        if let m = text.firstMatch(of: pattern) {
            return String(m.output.1).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return text
    }

    private func regexExtract(_ text: String) -> [String: Any]? {
        guard let flagMatch = text.firstMatch(of: /"is_anomalous"\s*:\s*(true|false|"true"|"false"|"yes"|"no"|1|0)/i) else {
            return nil
        }
        let flagRaw = String(flagMatch.output.1).lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        var d: [String: Any] = ["is_anomalous": ["true","yes","1"].contains(flagRaw)]
        let isAnom = d["is_anomalous"] as? Bool ?? false

        if let cm = text.firstMatch(of: /"confidence"\s*:\s*([0-9]*\.?[0-9]+)/),
           let conf = Double(String(cm.output.1)) {
            d["confidence"] = max(0, min(1, conf))
        } else {
            d["confidence"] = isAnom ? 1.0 : 0.0
        }

        if let dm = text.firstMatch(of: /"description"\s*:\s*"([^"]*)"/),
           let desc = Optional(String(dm.output.1)) { d["description"] = desc }

        if let dtm = text.firstMatch(of: /"defect_type"\s*:\s*"([^"]*)"/),
           let dt = Optional(String(dtm.output.1)) { d["defect_type"] = dt }

        return d
    }
}
