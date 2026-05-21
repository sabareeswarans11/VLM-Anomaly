import SwiftUI

struct AnomalyResult {
    let isAnomalous: Bool
    let confidence: Double
    let description: String
    let defectType: String?
    let latencyMs: Double
    let tokensPerSecond: Double
    let parseError: Bool
}

struct ResultView: View {
    let result: AnomalyResult

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                verdictBadge
                Spacer()
                Text(String(format: "%.0f ms · %.1f tok/s",
                            result.latencyMs, result.tokensPerSecond))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if result.parseError {
                Label("Could not parse model response", systemImage: "exclamationmark.circle")
                    .foregroundStyle(.orange)
                    .font(.caption)
            } else {
                Text(result.description.isEmpty ? "No description." : result.description)
                    .font(.body)

                if let defect = result.defectType {
                    Label(defect, systemImage: "tag")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(result.isAnomalous
                      ? Color.red.opacity(0.08)
                      : Color.green.opacity(0.08))
        )
        .padding(.horizontal)
    }

    private var verdictBadge: some View {
        HStack(spacing: 6) {
            Image(systemName: result.isAnomalous
                  ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
            Text(result.isAnomalous ? "Anomaly Detected" : "Normal")
                .fontWeight(.semibold)
            Text(String(format: "(%.0f%%)", result.confidence * 100))
                .foregroundStyle(.secondary)
        }
        .foregroundStyle(result.isAnomalous ? .red : .green)
    }
}
