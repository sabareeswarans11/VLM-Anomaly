import SwiftUI
import SwiftData

// MARK: - BenchmarkRun model (stored in local SQLite via SwiftData)

@Model
final class BenchmarkRun {
    var timestamp: Date
    var modelName: String
    var device: String
    var firstTokenMs: Double
    var tokensPerSecond: Double
    var peakMemoryMB: Double
    var promptKey: String
    var imageDescription: String

    init(modelName: String, device: String, firstTokenMs: Double,
         tokensPerSecond: Double, peakMemoryMB: Double,
         promptKey: String, imageDescription: String) {
        self.timestamp = Date()
        self.modelName = modelName
        self.device = device
        self.firstTokenMs = firstTokenMs
        self.tokensPerSecond = tokensPerSecond
        self.peakMemoryMB = peakMemoryMB
        self.promptKey = promptKey
        self.imageDescription = imageDescription
    }
}

// MARK: - BenchmarkView

struct BenchmarkView: View {
    @Query(sort: \BenchmarkRun.timestamp, order: .reverse) private var runs: [BenchmarkRun]
    @Environment(\.modelContext) private var context

    var body: some View {
        NavigationStack {
            List {
                // Target row
                Section("Reference Target") {
                    targetRow(label: "iPhone 16 Pro Max",
                              model: "MiniCPM-V 4.6 Q4_K_M",
                              firstToken: 2000,
                              tokPerSec: 17.9,
                              peakMB: nil)
                }

                Section("Your Runs (\(runs.count))") {
                    if runs.isEmpty {
                        Text("Run an analysis in the Inspect tab to record a benchmark.")
                            .foregroundStyle(.secondary)
                            .font(.caption)
                    } else {
                        ForEach(runs) { run in
                            runRow(run)
                        }
                        .onDelete(perform: deleteRuns)
                    }
                }
            }
            .navigationTitle("Benchmark")
            .toolbar {
                EditButton()
            }
        }
    }

    private func targetRow(label: String, model: String,
                            firstToken: Double, tokPerSec: Double,
                            peakMB: Double?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).fontWeight(.semibold)
            Text(model).font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 16) {
                metric("First token", value: String(format: "%.0f ms", firstToken))
                metric("Throughput", value: String(format: "%.1f tok/s", tokPerSec))
                if let mb = peakMB {
                    metric("Peak RAM", value: String(format: "%.0f MB", mb))
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func runRow(_ run: BenchmarkRun) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(run.timestamp, style: .relative).fontWeight(.medium)
            Text(run.modelName).font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 16) {
                metric("First token", value: String(format: "%.0f ms", run.firstTokenMs))
                metric("Throughput", value: String(format: "%.1f tok/s", run.tokensPerSecond))
                metric("Peak RAM", value: String(format: "%.0f MB", run.peakMemoryMB))
            }
        }
        .padding(.vertical, 4)
    }

    private func metric(_ label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.caption).fontWeight(.semibold)
        }
    }

    private func deleteRuns(at offsets: IndexSet) {
        for idx in offsets { context.delete(runs[idx]) }
    }
}
