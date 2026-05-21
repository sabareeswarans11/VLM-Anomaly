import SwiftUI

@main
struct VLMAnomalyEdgeApp: App {
    @StateObject private var modelStore = ModelStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelStore)
        }
    }
}

// MARK: - ContentView

struct ContentView: View {
    @EnvironmentObject var store: ModelStore

    var body: some View {
        TabView {
            CaptureView()
                .tabItem { Label("Inspect", systemImage: "camera") }

            BenchmarkView()
                .tabItem { Label("Benchmark", systemImage: "gauge") }
        }
        .task {
            await store.downloadModelIfNeeded()
        }
    }
}

// MARK: - ModelStore

@MainActor
final class ModelStore: ObservableObject {
    @Published var isModelReady = false
    @Published var downloadProgress: Double = 0
    @Published var statusMessage = "Checking model…"

    private let runner = MiniCPMRunner.shared

    func downloadModelIfNeeded() async {
        if runner.isModelLoaded {
            isModelReady = true
            statusMessage = "Model ready"
            return
        }
        statusMessage = "Downloading model (~2 GB)…"
        do {
            try await runner.downloadAndLoad { [weak self] progress in
                Task { @MainActor in
                    self?.downloadProgress = progress
                }
            }
            isModelReady = true
            statusMessage = "Model ready"
        } catch {
            statusMessage = "Download failed: \(error.localizedDescription)"
        }
    }
}
