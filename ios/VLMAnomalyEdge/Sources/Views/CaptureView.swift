import SwiftUI
import PhotosUI

struct CaptureView: View {
    @EnvironmentObject var store: ModelStore
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var prediction: AnomalyResult?
    @State private var isInferring = false
    @State private var promptKey = "generic.detailed"

    private let promptKeys = ["generic.simple", "generic.detailed", "generic.cot",
                               "manufacturing.simple", "manufacturing.detailed",
                               "infrastructure.simple", "infrastructure.detailed"]

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                // Model status banner
                if !store.isModelReady {
                    ProgressView(value: store.downloadProgress) {
                        Text(store.statusMessage).font(.caption)
                    }
                    .padding(.horizontal)
                }

                // Image preview
                imagePreview

                // Controls
                Picker("Prompt", selection: $promptKey) {
                    ForEach(promptKeys, id: \.self) { Text($0).tag($0) }
                }
                .pickerStyle(.menu)
                .padding(.horizontal)

                HStack(spacing: 12) {
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        Label("Choose Photo", systemImage: "photo")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)

                    Button(action: runInference) {
                        Label(isInferring ? "Analyzing…" : "Analyze",
                              systemImage: "wand.and.stars")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!store.isModelReady || selectedImage == nil || isInferring)
                }
                .padding(.horizontal)

                // Result
                if let pred = prediction {
                    ResultView(result: pred)
                }

                Spacer()
            }
            .navigationTitle("VLM Anomaly Inspect")
            .onChange(of: selectedItem) { _, item in
                Task { await loadImage(from: item) }
            }
        }
    }

    @ViewBuilder
    private var imagePreview: some View {
        if let img = selectedImage {
            Image(uiImage: img)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 280)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal)
                .overlay(alignment: .topTrailing) {
                    if let pred = prediction {
                        Image(systemName: pred.isAnomalous ? "exclamationmark.triangle.fill"
                                                           : "checkmark.circle.fill")
                            .foregroundStyle(pred.isAnomalous ? .red : .green)
                            .font(.largeTitle)
                            .padding(8)
                    }
                }
        } else {
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
                .frame(height: 200)
                .overlay { Label("No image selected", systemImage: "photo") }
                .padding(.horizontal)
        }
    }

    private func loadImage(from item: PhotosPickerItem?) async {
        guard let item,
              let data = try? await item.loadTransferable(type: Data.self),
              let uiImage = UIImage(data: data) else { return }
        selectedImage = uiImage
        prediction = nil
    }

    private func runInference() {
        guard let img = selectedImage else { return }
        isInferring = true
        prediction = nil

        Task {
            let prompt = PromptBuilder.shared.render(key: promptKey)
            let result = await MiniCPMRunner.shared.predict(image: img, prompt: prompt)
            prediction = result
            isInferring = false
        }
    }
}
