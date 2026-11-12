import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = SonarViewModel()

    @AppStorage("serverAddress") private var serverAddress = AppConfig.defaultServerAddress
    @AppStorage("chirpBand") private var band = ChirpBand.high
    @AppStorage("recordingLabel") private var label = RecordingLabel.still
    @AppStorage("measurementMode") private var prefersMeasurementMode = true

    private var settings: RecordingSettings {
        RecordingSettings(serverAddress: serverAddress, band: band, label: label,
                          prefersMeasurementMode: prefersMeasurementMode)
    }

    var body: some View {
        NavigationStack {
            List {
                controlSection
                if viewModel.phase == .idle {
                    Section {
                        Picker("Label", selection: $label) {
                            ForEach(RecordingLabel.allCases) { option in
                                Text(option.title).tag(option)
                            }
                        }
                        .pickerStyle(.navigationLink)
                    }
                }
                if let status = viewModel.status {
                    signalSection(status)
                }
                if let info = viewModel.sessionInfo {
                    hardwareSection(info)
                }
                uploadSection
                Section {
                    NavigationLink("Settings") {
                        SettingsView(viewModel: viewModel, serverAddress: $serverAddress, band: $band,
                                     prefersMeasurementMode: $prefersMeasurementMode)
                    }
                    NavigationLink("Volume") {
                        VolumeView()
                    }
                }
            }
            .navigationTitle("Sonar")
        }
    }

    private var controlSection: some View {
        Section {
            switch viewModel.phase {
            case .idle:
                Button("Start recording") { viewModel.start(settings: settings) }
                    .tint(.green)
            case .recording:
                Button("Stop") { viewModel.stop() }
                    .tint(.red)
            case .uploading:
                ProgressView()
            case .uploadFailed:
                Button("Retry upload") { viewModel.retryUpload() }
                Button("Discard", role: .destructive) { viewModel.discardUpload() }
            }
            Text(viewModel.headline)
                .font(.footnote)
        }
    }

    private func signalSection(_ status: SonarEngine.Status) -> some View {
        Section("Signal") {
            LabeledContent("Time", value: String(format: "%.0f s", status.elapsedSeconds))
            if let gain = status.chirpGain {
                LabeledContent("Chirp", value: chirpVerdict(gain))
                LabeledContent("Chirp gain", value: String(format: "%+.0f dB", gain))
            } else if status.bandLevel == nil {
                LabeledContent("Chirp", value: "Mic rate too low")
            } else {
                LabeledContent("Chirp", value: "Measuring noise")
            }
            if let level = status.bandLevel {
                LabeledContent("Band level", value: String(format: "%.0f dB", level))
            }
            if let baseline = status.baselineBandLevel {
                LabeledContent("Band noise", value: String(format: "%.0f dB", baseline))
            }
            LabeledContent("Mic level", value: String(format: "%.0f dB", status.totalLevel))
        }
    }

    private func hardwareSection(_ info: SonarEngine.SessionInfo) -> some View {
        Section("Hardware") {
            LabeledContent("Mic rate", value: String(format: "%.0f Hz", info.inputSampleRate))
            LabeledContent("Speaker rate", value: String(format: "%.0f Hz", info.outputSampleRate))
            LabeledContent("Audio mode", value: info.audioSessionMode)
            LabeledContent("Volume", value: String(format: "%.0f %%", info.outputVolume * 100))
        }
    }

    private var uploadSection: some View {
        Section("Upload") {
            LabeledContent("Sent", value: "\(viewModel.upload.sentChunks) s")
            LabeledContent("Waiting", value: "\(viewModel.upload.pendingRequests)")
            if let error = viewModel.upload.lastError {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }
        }
    }

    /// Same thresholds as the analyzer's in-band signal-to-noise check.
    private func chirpVerdict(_ gain: Double) -> String {
        if gain >= 20 { return "Detected" }
        if gain >= 10 { return "Weak" }
        return "Not detected"
    }
}

struct SettingsView: View {
    @ObservedObject var viewModel: SonarViewModel
    @Binding var serverAddress: String
    @Binding var band: ChirpBand
    @Binding var prefersMeasurementMode: Bool

    var body: some View {
        List {
            Section("Receiver address") {
                TextField("host:port or https URL", text: $serverAddress)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Button("Test connection") {
                    viewModel.testConnection(address: serverAddress,
                                             prefersMeasurementMode: prefersMeasurementMode)
                }
                if !viewModel.connectionResult.isEmpty {
                    Text(viewModel.connectionResult)
                        .font(.footnote)
                }
            }
            Section("Chirp") {
                Picker("Band", selection: $band) {
                    ForEach(ChirpBand.allCases) { option in
                        Text(option.title).tag(option)
                    }
                }
                .pickerStyle(.navigationLink)
                Toggle("Measurement mode", isOn: $prefersMeasurementMode)
            }
        }
        .navigationTitle("Settings")
    }
}
