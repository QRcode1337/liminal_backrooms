import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var store: ConversationStore
    @AppStorage("omniRouteBaseURL") private var baseURL = "http://127.0.0.1:20128/v1"
    @AppStorage("omniRouteAPIKey") private var apiKey = ""
    @State private var selectedTab: AppTab = .conversation

    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationStack {
                ConversationView()
            }
            .tabItem { Label("Run", systemImage: "terminal") }
            .tag(AppTab.conversation)

            NavigationStack {
                AgentsView()
            }
            .tabItem { Label("Agents", systemImage: "person.2") }
            .tag(AppTab.agents)

            NavigationStack {
                SettingsView(baseURL: $baseURL, apiKey: $apiKey)
            }
            .tabItem { Label("Settings", systemImage: "gearshape") }
            .tag(AppTab.settings)
        }
        .tint(.green)
        .environment(\.omniRouteBaseURL, baseURL)
        .environment(\.omniRouteAPIKey, apiKey)
    }
}

private enum AppTab {
    case conversation
    case agents
    case settings
}

private struct ConversationView: View {
    @EnvironmentObject private var store: ConversationStore

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        if store.messages.isEmpty {
                            SeedPromptEditor()
                        } else if store.loadedSessionSource != nil {
                            ImportedSessionHeader()
                        }

                        ForEach(store.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }

                        if let currentSpeaker = store.currentSpeaker {
                            HStack(spacing: 8) {
                                ProgressView()
                                Text("\(currentSpeaker) is thinking")
                                    .font(.footnote.monospaced())
                            }
                            .foregroundStyle(.secondary)
                            .padding(.vertical, 8)
                        }
                    }
                    .padding()
                }
                .onChange(of: store.messages.count) { _, _ in
                    guard let last = store.messages.last else { return }
                    withAnimation(.easeOut(duration: 0.25)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }

            ControlBar()
        }
        .background(.black)
        .navigationTitle("Liminal Backrooms")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if !store.exportText.isEmpty {
                    ShareLink(item: store.exportText) {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .accessibilityLabel("Export conversation")
                }
            }
        }
        .alert("Run failed", isPresented: Binding(
            get: { store.errorMessage != nil },
            set: { if !$0 { store.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { store.errorMessage = nil }
        } message: {
            Text(store.errorMessage ?? "")
        }
    }
}

private struct ImportedSessionHeader: View {
    @EnvironmentObject private var store: ConversationStore

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(store.loadedSessionSource ?? "Imported session")
                .font(.caption.monospaced().weight(.semibold))
                .foregroundStyle(.green)

            if let loadedSessionTimestamp = store.loadedSessionTimestamp {
                Text(loadedSessionTimestamp)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }

            Text("\(store.messages.count) messages loaded from the desktop autosave format.")
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.green.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct SeedPromptEditor: View {
    @EnvironmentObject private var store: ConversationStore

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Seed")
                .font(.caption.monospaced())
                .foregroundStyle(.green)
            TextEditor(text: $store.seedPrompt)
                .font(.callout.monospaced())
                .foregroundStyle(.primary)
                .frame(minHeight: 96)
                .scrollContentBackground(.hidden)
                .padding(10)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
        }
    }
}

private struct ControlBar: View {
    @EnvironmentObject private var store: ConversationStore
    @Environment(\.omniRouteBaseURL) private var baseURL
    @Environment(\.omniRouteAPIKey) private var apiKey

    var body: some View {
        VStack(spacing: 12) {
            Stepper("Turns: \(store.turnLimit)", value: $store.turnLimit, in: 2...30)
                .font(.callout.monospaced())

            HStack {
                Button {
                    store.reset()
                } label: {
                    Label("Reset", systemImage: "arrow.counterclockwise")
                }
                .buttonStyle(.bordered)
                .disabled(store.isRunning)

                Spacer()

                Button {
                    store.isRunning ? store.stop() : store.run(baseURL: baseURL, apiKey: apiKey)
                } label: {
                    Label(store.isRunning ? "Stop" : "Propagate", systemImage: store.isRunning ? "stop.fill" : "play.fill")
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .background(.regularMaterial)
    }
}

private struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text(message.speaker)
                    .font(.caption.monospaced().weight(.semibold))
                    .foregroundStyle(.green)
                if let model = message.model {
                    Text(model)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                Text(message.date, style: .time)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }

            Text(message.content)
                .font(.body.monospaced())
                .textSelection(.enabled)
                .foregroundStyle(.primary)

            if let desktopType = message.desktopType {
                Text(desktopType)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(backgroundColor)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var backgroundColor: Color {
        switch message.role {
        case .user:
            return Color.green.opacity(0.10)
        case .system:
            return Color.yellow.opacity(0.10)
        case .assistant:
            return Color(uiColor: .secondarySystemBackground)
        }
    }
}

private struct AgentsView: View {
    @EnvironmentObject private var store: ConversationStore

    var body: some View {
        List {
            ForEach($store.agents) { $agent in
                Section {
                    Toggle("Enabled", isOn: $agent.isEnabled)

                    TextField("Name", text: $agent.name)

                    Picker("Model", selection: $agent.modelID) {
                        ForEach(LiminalDefaults.models) { model in
                            Text(model.label).tag(model.modelID)
                        }
                    }

                    TextEditor(text: $agent.persona)
                        .font(.callout.monospaced())
                        .frame(minHeight: 140)
                } header: {
                    Text(agent.name)
                }
            }

            Button {
                store.agents.append(AgentProfile(
                    name: "AI-\(store.agents.count + 1)",
                    modelID: LiminalDefaults.models.first?.modelID ?? "anthropic/claude-sonnet-4-6",
                    persona: "You are a new participant in a liminal AI conversation. Add a distinct perspective without derailing the thread."
                ))
            } label: {
                Label("Add Agent", systemImage: "plus")
            }
        }
        .navigationTitle("Agents")
    }
}

private struct SettingsView: View {
    @Binding var baseURL: String
    @Binding var apiKey: String

    var body: some View {
        Form {
            Section("OmniRoute") {
                TextField("Base URL", text: $baseURL)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)
                SecureField("API key (optional for local)", text: $apiKey)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Text("Simulator can use http://127.0.0.1:20128/v1. A physical device needs your Mac's LAN IP. Model IDs must be live OmniRoute IDs.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Section("About") {
                LabeledContent("Version", value: "0.1.0")
                Text("This iOS target is a native SwiftUI port of the core conversation loop. Text models route through OmniRoute, matching the desktop app.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Settings")
    }
}

private struct OmniRouteBaseURLKey: EnvironmentKey {
    static let defaultValue = "http://127.0.0.1:20128/v1"
}

private struct OmniRouteAPIKeyKey: EnvironmentKey {
    static let defaultValue = ""
}

private extension EnvironmentValues {
    var omniRouteBaseURL: String {
        get { self[OmniRouteBaseURLKey.self] }
        set { self[OmniRouteBaseURLKey.self] = newValue }
    }

    var omniRouteAPIKey: String {
        get { self[OmniRouteAPIKeyKey.self] }
        set { self[OmniRouteAPIKeyKey.self] = newValue }
    }
}

#Preview {
    ContentView()
        .environmentObject(ConversationStore())
}
