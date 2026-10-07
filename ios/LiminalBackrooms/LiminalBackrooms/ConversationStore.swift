import Foundation

@MainActor
final class ConversationStore: ObservableObject {
    @Published var agents: [AgentProfile] = LiminalDefaults.agents
    @Published var messages: [ChatMessage] = []
    @Published var isRunning = false
    @Published var currentSpeaker: String?
    @Published var errorMessage: String?
    @Published var loadedSessionTimestamp: String?
    @Published var loadedSessionSource: String?
    @Published var turnLimit = 6
    @Published var seedPrompt = "A fluorescent hallway keeps repeating, but every door opens into a different model's memory."

    init() {
        loadBundledDesktopAutosave()
    }

    func reset() {
        messages.removeAll()
        errorMessage = nil
        currentSpeaker = nil
        loadedSessionTimestamp = nil
        loadedSessionSource = nil
    }

    func run(baseURL: String, apiKey: String) {
        guard !isRunning else { return }
        isRunning = true
        errorMessage = nil

        Task {
            await runConversation(baseURL: baseURL, apiKey: apiKey)
        }
    }

    func stop() {
        isRunning = false
        currentSpeaker = nil
    }

    var exportText: String {
        messages.map { message in
            let modelSuffix = message.model.map { " (\($0))" } ?? ""
            return "[\(message.speaker)\(modelSuffix)] \(message.content)"
        }
        .joined(separator: "\n\n")
    }

    func loadBundledDesktopAutosave() {
        guard let url = Bundle.main.url(
            forResource: ".autosave_conversation",
            withExtension: "json"
        ) else {
            return
        }

        do {
            let data = try Data(contentsOf: url)
            let autosave = try JSONDecoder().decode(DesktopAutosave.self, from: data)
            let importedMessages = autosave.conversation
                .map(\.chatMessage)
                .filter { !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

            guard !importedMessages.isEmpty else { return }

            messages = importedMessages
            loadedSessionTimestamp = autosave.timestamp
            loadedSessionSource = "Desktop autosave"
        } catch {
            errorMessage = "Could not load desktop autosave: \(error.localizedDescription)"
        }
    }

    private func runConversation(baseURL: String, apiKey: String) async {
        let client = OmniRouteClient(baseURL: baseURL, apiKey: apiKey)
        let activeAgents = agents.filter(\.isEnabled)

        guard activeAgents.count >= 2 else {
            errorMessage = "Enable at least two agents before starting."
            stop()
            return
        }

        if messages.isEmpty {
            messages.append(ChatMessage(role: .user, speaker: "Human", content: seedPrompt))
        }

        for turn in 0..<turnLimit {
            guard isRunning else { break }
            let agent = activeAgents[turn % activeAgents.count]
            currentSpeaker = agent.name

            do {
                let reply = try await client.complete(
                    modelID: agent.modelID,
                    systemPrompt: agent.persona,
                    conversation: messages
                )
                messages.append(ChatMessage(role: .assistant, speaker: agent.name, content: reply))
            } catch {
                errorMessage = error.localizedDescription
                break
            }
        }

        stop()
    }
}
