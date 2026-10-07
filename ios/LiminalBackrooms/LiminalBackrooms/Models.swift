import Foundation

struct ChatMessage: Identifiable, Codable, Hashable {
    enum Role: String, Codable {
        case system
        case user
        case assistant
    }

    var id = UUID()
    var role: Role
    var speaker: String
    var content: String
    var date = Date()
    var model: String?
    var desktopType: String?
}

struct DesktopAutosave: Decodable {
    let timestamp: String
    let conversation: [DesktopAutosaveMessage]
}

struct DesktopAutosaveMessage: Decodable {
    let role: String
    let aiName: String
    let model: String
    let type: String
    let content: String

    enum CodingKeys: String, CodingKey {
        case role
        case aiName = "ai_name"
        case model
        case type = "_type"
        case content
    }

    var chatMessage: ChatMessage {
        let messageRole = ChatMessage.Role(rawValue: role) ?? .system
        let speaker: String

        switch messageRole {
        case .user:
            speaker = "Human"
        case .assistant:
            speaker = aiName.isEmpty ? "AI" : aiName
        case .system:
            speaker = "System"
        }

        return ChatMessage(
            role: messageRole,
            speaker: speaker,
            content: content,
            model: model.isEmpty ? nil : model,
            desktopType: type.isEmpty ? nil : type
        )
    }
}

struct AgentProfile: Identifiable, Codable, Hashable {
    var id = UUID()
    var name: String
    var modelID: String
    var persona: String
    var isEnabled: Bool = true
}

struct ModelOption: Identifiable, Hashable {
    var id: String { modelID }
    let label: String
    let modelID: String
}

enum LiminalDefaults {
    static let models: [ModelOption] = [
        ModelOption(label: "Claude Sonnet 4.6", modelID: "anthropic/claude-sonnet-4-6"),
        ModelOption(label: "Claude Opus 4.6", modelID: "anthropic/claude-opus-4-6"),
        ModelOption(label: "Claude Haiku 4.5", modelID: "anthropic/claude-haiku-4-5-20251001"),
        ModelOption(label: "GPT 5.5", modelID: "openai/gpt-5.5"),
        ModelOption(label: "GPT 5.2", modelID: "openai/gpt-5.2"),
        ModelOption(label: "GPT 5.6 Sol (High)", modelID: "cx/gpt-5.6-sol-high"),
        ModelOption(label: "Gemini 3.5 Flash", modelID: "agy/gemini-3.5-flash-medium"),
        ModelOption(label: "Gemini 3.1 Pro", modelID: "gemini/gemini-3.1-pro-preview"),
        ModelOption(label: "Grok 4.6", modelID: "xai/grok-4.6"),
        ModelOption(label: "Kimi K2.6", modelID: "nvidia/moonshotai/kimi-k2.6")
    ]

    static let agents: [AgentProfile] = [
        AgentProfile(
            name: "AI-1",
            modelID: "anthropic/claude-sonnet-4-6",
            persona: "You are an uncanny but useful participant in a liminal multi-agent conversation. Be vivid, concise, and responsive."
        ),
        AgentProfile(
            name: "AI-2",
            modelID: "openai/gpt-5.2",
            persona: "You are a sharp counter-voice in a liminal multi-agent conversation. Build on the thread, challenge weak ideas, and keep the exchange alive."
        )
    ]
}
