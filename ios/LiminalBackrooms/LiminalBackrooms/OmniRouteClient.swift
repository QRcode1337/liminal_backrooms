import Foundation

struct OmniRouteClient {
    var baseURL: String
    var apiKey: String

    func complete(modelID: String, systemPrompt: String, conversation: [ChatMessage]) async throws -> String {
        let trimmedBase = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmedBase.hasSuffix("/chat/completions")
            ? trimmedBase
            : trimmedBase.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/chat/completions"
        ) else {
            throw LiminalError.invalidBaseURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Liminal Backrooms iOS", forHTTPHeaderField: "X-Title")
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedKey.isEmpty {
            request.setValue("Bearer \(trimmedKey)", forHTTPHeaderField: "Authorization")
        }

        let messages = [OmniRouteMessage(role: "system", content: systemPrompt)]
            + conversation.suffix(20).map { message in
                OmniRouteMessage(
                    role: message.role == .assistant ? "assistant" : "user",
                    content: "\(message.speaker): \(message.content)"
                )
            }

        request.httpBody = try JSONEncoder().encode(OmniRouteRequest(model: modelID, messages: messages))

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw LiminalError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? "No response body"
            throw LiminalError.requestFailed(status: httpResponse.statusCode, body: body)
        }

        let decoded = try JSONDecoder().decode(OmniRouteResponse.self, from: data)
        guard let content = decoded.choices.first?.message.content, !content.isEmpty else {
            throw LiminalError.emptyResponse
        }
        return content
    }
}

enum LiminalError: LocalizedError {
    case invalidBaseURL
    case invalidResponse
    case emptyResponse
    case requestFailed(status: Int, body: String)

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL:
            "Set a valid OmniRoute base URL in Settings (default http://127.0.0.1:20128/v1)."
        case .invalidResponse:
            "OmniRoute returned an invalid response."
        case .emptyResponse:
            "The selected model returned an empty response."
        case .requestFailed(let status, let body):
            "OmniRoute request failed with HTTP \(status): \(body)"
        }
    }
}

private struct OmniRouteRequest: Encodable {
    let model: String
    let messages: [OmniRouteMessage]
    let temperature: Double = 1.0
}

private struct OmniRouteMessage: Codable {
    let role: String
    let content: String
}

private struct OmniRouteResponse: Decodable {
    struct Choice: Decodable {
        struct Message: Decodable {
            let content: String
        }

        let message: Message
    }

    let choices: [Choice]
}
