import SwiftUI

@main
struct LiminalBackroomsApp: App {
    @StateObject private var store = ConversationStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
        }
    }
}
