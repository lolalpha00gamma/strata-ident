import SwiftUI

@main
struct StrataIdentApp: App {
    @StateObject private var workspace = Workspace()

    var body: some Scene {
        WindowGroup {
            WorkbenchView()
                .environmentObject(workspace)
                .frame(minWidth: 980, minHeight: 640)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Medien importieren…") {
                    workspace.importPanel()
                }
                .keyboardShortcut("o", modifiers: [.command])
            }
        }
        Settings {
            Form {
                Picker("Video-FPS", selection: $workspace.fps) {
                    ForEach([1, 2, 3, 4, 6], id: \.self) { Text("\($0)").tag($0) }
                }
                Stepper("Max. Frames: \(workspace.maxFrames)", value: $workspace.maxFrames, in: 8...120)
            }
            .formStyle(.grouped)
            .frame(width: 360)
        }
    }
}
