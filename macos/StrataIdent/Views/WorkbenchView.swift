import SwiftUI

struct WorkbenchView: View {
    @EnvironmentObject private var workspace: Workspace

    var body: some View {
        NavigationSplitView {
            mediaList
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 320)
        } content: {
            stage
                .navigationSplitViewColumnWidth(min: 420, ideal: 560)
        } detail: {
            detail
                .navigationSplitViewColumnWidth(min: 280, ideal: 320, max: 400)
        }
        .navigationTitle("Strata")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("Import") { workspace.importPanel() }
                    .keyboardShortcut("o", modifiers: [.command])
            }
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for provider in providers {
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    Task { @MainActor in
                        await workspace.ingest(urls: [url])
                    }
                }
            }
            return true
        }
    }

    private var mediaList: some View {
        List(selection: $workspace.selectedMediaID) {
            Section("Medien") {
                if workspace.media.isEmpty {
                    Text("Fotos und Videos hierher ziehen.")
                        .foregroundStyle(.secondary)
                }
                ForEach(workspace.media) { item in
                    HStack {
                        Text(item.name).lineLimit(1)
                        Spacer()
                        Text(item.isVideo ? "\(item.frameCount) Frames" : "Foto")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    .tag(item.id)
                }
            }
        }
        .safeAreaInset(edge: .bottom) {
            Text(workspace.status)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
        }
    }

    private var stage: some View {
        VStack(spacing: 12) {
            if let media = workspace.media.first(where: { $0.id == workspace.selectedMediaID }) ?? workspace.media.first {
                let frame = workspace.frames.first(where: { $0.id == workspace.selectedFrameID }) ?? workspace.frames.first(where: { $0.mediaID == media.id })
                let image = frame?.image ?? FrameExtractor.image(from: media.url)
                if let image {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                ScrollView(.horizontal) {
                    HStack(spacing: 8) {
                        ForEach(workspace.frames.filter { $0.mediaID == media.id }) { fr in
                            Image(decorative: fr.image, scale: 1)
                                .resizable()
                                .frame(width: 96, height: 54)
                                .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                                .overlay {
                                    if fr.id == workspace.selectedFrameID {
                                        RoundedRectangle(cornerRadius: 4).stroke(.primary, lineWidth: 1)
                                    }
                                }
                                .onTapGesture { workspace.selectedFrameID = fr.id }
                        }
                        ForEach(workspace.faces.filter { $0.mediaID == media.id }) { face in
                            if let crop = face.crop {
                                VStack(spacing: 4) {
                                    Image(decorative: crop, scale: 1)
                                        .resizable()
                                        .frame(width: 56, height: 56)
                                        .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                                    Text(percentLabel(face.id))
                                        .font(.caption2.monospacedDigit())
                                        .foregroundStyle(.secondary)
                                }
                                .onTapGesture { workspace.selectedFaceID = face.id }
                            }
                        }
                    }
                }
            } else {
                ContentUnavailableView(
                    "Strata Ident",
                    systemImage: "face.dashed",
                    description: Text("Fotos oder Videos importieren. Frames werden extrahiert, Gesichter in mehreren Straten verglichen.")
                )
            }
        }
        .padding()
    }

    private var detail: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Identitäten").font(.headline)
                HStack {
                    TextField("Name", text: $workspace.enrollName)
                    Button("Einschreiben", action: workspace.enrollSelected)
                        .disabled(workspace.selectedFaceID == nil)
                }
                ForEach(workspace.identities) { ident in
                    HStack {
                        Text(ident.name)
                        Spacer()
                        Button(role: .destructive) { workspace.deleteIdentity(ident.id) } label: {
                            Image(systemName: "xmark")
                        }
                        .buttonStyle(.borderless)
                    }
                }
                Divider()
                Text("Match").font(.headline)
                if let faceID = workspace.selectedFaceID, let rows = workspace.matches[faceID] {
                    ForEach(rows.prefix(4)) { row in
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(row.identityName)
                                Spacer()
                                Text("\(Int(row.ensemble.rounded()))%")
                                    .monospacedDigit()
                            }
                            ForEach(row.strata) { s in
                                HStack {
                                    Text(s.label).foregroundStyle(.secondary)
                                    Spacer()
                                    Text("\(Int(s.percent.rounded()))")
                                        .monospacedDigit()
                                        .foregroundStyle(.secondary)
                                }
                                .font(.caption)
                                ProgressView(value: s.percent / 100)
                            }
                        }
                        .padding(.bottom, 8)
                    }
                } else {
                    Text("Gesicht wählen und Person einschreiben, dann erscheinen die Prozentwerte je Strate.")
                        .foregroundStyle(.secondary)
                }
                Divider()
                Text("Die native App nutzt Vision. Für ArcFace (das Experiment gegen Apple Photos) den Python-Engine aus dem README verwenden.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding()
        }
    }

    private func percentLabel(_ faceID: UUID) -> String {
        guard let top = workspace.matches[faceID]?.first else { return "—" }
        return "\(Int(top.ensemble.rounded()))%"
    }
}
