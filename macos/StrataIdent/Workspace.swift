import AppKit
import Foundation
import UniformTypeIdentifiers

@MainActor
final class Workspace: ObservableObject {
    @Published var media: [MediaItem] = []
    @Published var frames: [FrameItem] = []
    @Published var faces: [DetectedFace] = []
    @Published var identities: [Identity] = []
    @Published var matches: [UUID: [MatchRow]] = [:]
    @Published var selectedMediaID: UUID?
    @Published var selectedFaceID: UUID?
    @Published var selectedFrameID: UUID?
    @Published var enrollName = ""
    @Published var status = "Bereit"
    @Published var fps = 2
    @Published var maxFrames = 40
    @Published var busy = false

    func importPanel() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = true
        panel.allowedContentTypes = [.image, .movie, .mpeg4Movie, .quickTimeMovie, .png, .jpeg, .heic, .webP]
        panel.begin { [weak self] result in
            guard result == .OK else { return }
            Task { @MainActor in
                await self?.ingest(urls: panel.urls)
            }
        }
    }

    func ingest(urls: [URL]) async {
        busy = true
        defer { busy = false }
        for url in urls {
            if url.hasDirectoryPath {
                let kids = (try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? []
                await ingest(urls: kids)
                continue
            }
            let ext = url.pathExtension.lowercased()
            let videoExt = ["mp4", "mov", "m4v", "avi", "mkv", "webm"]
            if videoExt.contains(ext) {
                await ingestVideo(url)
            } else {
                await ingestImage(url)
            }
        }
        recompute()
    }

    private func ingestImage(_ url: URL) async {
        status = "Foto \(url.lastPathComponent)"
        guard let image = FrameExtractor.image(from: url) else { return }
        let item = MediaItem(id: UUID(), name: url.lastPathComponent, url: url, isVideo: false, width: image.width, height: image.height, frameCount: 1)
        media.append(item)
        selectedMediaID = item.id
        await detect(image: image, mediaID: item.id, frameIndex: nil, timeMs: nil)
    }

    private func ingestVideo(_ url: URL) async {
        status = "Video \(url.lastPathComponent)"
        do {
            let extracted = try await FrameExtractor.frames(from: url, fps: fps, maxFrames: maxFrames)
            guard let first = extracted.first else { return }
            let item = MediaItem(id: UUID(), name: url.lastPathComponent, url: url, isVideo: true, width: first.image.width, height: first.image.height, frameCount: extracted.count)
            media.append(item)
            selectedMediaID = item.id
            for (index, timeMs, image) in extracted {
                let frame = FrameItem(id: UUID(), mediaID: item.id, index: index, timeMs: timeMs, image: image)
                frames.append(frame)
                if selectedFrameID == nil { selectedFrameID = frame.id }
                status = "Frame \(index + 1)/\(extracted.count)"
                await detect(image: image, mediaID: item.id, frameIndex: index, timeMs: timeMs)
            }
        } catch {
            status = error.localizedDescription
        }
    }

    private func detect(image: CGImage, mediaID: UUID, frameIndex: Int?, timeMs: Int?) async {
        do {
            let found = try VisionPipeline.detect(in: image)
            for (box, landmarks, q) in found {
                guard let crop = VisionPipeline.crop(image, box: box) else { continue }
                let packed = Strategies.pack(from: crop, landmarks: landmarks)
                var quality = packed.1
                quality.frontal = q
                quality.score = 0.4 * packed.1.sharpness + 0.3 * q + 0.3 * min(1, packed.1.size / 160)
                let face = DetectedFace(
                    id: UUID(),
                    mediaID: mediaID,
                    frameIndex: frameIndex,
                    timeMs: timeMs,
                    box: box,
                    landmarks: landmarks,
                    features: packed.0,
                    quality: quality,
                    crop: crop
                )
                faces.append(face)
                selectedFaceID = face.id
            }
        } catch {
            status = error.localizedDescription
        }
    }

    func enrollSelected() {
        guard let id = selectedFaceID, let face = faces.first(where: { $0.id == id }) else { return }
        let name = enrollName.trimmingCharacters(in: .whitespacesAndNewlines)
        let label = name.isEmpty ? "Person \(identities.count + 1)" : name
        if let idx = identities.firstIndex(where: { $0.name.compare(label, options: .caseInsensitive) == .orderedSame }) {
            identities[idx].faceIDs.append(face.id)
            let pack = identities[idx].faceIDs.compactMap { fid in faces.first(where: { $0.id == fid })?.features }
            identities[idx].prototype = FeaturePack(
                geometry: Strategies.mean(pack.map(\.geometry)),
                lbp: Strategies.mean(pack.map(\.lbp)),
                hog: Strategies.mean(pack.map(\.hog)),
                color: Strategies.mean(pack.map(\.color))
            )
        } else {
            identities.append(Identity(id: UUID(), name: label, faceIDs: [face.id], prototype: face.features, cover: face.crop))
        }
        enrollName = ""
        recompute()
    }

    func deleteIdentity(_ id: UUID) {
        identities.removeAll { $0.id == id }
        recompute()
    }

    func recompute() {
        var map: [UUID: [MatchRow]] = [:]
        for face in faces {
            var rows: [MatchRow] = []
            for ident in identities {
                let scored = Strategies.score(probe: face.features, proto: ident.prototype, quality: face.quality.score)
                let decision: String
                if scored.ensemble >= 72 { decision = "match" }
                else if scored.ensemble >= 52 { decision = "possible" }
                else { decision = "reject" }
                rows.append(MatchRow(faceID: face.id, identityID: ident.id, identityName: ident.name, strata: scored.strata, ensemble: scored.ensemble, temporal: nil, decision: decision))
            }
            rows.sort { $0.ensemble > $1.ensemble }
            map[face.id] = rows
        }
        matches = map
        status = "\(faces.count) Gesichter · \(identities.count) Personen"
    }
}
