import Foundation
import CoreGraphics

struct Box: Hashable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double
}

struct FeaturePack {
    var geometry: [Double]
    var lbp: [Double]
    var hog: [Double]
    var color: [Double]
}

struct Quality {
    var size: Double
    var sharpness: Double
    var frontal: Double
    var score: Double
}

struct DetectedFace: Identifiable, Hashable {
    let id: UUID
    var mediaID: UUID
    var frameIndex: Int?
    var timeMs: Int?
    var box: Box
    var landmarks: [CGPoint]
    var features: FeaturePack
    var quality: Quality
    var crop: CGImage?

    static func == (lhs: DetectedFace, rhs: DetectedFace) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

struct MediaItem: Identifiable {
    let id: UUID
    var name: String
    var url: URL
    var isVideo: Bool
    var width: Int
    var height: Int
    var frameCount: Int
}

struct FrameItem: Identifiable {
    let id: UUID
    var mediaID: UUID
    var index: Int
    var timeMs: Int
    var image: CGImage
}

struct Identity: Identifiable {
    let id: UUID
    var name: String
    var faceIDs: [UUID]
    var prototype: FeaturePack
    var cover: CGImage?
}

struct StratumScore: Identifiable {
    var id: String { key }
    var key: String
    var label: String
    var percent: Double
    var detail: String
}

struct MatchRow: Identifiable {
    var id: String { "\(faceID)-\(identityID)" }
    var faceID: UUID
    var identityID: UUID
    var identityName: String
    var strata: [StratumScore]
    var ensemble: Double
    var temporal: Double?
    var decision: String
}

enum StrataMeta {
    static let all: [(key: String, label: String, weight: Double)] = [
        ("geometry", "Landmark-Geometrie", 0.28),
        ("lbp", "Textur (LBP)", 0.26),
        ("hog", "Gradient (HOG)", 0.26),
        ("color", "Farb-Signatur", 0.20),
    ]
}
