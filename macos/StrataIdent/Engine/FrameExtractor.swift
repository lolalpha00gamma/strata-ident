import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum FrameExtractor {
    static func image(from url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldCache: true] as CFDictionary)
    }

    static func frames(from video: URL, fps: Int, maxFrames: Int) async throws -> [(index: Int, timeMs: Int, image: CGImage)] {
        let asset = AVURLAsset(url: video)
        let duration = try await asset.load(.duration)
        let seconds = max(0.01, CMTimeGetSeconds(duration))
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = .zero
        gen.maximumSize = CGSize(width: 1600, height: 1600)

        let interval = 1.0 / Double(max(1, fps))
        var times: [CMTime] = []
        var t = 0.0
        while t < seconds && times.count < maxFrames {
            times.append(CMTime(seconds: t, preferredTimescale: 600))
            t += interval
        }
        if times.isEmpty { times = [.zero] }

        var out: [(Int, Int, CGImage)] = []
        for (i, time) in times.enumerated() {
            let image = try gen.copyCGImage(at: time, actualTime: nil)
            out.append((i, Int(CMTimeGetSeconds(time) * 1000), image))
        }
        return out
    }
}
