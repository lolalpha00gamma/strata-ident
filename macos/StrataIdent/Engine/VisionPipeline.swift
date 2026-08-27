import CoreGraphics
import Vision

enum VisionPipeline {
    static func detect(in image: CGImage) throws -> [(box: Box, landmarks: [CGPoint], quality: Double)] {
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        let faceReq = VNDetectFaceLandmarksRequest()
        faceReq.revision = VNDetectFaceLandmarksRequestRevision3
        let qualityReq = VNDetectFaceCaptureQualityRequest()
        try handler.perform([faceReq, qualityReq])

        let qualities = (qualityReq.results ?? []).compactMap { obs -> (CGRect, Double)? in
            guard let q = obs.faceCaptureQuality else { return nil }
            return (obs.boundingBox, Double(q))
        }

        var out: [(Box, [CGPoint], Double)] = []
        for obs in faceReq.results ?? [] {
            let bb = obs.boundingBox
            let box = Box(
                x: bb.minX * Double(image.width),
                y: (1 - bb.minY - bb.height) * Double(image.height),
                width: bb.width * Double(image.width),
                height: bb.height * Double(image.height)
            )
            var pts: [CGPoint] = []
            if let lm = obs.landmarks {
                let regions = [
                    lm.faceContour, lm.leftEye, lm.rightEye, lm.nose, lm.outerLips,
                    lm.leftEyebrow, lm.rightEyebrow, lm.medianLine, lm.innerLips,
                ]
                for region in regions {
                    guard let region else { continue }
                    for p in region.normalizedPoints {
                        let x = (bb.minX + Double(p.x) * bb.width) * Double(image.width)
                        let y = (1 - (bb.minY + Double(p.y) * bb.height)) * Double(image.height)
                        pts.append(CGPoint(x: x, y: y))
                    }
                }
            }
            let q = qualities.min(by: { hypot($0.0.midX - bb.midX, $0.0.midY - bb.midY) < hypot($1.0.midX - bb.midX, $1.0.midY - bb.midY) })?.1 ?? 0.5
            out.append((box, pts, q))
        }
        return out
    }

    static func crop(_ image: CGImage, box: Box, size: Int = 112) -> CGImage? {
        let padW = box.width * 0.22
        let padH = box.height * 0.22
        let rect = CGRect(
            x: max(0, box.x - padW),
            y: max(0, box.y - padH),
            width: min(Double(image.width), box.width + padW * 2),
            height: min(Double(image.height), box.height + padH * 2)
        )
        guard let sliced = image.cropping(to: rect) else { return image }
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return sliced }
        ctx.interpolationQuality = .high
        ctx.draw(sliced, in: CGRect(x: 0, y: 0, width: size, height: size))
        return ctx.makeImage()
    }
}
