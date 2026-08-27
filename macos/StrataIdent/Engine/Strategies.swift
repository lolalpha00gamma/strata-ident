import Foundation
import CoreGraphics

enum Strategies {
    static func geometry(from points: [CGPoint]) -> [Double] {
        guard !points.isEmpty else { return [] }
        let cx = points.map(\.x).reduce(0, +) / Double(points.count)
        let cy = points.map(\.y).reduce(0, +) / Double(points.count)
        let minX = points.map(\.x).min() ?? 0
        let maxX = points.map(\.x).max() ?? 1
        let minY = points.map(\.y).min() ?? 0
        let maxY = points.map(\.y).max() ?? 1
        let scale = max(hypot(maxX - minX, maxY - minY), 1)
        return points.flatMap { [($0.x - cx) / scale, ($0.y - cy) / scale] }
    }

    static func lbp(_ pixels: [UInt8], width: Int, height: Int) -> [Double] {
        var hist = [Double](repeating: 0, count: 256)
        guard width > 2, height > 2 else { return hist }
        func gray(_ x: Int, _ y: Int) -> Int {
            let i = (y * width + x) * 4
            let r = Int(pixels[i])
            let g = Int(pixels[i + 1])
            let b = Int(pixels[i + 2])
            return (r * 299 + g * 587 + b * 114) / 1000
        }
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let c = gray(x, y)
                var code = 0
                code |= (gray(x - 1, y - 1) >= c ? 1 : 0) << 7
                code |= (gray(x, y - 1) >= c ? 1 : 0) << 6
                code |= (gray(x + 1, y - 1) >= c ? 1 : 0) << 5
                code |= (gray(x + 1, y) >= c ? 1 : 0) << 4
                code |= (gray(x + 1, y + 1) >= c ? 1 : 0) << 3
                code |= (gray(x, y + 1) >= c ? 1 : 0) << 2
                code |= (gray(x - 1, y + 1) >= c ? 1 : 0) << 1
                code |= (gray(x - 1, y) >= c ? 1 : 0)
                hist[code] += 1
            }
        }
        let sum = hist.reduce(0, +)
        if sum > 0 { for i in 0..<hist.count { hist[i] /= sum } }
        return hist
    }

    static func hog(_ pixels: [UInt8], width: Int, height: Int, cells: Int = 8, bins: Int = 8) -> [Double] {
        func gray(_ x: Int, _ y: Int) -> Double {
            let i = (y * width + x) * 4
            return Double(pixels[i]) * 0.299 + Double(pixels[i + 1]) * 0.587 + Double(pixels[i + 2]) * 0.114
        }
        var out = [Double](repeating: 0, count: cells * cells * bins)
        let cellW = max(1, width / cells)
        let cellH = max(1, height / cells)
        for cy in 0..<cells {
            for cx in 0..<cells {
                let base = (cy * cells + cx) * bins
                let y0 = cy * cellH
                let x0 = cx * cellW
                for y in max(1, y0)..<min(height - 1, y0 + cellH) {
                    for x in max(1, x0)..<min(width - 1, x0 + cellW) {
                        let dx = gray(x + 1, y) - gray(x - 1, y)
                        let dy = gray(x, y + 1) - gray(x, y - 1)
                        let mag = hypot(dx, dy)
                        var ang = atan2(dy, dx)
                        if ang < 0 { ang += 2 * .pi }
                        let b = min(bins - 1, Int(ang / (2 * .pi) * Double(bins)))
                        out[base + b] += mag
                    }
                }
            }
        }
        let n = sqrt(out.reduce(0) { $0 + $1 * $1 })
        if n > 0 { for i in 0..<out.count { out[i] /= n } }
        return out
    }

    static func color(_ pixels: [UInt8], count: Int) -> [Double] {
        var hist = [Double](repeating: 0, count: 8 * 4 * 4)
        for i in stride(from: 0, to: count * 4, by: 4) {
            let r = Double(pixels[i]) / 255
            let g = Double(pixels[i + 1]) / 255
            let b = Double(pixels[i + 2]) / 255
            let maxv = max(r, g, b)
            let minv = min(r, g, b)
            let d = maxv - minv
            var h = 0.0
            if d > 1e-6 {
                if maxv == r { h = ((g - b) / d).truncatingRemainder(dividingBy: 6) }
                else if maxv == g { h = (b - r) / d + 2 }
                else { h = (r - g) / d + 4 }
                h /= 6
                if h < 0 { h += 1 }
            }
            let s = maxv == 0 ? 0 : d / maxv
            let hi = min(7, Int(h * 8))
            let si = min(3, Int(s * 4))
            let vi = min(3, Int(maxv * 4))
            hist[hi * 16 + si * 4 + vi] += 1
        }
        let sum = hist.reduce(0, +)
        if sum > 0 { for i in 0..<hist.count { hist[i] /= sum } }
        return hist
    }

    static func cosine(_ a: [Double], _ b: [Double]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<n {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        if na == 0 || nb == 0 { return 0 }
        return dot / (sqrt(na) * sqrt(nb))
    }

    static func chi2(_ a: [Double], _ b: [Double]) -> Double {
        let n = min(a.count, b.count)
        var s = 0.0
        for i in 0..<n {
            let m = a[i] + b[i]
            if m > 1e-9 {
                let d = a[i] - b[i]
                s += (d * d) / m
            }
        }
        return s
    }

    static func distPercent(_ dist: Double, good: Double, bad: Double) -> Double {
        if bad <= good { return dist <= good ? 100 : 0 }
        return max(0, min(100, 100 * (1 - (dist - good) / (bad - good))))
    }

    static func mean(_ list: [[Double]]) -> [Double] {
        guard let first = list.first else { return [] }
        var out = [Double](repeating: 0, count: first.count)
        for v in list {
            for i in 0..<min(out.count, v.count) { out[i] += v[i] }
        }
        for i in 0..<out.count { out[i] /= Double(list.count) }
        return out
    }

    static func pack(from image: CGImage, landmarks: [CGPoint]) -> (FeaturePack, Quality, [UInt8]) {
        let w = image.width
        let h = image.height
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: &pixels,
            width: w,
            height: h,
            bitsPerComponent: 8,
            bytesPerRow: w * 4,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        ctx?.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        let geo = geometry(from: landmarks)
        let lbpV = lbp(pixels, width: w, height: h)
        let hogV = hog(pixels, width: w, height: h)
        let col = color(pixels, count: w * h)
        var lap = 0.0
        var n = 0.0
        func gray(_ x: Int, _ y: Int) -> Double {
            let i = (y * w + x) * 4
            return Double(pixels[i]) * 0.299 + Double(pixels[i + 1]) * 0.587 + Double(pixels[i + 2]) * 0.114
        }
        if w > 2 && h > 2 {
            for y in stride(from: 1, to: h - 1, by: 2) {
                for x in stride(from: 1, to: w - 1, by: 2) {
                    let v = gray(x + 1, y) + gray(x - 1, y) + gray(x, y + 1) + gray(x, y - 1) - 4 * gray(x, y)
                    lap += v * v
                    n += 1
                }
            }
        }
        let sharpness = min(1, sqrt(lap / max(n, 1)) / 40)
        let sizeN = min(1, sqrt(Double(w * h)) / 160)
        let quality = Quality(size: sqrt(Double(w * h)), sharpness: sharpness, frontal: 0.7, score: 0.5 * sharpness + 0.5 * sizeN)
        return (FeaturePack(geometry: geo, lbp: lbpV, hog: hogV, color: col), quality, pixels)
    }

    static func score(probe: FeaturePack, proto: FeaturePack, quality: Double) -> (strata: [StratumScore], ensemble: Double) {
        let geo = cosine(probe.geometry, proto.geometry)
        let lbpD = chi2(probe.lbp, proto.lbp)
        let hog = cosine(probe.hog, proto.hog)
        let col = chi2(probe.color, proto.color)
        let strata = [
            StratumScore(key: "geometry", label: "Landmark-Geometrie", percent: distPercent(1 - geo, good: 0.02, bad: 0.4), detail: String(format: "cos %.3f", geo)),
            StratumScore(key: "lbp", label: "Textur (LBP)", percent: distPercent(lbpD, good: 0.12, bad: 1.6), detail: String(format: "χ² %.3f", lbpD)),
            StratumScore(key: "hog", label: "Gradient (HOG)", percent: distPercent(1 - hog, good: 0.08, bad: 0.7), detail: String(format: "cos %.3f", hog)),
            StratumScore(key: "color", label: "Farb-Signatur", percent: distPercent(col, good: 0.08, bad: 1.4), detail: String(format: "χ² %.3f", col)),
        ]
        var num = 0.0
        var den = 0.0
        for meta in StrataMeta.all {
            guard let s = strata.first(where: { $0.key == meta.key }) else { continue }
            let q = 0.45 + 0.55 * quality
            num += s.percent * meta.weight * q
            den += meta.weight * q
        }
        let ensemble = den == 0 ? 0 : num / den
        return (strata + [StratumScore(key: "ensemble", label: "Fusion", percent: ensemble, detail: String(format: "q %.2f", quality))], ensemble)
    }
}
