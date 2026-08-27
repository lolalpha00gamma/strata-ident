from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

STRATA = (
    ("arcface", "ArcFace-Embedding", 0.78),
    ("geometry", "Landmark-Geometrie", 0.05),
    ("lbp", "Textur (LBP)", 0.05),
    ("hog", "Gradient (HOG)", 0.08),
    ("color", "Farb-Signatur", 0.04),
)


@dataclass
class FaceRecord:
    source: str
    frame_index: int | None
    time_ms: int | None
    bbox: list[float]
    det_score: float
    embedding: list[float]
    geometry: list[float]
    lbp: list[float]
    hog: list[float]
    color: list[float]
    quality: float
    crop_path: str | None = None


@dataclass
class StratumScore:
    id: str
    label: str
    percent: float
    detail: str


@dataclass
class MatchResult:
    probe: str
    identity: str
    strata: list[StratumScore]
    ensemble: float
    temporal: float | None
    decision: str


@dataclass
class Identity:
    name: str
    faces: list[FaceRecord] = field(default_factory=list)

    @property
    def prototype(self) -> dict[str, np.ndarray]:
        def mean(attr: str) -> np.ndarray:
            mats = [np.asarray(getattr(f, attr), dtype=np.float32) for f in self.faces]
            return np.mean(np.stack(mats, axis=0), axis=0)

        return {
            "embedding": mean("embedding"),
            "geometry": mean("geometry"),
            "lbp": mean("lbp"),
            "hog": mean("hog"),
            "color": mean("color"),
        }


def _clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def _dist_percent(dist: float, good: float, bad: float) -> float:
    if bad <= good:
        return 100.0 if dist <= good else 0.0
    return _clamp(100.0 * (1.0 - (dist - good) / (bad - good)), 0.0, 100.0)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _chi2(a: np.ndarray, b: np.ndarray) -> float:
    s = a + b
    mask = s > 1e-9
    d = a - b
    return float(np.sum((d[mask] ** 2) / s[mask]))


def _lbp_hist(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    bits = [
        g[:-2, :-2] >= c,
        g[:-2, 1:-1] >= c,
        g[:-2, 2:] >= c,
        g[1:-1, 2:] >= c,
        g[2:, 2:] >= c,
        g[2:, 1:-1] >= c,
        g[2:, :-2] >= c,
        g[1:-1, :-2] >= c,
    ]
    code = np.zeros_like(c, dtype=np.uint8)
    for i, bit in enumerate(bits):
        code |= bit.astype(np.uint8) << (7 - i)
    hist = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    hist /= max(hist.sum(), 1.0)
    return hist


def _hog(gray: np.ndarray, cells: int = 8, bins: int = 8) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    h, w = gray.shape
    cell_h = max(1, h // cells)
    cell_w = max(1, w // cells)
    out = np.zeros(cells * cells * bins, dtype=np.float32)
    for cy in range(cells):
        for cx in range(cells):
            y0, x0 = cy * cell_h, cx * cell_w
            patch_m = mag[y0 : y0 + cell_h, x0 : x0 + cell_w]
            patch_a = ang[y0 : y0 + cell_h, x0 : x0 + cell_w]
            b = np.clip((patch_a / (2 * np.pi) * bins).astype(np.int32), 0, bins - 1)
            base = (cy * cells + cx) * bins
            for i in range(bins):
                out[base + i] = float(patch_m[b == i].sum())
    n = np.linalg.norm(out)
    if n > 0:
        out /= n
    return out


def _color_sig(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    hist /= max(hist.sum(), 1.0)
    return hist


def _geometry(kps: np.ndarray) -> np.ndarray:
    if kps is None or len(kps) == 0:
        return np.zeros(10, dtype=np.float32)
    pts = kps.astype(np.float32)
    center = pts.mean(axis=0)
    scale = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) or 1.0
    rel = (pts - center) / scale
    return rel.flatten().astype(np.float32)


def _sharpness(gray: np.ndarray) -> float:
    return float(min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 400.0))


def extract_frames(path: Path, fps: float, max_frames: int, out_dir: Path | None = None) -> list[tuple[int, int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video nicht lesbar: {path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(native / max(0.5, fps))))
    frames: list[tuple[int, int, np.ndarray]] = []
    idx = 0
    kept = 0
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            frames.append((kept, time_ms, frame))
            if out_dir is not None:
                cv2.imwrite(str(out_dir / f"frame_{kept:04d}.jpg"), frame)
            kept += 1
            if kept >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


class StrataEngine:
    def __init__(self, model: str = "buffalo_l") -> None:
        self.model_name = model
        self._app = None

    def _app_or_die(self):
        if self._app is not None:
            return self._app
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "insightface fehlt. Auf dem Mac: python3 -m venv .venv && "
                ".venv/bin/pip install -r python/requirements.txt"
            ) from exc
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        app = FaceAnalysis(name=self.model_name, providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
        self._app = app
        return app

    def detect(self, bgr: np.ndarray) -> list[Any]:
        return list(self._app_or_die().get(bgr))

    def encode_face(self, bgr: np.ndarray, face: Any, source: str, frame_index: int | None, time_ms: int | None) -> FaceRecord:
        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        x1c, y1c = max(0, int(x1)), max(0, int(y1))
        crop = bgr[y1c:max(y1c + 1, int(y2)), x1c:max(x1c + 1, int(x2))]
        if crop.size == 0:
            crop = bgr
        small = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kps = np.asarray(getattr(face, "kps", np.zeros((5, 2))), dtype=np.float32)
        emb = np.asarray(face.embedding, dtype=np.float32)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        quality = 0.5 * _sharpness(gray) + 0.3 * float(getattr(face, "det_score", 0.5)) + 0.2 * min(1.0, (x2 - x1) / 160.0)
        return FaceRecord(
            source=source,
            frame_index=frame_index,
            time_ms=time_ms,
            bbox=[x1, y1, x2, y2],
            det_score=float(getattr(face, "det_score", 0.0)),
            embedding=emb.tolist(),
            geometry=_geometry(kps).tolist(),
            lbp=_lbp_hist(gray).tolist(),
            hog=_hog(gray).tolist(),
            color=_color_sig(small).tolist(),
            quality=float(_clamp(quality, 0.0, 1.0)),
        )

    def scan_image(self, path: Path) -> list[FaceRecord]:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Bild nicht lesbar: {path}")
        return [self.encode_face(img, face, str(path), None, None) for face in self.detect(img)]

    def scan_video(self, path: Path, fps: float, max_frames: int, frame_dir: Path | None = None) -> list[FaceRecord]:
        out: list[FaceRecord] = []
        for index, time_ms, frame in extract_frames(path, fps, max_frames, frame_dir):
            for face in self.detect(frame):
                out.append(self.encode_face(frame, face, str(path), index, time_ms))
        return out

    def scan_path(self, path: Path, fps: float, max_frames: int, frame_dir: Path | None = None) -> list[FaceRecord]:
        if path.is_dir():
            recs: list[FaceRecord] = []
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in IMAGE_EXT:
                    recs.extend(self.scan_image(child))
                elif child.suffix.lower() in VIDEO_EXT:
                    recs.extend(self.scan_video(child, fps, max_frames, frame_dir))
            return recs
        if path.suffix.lower() in VIDEO_EXT:
            return self.scan_video(path, fps, max_frames, frame_dir)
        return self.scan_image(path)


def score_pair(probe: FaceRecord, proto: dict[str, np.ndarray]) -> list[StratumScore]:
    emb = np.asarray(probe.embedding, dtype=np.float32)
    geo = np.asarray(probe.geometry, dtype=np.float32)
    lbp = np.asarray(probe.lbp, dtype=np.float32)
    hog = np.asarray(probe.hog, dtype=np.float32)
    col = np.asarray(probe.color, dtype=np.float32)
    cos = _cos(emb, proto["embedding"])
    return [
        StratumScore("arcface", "ArcFace-Embedding", _dist_percent(1.0 - cos, 0.15, 0.65), f"cos {cos:.3f}"),
        StratumScore("geometry", "Landmark-Geometrie", _dist_percent(1.0 - _cos(geo, proto["geometry"]), 0.02, 0.35), f"cos {_cos(geo, proto['geometry']):.3f}"),
        StratumScore("lbp", "Textur (LBP)", _dist_percent(_chi2(lbp, proto["lbp"]), 0.12, 1.6), f"χ² {_chi2(lbp, proto['lbp']):.3f}"),
        StratumScore("hog", "Gradient (HOG)", _dist_percent(1.0 - _cos(hog, proto["hog"]), 0.08, 0.7), f"cos {_cos(hog, proto['hog']):.3f}"),
        StratumScore("color", "Farb-Signatur", _dist_percent(_chi2(col, proto["color"]), 0.08, 1.4), f"χ² {_chi2(col, proto['color']):.3f}"),
    ]


def fuse(strata: Iterable[StratumScore], quality: float) -> float:
    net = 0.0
    appearance = 0.0
    den = 0.0
    for s in strata:
        if s.id == "arcface":
            net = s.percent
            continue
        w = {"geometry": 0.05, "lbp": 0.05, "hog": 0.08, "color": 0.04}.get(s.id, 0.0)
        if w <= 0:
            continue
        appearance += s.percent * w
        den += w
    appearance = net if den == 0 else appearance / den
    q = 0.7 + 0.3 * quality
    ensemble = (0.82 * net + 0.18 * appearance) * q + net * (1 - q)
    if net < 42:
        ensemble = min(ensemble, net + 6)
    if net >= 78:
        ensemble = max(ensemble, net - 6)
    return _clamp(ensemble, 0.0, 100.0)


def decide(percent: float) -> str:
    if percent >= 74:
        return "match"
    if percent >= 55:
        return "possible"
    return "reject"


def match_faces(probes: list[FaceRecord], gallery: list[Identity]) -> list[MatchResult]:
    tracks: dict[str, list[FaceRecord]] = {}
    for face in probes:
        tracks.setdefault(face.source, []).append(face)

    temporal: dict[tuple[str, str, int], float] = {}
    for source, group in tracks.items():
        if len(group) < 2:
            continue
        buckets: list[list[FaceRecord]] = []
        for face in group:
            placed = False
            fe = np.asarray(face.embedding)
            for bucket in buckets:
                if _cos(fe, np.asarray(bucket[0].embedding)) > 0.45:
                    bucket.append(face)
                    placed = True
                    break
            if not placed:
                buckets.append([face])
        for bucket in buckets:
            for ident in gallery:
                scores = [fuse(score_pair(f, ident.prototype), f.quality) for f in bucket]
                mean = sum(scores) / len(scores)
                for f in bucket:
                    temporal[(f.source, ident.name, id(f))] = mean

    results: list[MatchResult] = []
    for face in probes:
        for ident in gallery:
            strata = score_pair(face, ident.prototype)
            ensemble = fuse(strata, face.quality)
            temp = temporal.get((face.source, ident.name, id(face)))
            chosen = temp if temp is not None else ensemble
            results.append(
                MatchResult(
                    probe=face.source,
                    identity=ident.name,
                    strata=strata
                    + [StratumScore("ensemble", "Fusion", ensemble, f"q {face.quality:.2f}")]
                    + (
                        [StratumScore("temporal", "Video-Konsens", temp, "Track-Mittel")]
                        if temp is not None
                        else []
                    ),
                    ensemble=ensemble,
                    temporal=temp,
                    decision=decide(chosen),
                )
            )
    results.sort(key=lambda r: r.temporal if r.temporal is not None else r.ensemble, reverse=True)
    return results


def save_gallery(gallery: list[Identity], path: Path) -> None:
    payload = [{"name": g.name, "faces": [asdict(f) for f in g.faces]} for g in gallery]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_gallery(path: Path) -> list[Identity]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[Identity] = []
    for item in raw:
        faces = [FaceRecord(**f) for f in item["faces"]]
        out.append(Identity(name=item["name"], faces=faces))
    return out


def digest_file(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]
