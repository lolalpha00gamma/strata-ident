from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import (
    Identity,
    MatchResult,
    StrataEngine,
    extract_frames,
    load_gallery,
    match_faces,
    save_gallery,
)


def _print_matches(rows: list[MatchResult], top: int) -> None:
    seen: set[tuple[str, str]] = set()
    n = 0
    for row in rows:
        key = (row.probe, row.identity)
        if key in seen:
            continue
        seen.add(key)
        score = row.temporal if row.temporal is not None else row.ensemble
        print(f"{score:6.1f}%  {row.decision:8}  {Path(row.probe).name}  →  {row.identity}")
        for s in row.strata:
            print(f"          {s.percent:5.1f}%  {s.label:22}  {s.detail}")
        print()
        n += 1
        if n >= top:
            break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="strata-ident",
        description="Strata Ident — Frames extrahieren, Gesichter matchen, Straten in Prozent.",
    )
    parser.add_argument("--model", default="buffalo_l", help="insightface pack: buffalo_l (genau) oder buffalo_sc (schnell)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="Frames aus einem Video ziehen")
    p_ex.add_argument("video")
    p_ex.add_argument("--out", required=True)
    p_ex.add_argument("--fps", type=float, default=2.0)
    p_ex.add_argument("--max-frames", type=int, default=80)

    p_en = sub.add_parser("enroll", help="Person in die Galerie schreiben")
    p_en.add_argument("--name", required=True)
    p_en.add_argument("--gallery", default="gallery.json")
    p_en.add_argument("images", nargs="+")

    p_sc = sub.add_parser("scan", help="Ordner/Datei gegen die Galerie scannen")
    p_sc.add_argument("path")
    p_sc.add_argument("--gallery", default="gallery.json")
    p_sc.add_argument("--fps", type=float, default=2.0)
    p_sc.add_argument("--max-frames", type=int, default=80)
    p_sc.add_argument("--frames-out")
    p_sc.add_argument("--json")
    p_sc.add_argument("--top", type=int, default=20)

    p_cmp = sub.add_parser("compare", help="Zwei Bilder direkt gegeneinander")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")

    args = parser.parse_args(argv)
    engine = StrataEngine(model=args.model)

    if args.cmd == "extract":
        out = Path(args.out)
        frames = extract_frames(Path(args.video), args.fps, args.max_frames, out)
        print(f"{len(frames)} Frames → {out}")
        return 0

    if args.cmd == "enroll":
        gallery_path = Path(args.gallery)
        gallery = load_gallery(gallery_path) if gallery_path.exists() else []
        ident = next((g for g in gallery if g.name.lower() == args.name.lower()), None)
        if ident is None:
            ident = Identity(name=args.name)
            gallery.append(ident)
        for img in args.images:
            faces = engine.scan_path(Path(img), fps=2, max_frames=40)
            if not faces:
                print(f"kein Gesicht: {img}", file=sys.stderr)
                continue
            ident.faces.extend(faces)
            print(f"+ {len(faces)} Gesicht(er)  {img}")
        save_gallery(gallery, gallery_path)
        print(f"Galerie {gallery_path} · {ident.name} · {len(ident.faces)} Referenzen")
        return 0

    if args.cmd == "compare":
        a = engine.scan_path(Path(args.a), fps=2, max_frames=8)
        b = engine.scan_path(Path(args.b), fps=2, max_frames=8)
        if not a or not b:
            print("Mindestens ein Bild ohne Gesicht.", file=sys.stderr)
            return 2
        ident = Identity(name=Path(args.b).name, faces=b)
        rows = match_faces(a, [ident])
        _print_matches(rows, top=5)
        return 0

    if args.cmd == "scan":
        gallery_path = Path(args.gallery)
        if not gallery_path.exists():
            print("Keine Galerie. Zuerst: strata-ident enroll --name NAME foto.jpg", file=sys.stderr)
            return 2
        gallery = load_gallery(gallery_path)
        probes = engine.scan_path(
            Path(args.path),
            fps=args.fps,
            max_frames=args.max_frames,
            frame_dir=Path(args.frames_out) if args.frames_out else None,
        )
        print(f"{len(probes)} Gesichter gefunden")
        rows = match_faces(probes, gallery)
        _print_matches(rows, top=args.top)
        if args.json:
            payload = [
                {
                    "probe": r.probe,
                    "identity": r.identity,
                    "ensemble": r.ensemble,
                    "temporal": r.temporal,
                    "decision": r.decision,
                    "strata": [s.__dict__ for s in r.strata],
                }
                for r in rows
            ]
            Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"JSON → {args.json}")
        return 0

    return 1
