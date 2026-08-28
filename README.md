# Strata Ident

**Download für macOS:** [StrataIdent.dmg](https://github.com/lolalpha00gamma/strata-ident/releases/latest/download/StrataIdent.dmg)

Die Datei liegt auch direkt im Repo unter [`dist/StrataIdent.dmg`](./dist/StrataIdent.dmg).

1. DMG öffnen, **StrataIdent** in Programme ziehen.
2. Einmal **Rechtsklick → Öffnen** (unsigniert, kein App Store).
3. Fotos und Videos importieren oder per Drag-and-Drop. Frames aus Videos, Gesichter mit Prozent je Strate.

---

Nativer **macOS-Gesichtsscanner**. Extrahiert Frames aus Videos, erkennt Gesichter auf Fotos und Frames, und gibt **je Erkennungsstrate einen Match in Prozent**.

Ziel des Experiments: Apples Personen-Erkennung (Fotos) in Genauigkeit schlagen — schneller im Scan, nachvollziehbarer in der Entscheidung.

Repo: [lolalpha00gamma/strata-ident](https://github.com/lolalpha00gamma/strata-ident)

## Zwei Wege

| Weg | Wann | Engine |
| --- | --- | --- |
| **Python CLI** | Das Genauigkeits-Experiment | ArcFace `buffalo_l` (InsightFace) + Geometrie, LBP, HOG, Farbe, Fusion, Video-Konsens |
| **SwiftUI-App** | Ohne Extra-Abhängigkeiten, Drag-and-Drop | Apple Vision + dieselben klassischen Straten |

Apple Photos nutzt *eine* interne Faceprint-Clusterung und zeigt **keine** Prozentwerte. Strata zeigt jede Strate einzeln und fusioniert sie qualitätsgewichtet. Die Python-Engine mit ArcFace (512-d, Cosine) ist das Stück, das Photos in 1:1-Verifikation typischerweise übertrifft.

## Straten

| Strate | Was | Rolle |
| --- | --- | --- |
| ArcFace-Embedding | InsightFace `buffalo_l` / `buffalo_sc` | Haupt-Identität. Nur Python. |
| Landmark-Geometrie | 5-Punkt / Vision-Landmarks, Procrustes | Robust gegen Licht |
| Textur (LBP) | Lokale Binärmuster der ausgerichteten Fläche | Haut, Bart, Falten |
| Gradient (HOG) | Orientierte Gradienten | Kontur Augen/Nase/Kiefer |
| Farb-Signatur | HSV-Histogramm | Hilfsstrate, niedrig gewichtet |
| Fusion | Qualitätsgewichtete Summe | Die Kennzahl für die Entscheidung |
| Video-Konsens | Mittel über einen Track in aufeinanderfolgenden Frames | Unterdrückt Einzelfehler |

Entscheidung: **≥ 74 % Treffer**, **55–74 % möglich**, darunter Ablehnung.

## Python — der genaue Scanner (empfohlen)

macOS 13+, Python 3.10+, Apple Silicon oder Intel.

```bash
git clone https://github.com/lolalpha00gamma/strata-ident.git
cd strata-ident
make install
```

Beim ersten Lauf lädt InsightFace das Modell `buffalo_l` (einmalig, ein paar hundert MB).

```bash
# Frames aus einem Video
.venv/bin/python -m strata_ident extract film.mov --out frames --fps 2

# Person einschreiben (mehrere Fotos = besseres Prototyp)
.venv/bin/python -m strata_ident enroll --name Anna --gallery gallery.json anna1.jpg anna2.jpg

# Ordner oder Video scannen
.venv/bin/python -m strata_ident scan ./fotos --gallery gallery.json --json report.json

# Zwei Bilder direkt
.venv/bin/python -m strata_ident compare a.jpg b.jpg
```

Schneller, etwas weniger genau:

```bash
.venv/bin/python -m strata_ident --model buffalo_sc scan ./fotos --gallery gallery.json
```

Video-Optionen: `--fps 2` (Default), `--max-frames 80`, `--frames-out frames/`.

## SwiftUI-App

Xcode 15+, macOS 14+.

```bash
open macos/StrataIdent.xcodeproj
```

Run (⌘R). Dateien per Import oder Drag-and-Drop. Person benennen, Gesicht einschreiben, Prozentwerte je Strate in der Seitenleiste.

Die App braucht kein Python. Für den Vergleich **gegen Apple Photos** den Python-Engine verwenden — Vision allein hat kein öffentliches ArcFace-Äquivalent.

## Ablauf gegen Fotos.app

1. Dieselben Personen in Fotos als Person anlegen (Referenz).
2. Dieselben Referenzfotos in Strata `enroll`.
3. Dieselben Videos/Alben mit `scan` laufen lassen.
4. `report.json` enthält je Probe × Person: Straten, Fusion, Video-Konsens, Entscheidung.
5. Zählen: richtige Zuordnung / falsche / von Fotos verpasst.

ArcFace auf ausgerichteten 112×112-Crops plus Track-Mittel über Video-Frames ist genau der Hebel, den Fotos nicht als Prozent und nicht als Multi-Strate anbietet.

## Datenschutz

Alles läuft lokal. Keine Cloud, keine Telemetrie. Die Galerie ist eine JSON-Datei auf deiner Platte.

## Lizenz

MIT
