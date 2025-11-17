# Chitra — Photo Intelligence CLI

Chitra is a local, privacy-preserving CLI that scans your photo folders, extracts EXIF, computes image embeddings, auto-tags, finds near-duplicates, clusters similar photos, detects faces (optional), and stores **everything in SQLite** — without moving your files. It can also export a static HTML gallery and provides a **Terminal UI** built with Textual.

## Highlights
- ✅ Photos remain untouched (no moves/copies)
- ✅ SQLite database for metadata, tags, groups, faces
- ✅ CLIP embeddings (open_clip + Torch) for similarity & natural language search
- ✅ FAISS for fast nearest-neighbor lookup and accelerated search
- ✅ GPS coordinate extraction from EXIF
- ✅ Incremental scanning and analysis (skip already-processed files)
- ✅ Duplicate detection using perceptual hashing
- ✅ Optional face clustering (install `dlib` + `face_recognition`)
- ✅ Static HTML gallery export
- ✅ Full TUI browser

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Initialize the database
python -m cli.main init

# Index your photos (no ML yet)
python -m cli.main scan --path ~/Pictures

# Analyze (embeddings + auto-tags)
python -m cli.main analyze

# Cluster visually similar photos
python -m cli.main cluster --threshold 0.78

# Detect faces (optional, requires face_recognition + dlib)
python -m cli.main faces

# Natural language search (FAISS-accelerated)
python -m cli.main search "beach sunset 2019"

# Find duplicate photos
python -m cli.main duplicates --threshold 5

# Export a static HTML gallery to ./chitra
python -m cli.main export gallery --output ./chitra

# Launch the TUI
python -m cli.main tui
```

> **Note on face clustering**: `face_recognition` depends on `dlib` which can be heavy to build. If you want this feature, install those two packages manually and Chitra will automatically enable face processing.

## Config
- Default output folder: `./chitra`
- Default DB path: `./photo.db` (configurable via `--db` on any command)

## Commands
- `init` — create database schema
- `scan` — index files, EXIF, checksum, GPS coordinates (supports incremental scanning)
- `analyze` — compute embeddings + auto tags (supports incremental analysis)
- `cluster` — group similar images into cluster IDs
- `faces` — optional face detection/clustering (auto-skips if deps missing)
- `search` — FAISS-accelerated natural language search over your library
- `duplicates` — find potential duplicate photos using perceptual hashing
- `export gallery` — build a static HTML gallery
- `tui` — interactive terminal UI

## License
MIT
