# Chitra Quick Reference

## 🚀 Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Initialize
python -m cli.main init

# Index your photos
python -m cli.main scan --path ~/Pictures

# Analyze with AI
python -m cli.main analyze

# Use the features!
python -m cli.main search "sunset beach"
python -m cli.main tui
```

---

## 📋 Command Reference

### Core Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `init` | Create database | `python -m cli.main init` |
| `scan` | Index photos | `python -m cli.main scan --path ~/Pictures` |
| `analyze` | Generate embeddings & tags | `python -m cli.main analyze` |
| `cluster` | Group similar photos | `python -m cli.main cluster --threshold 0.78` |
| `search` | Natural language search | `python -m cli.main search "cat playing"` |
| `duplicates` | Find duplicate photos | `python -m cli.main duplicates --threshold 5` |
| `faces` | Detect faces | `python -m cli.main faces` |
| `export gallery` | Create HTML gallery | `python -m cli.main export gallery --output ./gallery` |
| `tui` | Launch terminal UI | `python -m cli.main tui` |

---

## 🎛️ Common Options

### Global Options
- `--db PATH` - Specify database path (default: `photo.db`)
- `--help` - Show help for any command

### Scan Options
```bash
python -m cli.main scan --path ~/Pictures --incremental  # Default
python -m cli.main scan --path ~/Pictures --no-incremental  # Full rescan
```

### Analyze Options
```bash
python -m cli.main analyze --incremental  # Only new photos (default)
python -m cli.main analyze --no-incremental  # Reanalyze all
python -m cli.main analyze --limit 100  # Test mode
python -m cli.main analyze --tag-k 10  # More tags per photo
```

### Cluster Options
```bash
python -m cli.main cluster --threshold 0.78  # Default
python -m cli.main cluster --threshold 0.85  # Stricter (fewer groups)
python -m cli.main cluster --threshold 0.70  # Looser (more groups)
```

### Search Options
```bash
python -m cli.main search "query" --top-k 20  # Show top 20 results
python -m cli.main search "query" --db custom.db  # Use custom DB
```

### Duplicates Options
```bash
python -m cli.main duplicates --threshold 5  # Default (Hamming distance)
python -m cli.main duplicates --threshold 3  # Stricter (more similar)
python -m cli.main duplicates --threshold 8  # Looser (less similar)
```

### Faces Options
```bash
python -m cli.main faces  # Process all photos
python -m cli.main faces --limit 100  # Test mode
```

---

## 🔄 Typical Workflows

### First Time Setup
```bash
# 1. Create environment and install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Initialize database
python -m cli.main init

# 3. Index your photos
python -m cli.main scan --path ~/Pictures

# 4. Analyze with AI (takes time)
python -m cli.main analyze

# 5. Cluster similar photos
python -m cli.main cluster

# 6. Optional: detect faces
pip install face_recognition dlib  # Heavy dependencies
python -m cli.main faces

# 7. Try it out!
python -m cli.main search "birthday party"
python -m cli.main tui
```

### Adding New Photos
```bash
# Incremental mode automatically skips existing photos
python -m cli.main scan --path ~/Pictures
python -m cli.main analyze  # Only analyzes new photos
python -m cli.main cluster  # Re-cluster with new photos
```

### Finding Duplicates
```bash
# Find similar photos
python -m cli.main duplicates --threshold 5

# Review groups and manually delete unwanted copies
# Duplicates command only reports, doesn't delete
```

### Exploring Your Collection
```bash
# Search by description
python -m cli.main search "dog in park"

# Browse in terminal
python -m cli.main tui

# Create web gallery
python -m cli.main export gallery --output ./my-gallery
# Open ./my-gallery/index.html in browser
```

---

## 🎯 Use Cases

### 1. Organizing Vacation Photos
```bash
python -m cli.main scan --path ~/Pictures/Vacation2024
python -m cli.main analyze
python -m cli.main search "beach"
python -m cli.main search "restaurant food"
python -m cli.main cluster --threshold 0.80
```

### 2. Finding & Removing Duplicates
```bash
python -m cli.main scan --path ~/Pictures
python -m cli.main duplicates --threshold 5
# Review output and delete duplicates manually
```

### 3. Creating a Photo Gallery
```bash
python -m cli.main scan --path ~/Pictures/Portfolio
python -m cli.main analyze
python -m cli.main export gallery --output ~/Desktop/portfolio
# Share ~/Desktop/portfolio folder
```

### 4. Face Detection (Family Album)
```bash
pip install face_recognition dlib
python -m cli.main scan --path ~/Pictures/Family
python -m cli.main faces
# Future: cluster faces by person
```

### 5. Semantic Search
```bash
python -m cli.main analyze  # Must run first
python -m cli.main search "sunset over water"
python -m cli.main search "people celebrating"
python -m cli.main search "mountain landscape"
```

---

## 🐛 Troubleshooting

### "No embeddings found"
Run `analyze` first:
```bash
python -m cli.main analyze
```

### "Module not found" errors
Install dependencies:
```bash
pip install -r requirements.txt
```

### Face detection not working
Install optional dependencies:
```bash
pip install face_recognition dlib
```

### Slow analysis
- Use GPU if available (CUDA support)
- Process in batches: `--limit 1000`
- Takes ~1-2 seconds per photo on CPU

### Gallery images not showing
- Make sure to keep `thumbnails/` folder with `index.html`
- Gallery is portable - move entire folder together

---

## 💡 Tips & Tricks

### 1. Incremental Processing (Default)
Scan and analyze are incremental by default. Just re-run commands after adding photos!

### 2. Test Mode
Use `--limit 100` to test on small batches:
```bash
python -m cli.main analyze --limit 100
```

### 3. Custom Database
Keep separate databases for different collections:
```bash
python -m cli.main scan --path ~/Work --db work.db
python -m cli.main scan --path ~/Personal --db personal.db
```

### 4. Cluster Threshold Tuning
- Higher (0.85+): Fewer, tighter groups
- Medium (0.75-0.80): Balanced
- Lower (0.65-0.70): More, looser groups

### 5. Duplicate Detection Threshold
- Lower (3-4): Very similar only
- Medium (5-6): Near duplicates
- Higher (7-10): More tolerant

### 6. Search Quality
More tags = better search:
```bash
python -m cli.main analyze --tag-k 10
```

### 7. Export for Sharing
Gallery export creates standalone HTML:
```bash
python -m cli.main export gallery --output ./share
zip -r gallery.zip ./share
# Send gallery.zip to anyone
```

---

## 📊 Performance Tips

### Large Collections (10K+ photos)
- Analyze in batches: `--limit 5000`
- Use SSD for database
- Enable GPU acceleration (CUDA)
- Close other programs during analysis

### Fast Incremental Updates
```bash
# Add new photos
python -m cli.main scan --path ~/Pictures  # Fast
python -m cli.main analyze  # Only new photos
```

### Search Performance
Search is FAISS-accelerated and scales well:
- 1K photos: instant
- 10K photos: < 1 second
- 100K photos: < 2 seconds

---

## 🔧 Advanced Usage

### Multiple Photo Folders
```bash
python -m cli.main scan --path ~/Pictures
python -m cli.main scan --path ~/Desktop/Photos
python -m cli.main scan --path /media/external/Vacation
# All indexed in same database
```

### Custom Tag Set
Edit `core/tagger.py` and add your own labels to `DEFAULT_LABELS`.

### Exclude Patterns
Before scanning, move unwanted folders elsewhere or use file system filters.

---

## 📈 What Each Command Does

| Command | Reads From | Writes To | Time (1K photos) |
|---------|-----------|-----------|------------------|
| `scan` | File system | `photos` table | ~30 seconds |
| `analyze` | `photos` table | `embeddings`, `tags` | ~20 minutes (CPU) |
| `cluster` | `embeddings` | `clusters` table | ~5 seconds |
| `faces` | `photos` table | `faces` table | ~1 minute |
| `search` | `embeddings` | None (query only) | < 1 second |
| `duplicates` | `photos.phash` | None (report only) | ~2 seconds |
| `export gallery` | All tables | HTML files | ~1 minute |
| `tui` | All tables | None (view only) | instant |

---

## 🎓 Learning More

- Check `IMPROVEMENTS.md` for detailed feature docs
- Read `IMPLEMENTATION_SUMMARY.md` for technical details
- See `README.md` for project overview

---

**Happy photo organizing! 📸**

