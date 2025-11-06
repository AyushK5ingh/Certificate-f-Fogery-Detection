## Certificate Authenticity Detector

Detect whether a certificate (PDF/JPG/JPEG/PNG) is genuine or fake/tampered. The model analyzes visual and textual cues such as logos, signatures, stamps, edges/texture, and OCR text features.

### Key Features
- Supports PDFs and images (JPG/JPEG/PNG/BMP/TIFF)
- Accurate EfficientNet-B3 backbone with attention modules
- Optional focal loss and CosineAnnealingWarmRestarts scheduler
- CLI for single-file inference with OCR-backed feature summary (optional)

---

### Project Structure
```
certificate-authenticity-detector/
├─ dataset/
│  ├─ train_out/
│  │  ├─ fake/
│  │  └─ genuine/
│  ├─ valid_out/
│  │  ├─ fake/
│  │  └─ genuine/
│  └─ test_out/
│     ├─ fake/
│     └─ genuine/
├─ outputs/
│  └─ checkpoints/              # best_model.pth saved here after training
├─ src/
│  ├─ preprocessing.py          # PDF/image loading, preprocessing, dataset loader
│  ├─ feature_extraction.py     # Logo/signature/stamp/text/edge/texture features
│  ├─ model.py                  # EfficientNet-B3 w/ attention + FocalLoss
│  ├─ train.py                  # Training loop
│  └─ predict.py                # Inference CLI (single file)
├─ config.yaml                  # Training & inference defaults
├─ requirements.txt
└─ README.md
```

---

### Prerequisites
- Python 3.10+
- Windows: PowerShell (or any shell)
- PyTorch with CUDA (optional but recommended if you have a GPU)
- Poppler for Windows (for PDF to image): download and add `bin` to PATH
  - Download: `https://github.com/oschwartz10612/poppler-windows/releases/`
- Tesseract OCR (for optional OCR features): install and add to PATH
  - Download: `https://github.com/UB-Mannheim/tesseract/wiki`

### Installation
```bash
# from project root
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you have a specific CUDA version, prefer installing the appropriate torch build first (see PyTorch website), then install the rest of the requirements.

---

### Configuration
Edit `config.yaml` as needed. Important keys:
- `dataset_path`: path to your dataset root containing `train_out`, `valid_out`, `test_out` with class subfolders `fake` and `genuine`.
- `output_dir`: where logs and checkpoints will be saved.
- `model_name`: default `efficientnet_b3`.
- `img_size`: default `1024`.
- `inference.model_checkpoint`: path to `best_model.pth` for inference.

Example snippet:
```yaml
dataset_path: dataset
output_dir: outputs
img_size: 1024
model_name: efficientnet_b3
inference:
  model_checkpoint: outputs/checkpoints/best_model.pth
  device: auto
  img_size: 1024
```

---

### Training
Ensure your dataset is placed as shown in the structure section. Then run:
```bash
# Activate env first
.\.venv\Scripts\activate

# Run training (from project root)
python -m src.train --config config.yaml
```
This will log metrics to TensorBoard (`outputs/logs`) and save the best checkpoint to `outputs/checkpoints/best_model.pth`.

Tips for higher accuracy:
- Increase `img_size` (e.g., 1280) if GPU memory allows
- Increase `num_epochs`
- Enable focal loss for imbalanced datasets (`use_focal_loss: true`)
- Data quality matters: high DPI scans (300+), consistent labeling, remove duplicates

---

### Inference
Use the built-in CLI in `src/predict.py`. It supports PDFs and images.

Single file:
```bash
python -m src.predict \
  "path/to/certificate.pdf" \
  --model outputs/checkpoints/best_model.pth \
  --device auto \
  --img-size 1024 \
  --features \
  --output outputs/prediction.json
```

Arguments:
- `certificate_path` (positional): PDF/JPG/PNG path
- `--model`: path to `.pth` checkpoint (defaults to `outputs/checkpoints/best_model.pth`)
- `--device`: `auto` | `cuda` | `cpu`
- `--img-size`: input size (default 1024)
- `--features`: include feature summary (OCR, edge density, etc.)
- `--output`: save JSON result

Output example (JSON):
```json
{
  "prediction": "GENUINE",
  "confidence": 0.9812,
  "predicted_class": 1,
  "probabilities": { "fake": 0.0188, "genuine": 0.9812 },
  "features": {
    "logo": { "num_keypoints": 143, "num_contours": 12 },
    "signature": { "num_strokes": 487, "stroke_density": 0.34 },
    "stamp": { "num_circles": 3 },
    "text": { "num_words": 124, "avg_confidence": 86.7 },
    "edge": { "edge_density": 0.12 }
  }
}
```

Notes:
- If you installed Tesseract and Poppler correctly (and they are on PATH), PDFs and OCR features will work out of the box.
- If you run from project root, prefer `python -m src.predict ...` so imports resolve correctly.

---

### Dataset Expectations
- Place your files in `dataset/{train_out,valid_out,test_out}/{fake,genuine}`
- Accepted formats: `.pdf`, `.jpg`, `.jpeg`, `.png` (others like BMP/TIFF are supported for single inference but dataset loader filters the common set)
- Keep classes balanced for best results, or enable focal loss

---

### Troubleshooting
- `pdf2image` errors on Windows: ensure Poppler `bin` directory is in PATH
- OCR not working: ensure `tesseract.exe` is in PATH
- GPU OOM: reduce `batch_size` or `img_size`
- Import errors when running from root: use `python -m src.predict` / `python -m src.train`

---

### License
For internal/project use. Add your preferred license here.
