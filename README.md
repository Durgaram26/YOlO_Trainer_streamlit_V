# Project YOLO Trainer

A collection of scripts and utilities to prepare datasets, train YOLO models (Ultralytics), and run inference — with a Streamlit UI for interactive tasks.

## Key features

- Dataset preparation helpers (train/val split, YOLO label generation).
- Streamlit interfaces to upload models, inspect classes, and run lightweight inference (`trainerV1.py`, `Trainer_V3.py`).
- Training wrappers that call Ultralytics `YOLO` APIs with CUDA-aware defaults and configurable hyperparameters.
- Support for detection, classification, and segmentation dataset preparation.

## Repository structure (important files)

- `trainerV1.py` — Streamlit app with dataset preparation and model upload utilities.
- `Trainer_V3.py` — Enhanced Streamlit trainer with more configuration options and dataset handling.
- `trainer_v2.py` — Training utilities and scripts.
- `Base_Trainer.py` — Core training helper classes and functions.
- `Sample_model/` — Example model files or artifacts.
- `prepared_dataset/` — Default folder for formatted YOLO datasets (created by the helpers).

## Requirements

- Python 3.8 — 3.10 recommended
- Recommended packages (install via `requirements.txt` if present):

```bash
pip install ultralytics streamlit opencv-python torch torchvision numpy pandas pillow tqdm
```

If you use a GPU, install a matching `torch` build for your CUDA version (see https://pytorch.org).

## Quickstart

1. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. Install dependencies (see above).

3. Run the Streamlit UI to interactively prepare datasets, upload models, and run lightweight inference:

```bash
streamlit run trainerV1.py
# or
streamlit run Trainer_V3.py
```

The Streamlit app provides model upload, class inspection, and dataset preparation tools used to generate a `prepared_dataset` folder with the YOLO `train/` and `val/` subfolders.

## Dataset preparation

- The helpers expect image folders per class (e.g., `datasets/cats/`, `datasets/dogs/`) and will create:
  - `prepared_dataset/train/images`
  - `prepared_dataset/train/labels`
  - `prepared_dataset/val/images`
  - `prepared_dataset/val/labels`

- Labels are generated in YOLO format (one `.txt` per image). You can call the `prepare_dataset(...)` function from the Streamlit apps or run the script functions directly.

- After preparing the dataset, a YAML file is written (see generated `*_data.yaml`) describing `train`, `val`, `nc`, and `names` for Ultralytics training.

## Training

Example using Ultralytics API inside the provided scripts (this repo uses `from ultralytics import YOLO`):

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='prepared_dataset/yolov8n_data.yaml', epochs=50, imgsz=640, batch=16, device='0')
```

Or run your training script (modify args as needed):

```bash
python trainer_v2.py --data prepared_dataset/yolov8n_data.yaml --epochs 50 --imgsz 640 --batch 16
```

Notes:
- The training wrappers in this repo set conservative defaults for `batch` and `amp` to avoid common CUDA memory issues; adjust them to match your GPU.
- Checkpoints and logs are saved under `runs/train/` (or the `results/` folder depending on the script config).

## Inference / Detection

Use Ultralytics `predict`/`model.predict()` to run inference on images or video files:

```python
from ultralytics import YOLO
model = YOLO('runs/train/exp/weights/best.pt')
results = model.predict(source='images/input.jpg', imgsz=640, conf=0.25)
results.save(save_dir='runs/detect/')
```

Or use the Streamlit UI to upload an image and run the model via the `model = YOLO(model_path)` flow.

## Common commands

- Prepare dataset via Streamlit: `streamlit run trainerV1.py` and use the UI.
- Launch training (example): `python trainer_v2.py --data prepared_dataset/yolov8n_data.yaml --epochs 100`.
- Run interactive trainer: `streamlit run Trainer_V3.py`.

## Troubleshooting

- CUDA OOM: reduce `batch` or `imgsz`, disable `amp`, or use gradient accumulation.
- Missing classes or wrong labels: ensure the folder structure and label files (`.txt`) match YOLO format.
- Ultralytics version mismatch: upgrade/downgrade `ultralytics` to match model weights used (some weights require specific Ultralytics releases).

## Notes and next steps

- Add a `requirements.txt` that pins package versions used for training and inference.
- Add examples and sample datasets in `Sample_model/` to speed onboarding.

## License

Add a `LICENSE` file if you plan to publish or share this project. 