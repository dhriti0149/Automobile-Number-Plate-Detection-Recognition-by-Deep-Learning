# Automobile-Number-Plate-Detection-Recognition-by-Deep-Learning
Here, I have taken a dataset for car number plates and then I have applied object recognition methods and then I have applied Optical Character Recognition on the best object recognition model to read the number plates


<div align="center">

# 🚘 AutoPlate
### Self-Calibrating, Stress-Tested & Statistically Validated License Plate Recognition

[![Python](https://shields.io)](https://python.org)
[![PyTorch](https://shields.io)](https://pytorch.org)
[![Detector](https://shields.io)](https://ultralytics.com)
[![OCR](https://shields.io)](https://github.com)
[![License](https://shields.io)](https://opensource.org)

**No guesswork. No static thresholds. Every model and parameter is chosen by data.**

[Overview](#-overview) • [Pipeline Architecture](#-pipeline-architecture) • [Benchmark Results](#-benchmark-results) • [Statistical Proof](#-statistical-validation) • [Quick Start](#-quick-start)

</div>

---

##  Overview

Real-world license plate readers fail when faced with **headlight glare, motion blur, sensor noise, and distant low-resolution cameras**. Instead of arbitrarily picking popular models and hardcoding confidence thresholds, **AutoPlate** puts **3 distinct detector architectures** and **3 distinct OCR paradigms** through a rigorous, automated elimination pipeline:

1. **Stress-Tests Under Real Degradations:** Evaluates every model across clean images plus 4 synthetic camera failure modes (`Low-Light Glare`, `Motion Blur`, `Sensor Noise`, and `Downscale Blur`).
2. **Eliminates Sub-Optimal Combos:** Prunes all 9 detector–OCR pairs down to the Pareto-optimal winner (**`RT-DETR-l` + `PARSeq`**).
3. **Proves Gains Statistically:** Validates the winning pipeline using **McNemar's Exact Test**, **Wilcoxon Signed-Rank Test**, and **10,000-resample Bootstrap 95% Confidence Intervals**.
4. **Auto-Calibrates at Runtime:** Dynamically computes input resolution, max-F1 confidence cutoffs, NMS IoU thresholds, proportional border padding, and character-slot corrections directly from dataset statistics—with **zero cloud API keys**.

---

##  Pipeline Architecture

```text
┌───────────────────┐     ┌────────────────────────┐     ┌──────────────────────────┐
│  Input Image /    │     │  Stage 1: Detector     │     │  Dynamic Crop & Pad      │
│  Video Stream     │────▶│  RT-DETR-l (Apache 2.0)│────▶│  NMS + Proportional Pad  │
└───────────────────┘     └────────────────────────┘     └────────────┬─────────────┘
                                                                      │
┌───────────────────┐     ┌────────────────────────┐     ┌────────────▼─────────────┐
│  Annotated Output │     │  Post-Processing       │     │  Stage 2: Recognizer     │
│  + CSV / ByteTrack│◀────│  Inferred Slot Regex   │◀────│  PARSeq (ViT + PLM)      │
└───────────────────┘     └────────────────────────┘     └──────────────────────────┘
```

---

## 📊 Benchmark Results

### Stage 1: Object Detector Stress-Test Leaderboard
Three distinct detection paradigms—a real-time transformer (**RT-DETR-l**), a single-stage CNN (**YOLO11s**), and a two-stage detector (**Faster R-CNN v2**)—were evaluated on the exact same validation split across clean and degraded conditions:

| Rank | Detector Architecture | Reliability Index | Clean Recall | Stress Recall | Mean IoU | Latency | Peak VRAM |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **RT-DETR-l** *(Transformer)* | **`73.6`** | **`76.0%`** | **`66.1%`** | **`0.8367`** | `29.2 ms` | `1426 MB` |
| **#2** | **YOLO11s** *(Single-Stage CNN)* | `48.7` | `63.6%` | `58.3%` | `0.8301` | **`12.3 ms`** | **`1285 MB`** |
| **#3** | **Faster R-CNN v2** *(Two-Stage)* | `43.9` | `61.2%` | `42.6%` | `0.8343` | `186.5 ms` | `1931 MB` |

> **Why `RT-DETR-l` won Stage 1:** Held the highest recall across every degradation suite (`LowLight`, `Blur`, `Noise`, `Downscale`) and produced the tightest bounding boxes (`0.8367` Mean IoU) for downstream OCR cropping.

<p align="center">
  <img src="assets/detector_stability_dashboard.png" alt="Stage 1 Detector Stability Dashboard" width="90%">
</p>

---

### Stage 2: Text Recognizer Leaderboard (on `RT-DETR-l` Crops)
Three distinct vision-to-text architectures were benchmarked directly on predicted plate crops from Stage 1:

| Rank | Recognizer Architecture | Reliability Index | Valid Read Rate | Mean Confidence | Latency (Max Spike) | Peak VRAM |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **#1** | **PARSeq** *(Permutation ViT)* | **`94.8`** | `88.7%` | **`0.739`** | **`21.4 ms`** (`53.2 ms`) | `1723 MB` |
| 🥈 **#2** | **TrOCR-Small** *(Seq2Seq Transformer)* | `71.2` | **`92.4%`** | `0.631` | `48.7 ms` (`134.4 ms`) | `1736 MB` |
| 🥉 **#3** | **PP-OCR** *(DBNet + CTC)* | `33.3` | `68.8%` | `0.567` | `481.4 ms` (`4798 ms`) | **`1721 MB`** |

<p align="center">
  <img src="assets/ocr_stability_dashboard.png" alt="Stage 2 OCR Stability Dashboard" width="90%">
</p>

---

## 🔬 Statistical Validation

To verify that **`RT-DETR-l + PARSeq`** genuinely outperforms the runner-up **`YOLO11s + PARSeq`** rather than benefiting from sample variance, paired non-parametric hypothesis tests were executed across all `N = 109` validation images (`10,000` bootstrap resamples):

| Evaluation Metric | Statistical Test | `RT-DETR-l + PARSeq` | `YOLO11s + PARSeq` | p-value | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Clean End-to-End Success** | McNemar's Exact Test | **8 exclusive wins** | 1 exclusive win | `p = 0.0391` | ✅ **Significant** |
| **Bounding-Box Tightness (IoU)** | Wilcoxon Signed-Rank | **`0.7466`** `[0.71, 0.78]` | `0.7004` `[0.65, 0.75]` | `p = 0.0405` | ✅ **Significant** |
| **Stressed Recall (Blur + Noise)** | Bootstrap 95% CI | **`36.70%`** `[27.5, 45.9]` | `26.61%` `[18.4, 34.9]` | `p = 0.0801` | 📈 Practical Lead |
| **End-to-End Latency** | Wilcoxon Signed-Rank | `57.10 ms` `[53.8, 61.0]` | **`39.11 ms`** `[36.7, 41.7]` | `p < 0.0001` | ⚡ Speed Trade-off |

<p align="center">
  <img src="assets/statistical_significance_dashboard.png" alt="Paired Statistical Significance Dashboard" width="90%">
</p>

---

## 🖼️ Qualitative Output

Real-world inference examples showing tight localization, high character confidence (`0.83–1.00`), and automatic rejection of unreadable low-resolution background plates:

<p align="center">
  <img src="assets/qualitative_results.png" alt="AutoPlate Qualitative Predictions" width="85%">
</p>

---

##  Quick Start

### 1. Clone & Install
```bash
git clone https://github.com<your-username>/AutoPlate.git
cd AutoPlate
pip install ultralytics torch torchvision timm pillow opencv-python scipy matplotlib
```

### 2. Run Self-Calibrating Inference & Export Package
```python
from production_anpr import DynamicProductionANPR

# Automatically selects top models, calibrates thresholds, runs inference, and exports .pt + ONNX
anpr = DynamicProductionANPR(s1_metrics=metrics, s2_metrics=ocr_metrics)
anpr.run_all()
```

### 3. Outputs Generated Automatically
* `anpr_outputs_dynamic/` — Annotated images with bounding boxes, plate strings, and confidence badges.
* `anpr_results_dynamic.csv` — Structured log of every detected plate, OCR confidence, detector score, and bounding box.
* `anpr_trained_models.zip` — Portable deployment bundle containing `.pt`, `.onnx`, and `calibrated_config.json`.

---

##  Repository Structure

```text
AutoPlate/
├── assets/
│   ├── detector_stability_dashboard.png        # Stage 1 4-panel stress-test dashboard
│   ├── ocr_stability_dashboard.png             # Stage 2 4-panel recognizer dashboard
│   ├── statistical_significance_dashboard.png  # Paired McNemar, Wilcoxon & Bootstrap CIs
│   └── qualitative_results.png                 # Sample annotated predictions
├── prepare_dataset.py                          # Deterministic train/val split & YOLO/COCO export
├── benchmark_detectors.py                      # Stage 1 perturbation stress test & ranking
├── benchmark_ocr.py                            # Stage 2 zero-shot recognizer benchmark
├── statistical_tests.py                        # Non-parametric paired hypothesis testing
├── production_anpr.py                          # Self-calibrating pipeline + ONNX bundler
└── README.md
```

---

## 📜 License & Acknowledgements
* **Detector (`RT-DETR-l`):** [Ultralytics](https://github.comultralytics/ultralytics) — Apache 2.0 License
* **Recognizer (`PARSeq`):** [Baudm / PARSeq](https://github.com) — Apache 2.0 License

