# 🖐️ EMG Hand Gesture Classifier

An end-to-end pipeline for classifying hand gestures from surface EMG signals using classical machine learning. The project covers data loading, windowing, time-domain feature extraction, and subject-wise train/test evaluation across three task levels.

---

## 📁 Repository Structure

```
emg-hand-gesture-classifier/
├── data/
│   └── EMG_data_for_gestures-master/   # Raw dataset (36 subjects, .txt files)
├── notebooks/
│   ├── Binary_Classification.ipynb     # Rest vs Active (2-class)
│   ├── Multi_Class_Classification.ipynb         # 6-gesture classification (RMS + MAV)
│   └── Multi_Class_Classification_WL.ipynb      # 6-gesture classification (RMS + MAV + WL + ZC)
└── .gitignore
```

---

## 📊 Dataset

**EMG Data for Gestures** — 36 subjects, 8 EMG channels recorded at 1000 Hz.

| Class | Gesture |
|-------|---------|
| 0 | Unmarked / Transition |
| 1 | Hand at rest |
| 2 | Hand clenched in a fist |
| 3 | Wrist flexion |
| 4 | Wrist extension |
| 5 | Radial deviations |
| 6 | Ulnar deviations |
| 7 | Unclassified |

> Total raw samples: **4,237,907** — windowed down to **16,934** windows (250 ms, non-overlapping).

---

## ⚙️ Pipeline Overview

### 1. Data Loading
All subject `.txt` files are loaded and concatenated. Each file has 9 columns: `time`, 8 EMG channels (`channel1`–`channel8`), and `class`.

### 2. Windowing
- **Window size:** 250 samples (= 250 ms at 1000 Hz)
- **Step size:** 250 (non-overlapping)
- **Label assignment:** Majority vote per window
- **Per-subject windowing** to prevent inter-subject boundary leakage

### 3. Feature Extraction
Time-domain features computed per channel across each window:

| Feature | Description |
|---------|-------------|
| **RMS** | Root Mean Square — signal energy |
| **MAV** | Mean Absolute Value — average rectified amplitude |
| **WL** | Waveform Length — cumulative signal change |
| **ZC** | Zero Crossings (with threshold) — frequency content proxy |

### 4. Train / Test Split
- **Subject-wise split** — subjects 34, 35, 36 held out for testing
- Train: 5,515 windows | Test: 485 windows
- No data leakage: `StandardScaler` fit only on training data

---

## 🧪 Tasks & Results

### Task 1 — Binary Classification: Rest vs Active

Classes 1–6 are used; class 1 ("hand at rest") → label 0, classes 2–6 → label 1.  
Features: **RMS + MAV** (16 features)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 99.79% | 99.63% |
| Random Forest (20 trees) | **100.00%** | **100.00%** |

**Ablation study** (Logistic Regression):

| Features | Accuracy | Macro F1 |
|----------|----------|----------|
| RMS only | ~100% | ~100% |
| MAV only | ~100% | ~100% |
| RMS + MAV | ~100% | ~100% |

> Binary classification is a very easy task — all feature combinations achieve near-perfect results, indicating strong separability between rest and active gestures in EMG amplitude features.

---

### Task 2 — Multi-Class: 6-Gesture Recognition (RMS + MAV)

Features: **RMS + MAV** (16 features)

| Model | Accuracy | Macro F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Logistic Regression | 88.45% | 88.41% | 89.78% | — |
| Random Forest (20 trees) | 88.45% | 88.27% | 89.11% | 88.34% |

**Per-class breakdown (LR):**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Hand at rest | 1.00 | 1.00 | 1.00 |
| Hand clenched in a fist | 1.00 | 0.80 | 0.89 |
| Wrist flexion | 0.72 | 0.99 | 0.84 |
| Wrist extension | 0.93 | 0.81 | 0.87 |
| Radial deviations | 0.91 | 0.94 | 0.92 |
| Ulnar deviations | 0.82 | 0.76 | 0.79 |

---

### Task 3 — Multi-Class: 6-Gesture Recognition (RMS + MAV + WL + ZC)

Features: **RMS + MAV + WL + ZC** (32 features)

| Model | Accuracy | Macro F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Logistic Regression | 87.22% | 87.15% | 88.55% | — |
| Random Forest (20 trees) | **89.28%** | **89.10%** | **89.73%** | **89.18%** |

> Adding WL and ZC features improves Random Forest performance slightly (89.28% vs 88.45%) while Logistic Regression shows a marginal drop — suggesting that tree-based models benefit more from the richer feature set.

---

## 🔍 Key Observations

- **Persistent confusion pairs:** Wrist flexion ↔ Ulnar deviations, and Wrist extension ↔ Radial deviations. These gestures are biomechanically similar and produce overlapping EMG patterns.
- **"Hand at rest" is always perfectly classified** — its EMG amplitude profile is distinctly different from all active gestures.
- **Subject-wise evaluation is critical** — random splits would inflate accuracy due to subject-specific signal characteristics.
- **Data leakage check:** Training mean ≈ 0 and std ≈ 1; test mean ≠ 0 and std ≠ 1 — confirming no scaling leakage.

---

## 🛠️ Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

Install with:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

---

## 🚀 Running the Notebooks

1. Clone the repository:
```bash
git clone https://github.com/Akshar51k/emg-hand-gesture-classifier.git
cd emg-hand-gesture-classifier
```

2. Place the dataset under `data/EMG_data_for_gestures-master/` (one folder per subject, each containing `.txt` files).

3. Launch Jupyter and run the notebooks in order:
```bash
jupyter notebook
```

| Notebook | Task |
|----------|------|
| `Binary_Classification.ipynb` | Rest vs Active, ablation study |
| `Multi_Class_Classification.ipynb` | 6-class with RMS + MAV |
| `Multi_Class_Classification_WL.ipynb` | 6-class with RMS + MAV + WL + ZC |

---

## 📈 Feature Engineering Summary

```
Window (250 samples × 8 channels)
        ↓
Per channel:
  RMS  = sqrt(mean(x²))           → energy
  MAV  = mean(|x|)                → rectified amplitude  
  WL   = sum(|x[i+1] - x[i]|)    → signal complexity
  ZC   = count(sign changes)      → frequency proxy
        ↓
Feature vector: [8×RMS, 8×MAV] = 16 features
             or [8×RMS, 8×MAV, 8×WL, 8×ZC] = 32 features
```

---

## 📄 License

This project is for academic/research use. The EMG dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures).
