# EMG Hand Gesture Classifier

A complete EMG gesture classification pipeline developed as part of an internship project at Carleton University. An end-to-end pipeline for classifying hand gestures from surface EMG signals using classical machine learning. The project covers data loading, windowing, time-domain feature extraction, and subject-wise train/test evaluation across binary and 6-class gesture tasks.

---

## Results Summary

| Model | Task | Accuracy | Macro F1 |
|-------|------|----------|----------|
| Logistic Regression | Binary | 99.79% | 0.9963 |
| Random Forest (20 trees) | Binary | 100.00% | 1.0000 |
| Logistic Regression | 6-class (RMS + MAV) | 88.45% | 0.8841 |
| Random Forest (20 trees) | 6-class (RMS + MAV) | 88.45% | 0.8827 |
| Logistic Regression | 6-class (RMS + MAV + WL + ZC) | 87.22% | 0.8715 |
| Random Forest (20 trees) | 6-class (RMS + MAV + WL + ZC) | 89.28% | 0.8910 |

---

## Project Structure

```
emg-hand-gesture-classifier/
├── data/
│   └── EMG_data_for_gestures-master/        # Raw dataset (36 subjects, .txt files)
├── notebooks/
│   ├── Binary_Classification.ipynb          # Rest vs Active (2-class)
│   ├── Multi_Class_Classification.ipynb     # 6-gesture classification (RMS + MAV)
│   └── Multi_Class_Classification_WL.ipynb  # 6-gesture classification (RMS + MAV + WL + ZC)
└── .gitignore
```

---

## Dataset

UCI EMG Dataset for Gestures: https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures

Download and place in: `data/EMG_data_for_gestures-master/`

36 subjects, 8 EMG channels recorded at 1000 Hz.

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

## Installation

```
git clone https://github.com/Akshar51k/emg_hand_gesture_classifier.git
cd emg-hand-gesture-classifier
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

---

## Running the Notebooks

Each notebook in `notebooks/` is fully standalone and reproducible. Launch from the project root:

```
jupyter notebook
```

| Notebook | Task |
|----------|------|
| `Binary_Classification.ipynb` | Rest vs Active, ablation study |
| `Multi_Class_Classification.ipynb` | 6-class with RMS + MAV |
| `Multi_Class_Classification_WL.ipynb` | 6-class with RMS + MAV + WL + ZC |

---

## Key Findings

- Handcrafted feature baseline is strong and stable across all tasks
- Binary classification achieves near-perfect results with all feature combinations, confirming strong amplitude separability between rest and active gestures
- Adding WL and ZC improves Random Forest performance slightly (89.28% vs 88.45%) while Logistic Regression shows a marginal drop — suggesting tree-based models benefit more from the richer feature set
- Persistent confusion pairs: Wrist flexion vs Ulnar deviations, and Wrist extension vs Radial deviations — biomechanically similar gestures that produce overlapping EMG patterns
- Subject-wise evaluation is critical — random splits inflate accuracy due to subject-specific signal characteristics

---

## Author

Akshar Khatrani
Carleton University Internship
Project B: Conventional Labeler
Supervisor: Dr. Leonard MacEachern
