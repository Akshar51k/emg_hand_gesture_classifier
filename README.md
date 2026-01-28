# EMG Gesture Labeler

This repository implements a conventional EMG gesture classification pipeline using time-domain features and subject-wise evaluation.

## Overview
The project focuses on correct signal preprocessing, windowing, feature extraction, and fair evaluation rather than complex models.

## Dataset
- Publicly available UCI EMG dataset
- Multi-subject recordings
- Surface EMG (sEMG) signals

## Methodology
- Fixed-length windowing (250 ms, 50% overlap)
- Time-domain feature extraction (RMS, MAV)
- Subject-wise train/test split
- Classical machine learning models (Logistic Regression, SVM, Random Forest)

## Evaluation
- Accuracy and macro-F1 score
- Confusion matrices
- Error analysis with EMG visualizations

## Notes
This repository is part of an undergraduate capstone project and emphasizes experimental rigor and reproducibility.

## Author
Akshar Khatrani
