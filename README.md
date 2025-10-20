# A data-driven framework that combines unsupervised clustering with multivariate time series prediction to achieve scenario-adaptive predictions of cutter suction dredger (CSD) construction performance.

 


## 1. Boxplot Outlier Removal

Removes outliers using 1.5×IQR and fills missing values by linear interpolation:

python preprocessing/boxplot_preprocessing.py

## 2. Savitzky–Golay (S-G) Smoothing

Smooths time series to reduce noise:

python preprocessing/sg_smoothing.py

## 3. MIC-PCC Feature Selection

Two-stage feature selection:

MIC ≥ 0.2

Remove features with |PCC| ≥ 0.8

python preprocessing/mic_pcc_feature_selection.py


Output: preprocessed Excel files ready for model training.

## 4. DPSTimesNet Model

Multi-output real-time time series prediction

Features:

Sliding window FFT self-attention

ProbSparse sparse attention

8 inputs, 2 outputs (customizable)

Hyperparameters optimized using Kepler Optimization Algorithm (KOA)
