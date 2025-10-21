# A data-driven framework that combines unsupervised clustering with multivariate time series prediction to achieve scenario-adaptive predictions of cutter suction dredger (CSD) construction performance.

Note: The presented code represents a subset of the core components and does not constitute a complete implementation. It is intended solely to illustrate a methodological framework for scenario-adaptive prediction of CSD productivity and energy consumption.

## ⚙️ Environment Setup

Recommended Python version: 3.9+

python -m venv dpstimesnet_env
source dpstimesnet_env/bin/activate  # Linux/Mac
dpstimesnet_env\Scripts\activate     # Windows
pip install -r requirements.txt

requirements.txt:
pandas>=1.5.3  
numpy>=1.24.3  
matplotlib>=3.7.1  
scipy>=1.11.0  
minepy>=1.3.1  
scikit-learn>=1.3.0  
torch>=2.1.0  
torchvision>=0.16.0  
torchaudio>=2.1.0  
openpyxl>=3.1.2  
fcmeans>=1.1.0  
kepler-opt>=0.1.0  
shap>=0.42.1  

## 🧹 1. Data Preprocessing

### Boxplot Outlier Removal
Remove outliers using 1.5×IQR and interpolate missing values.

python preprocessing/boxplot_preprocessing.py


### Savitzky–Golay (S-G) Smoothing
Smooth time series to reduce noise:

python preprocessing/sg_smoothing.py


### MIC-PCC Feature Selection

MIC ≥ 0.2

Remove features with |PCC| ≥ 0.8

python preprocessing/mic_pcc_feature_selection.py


Output: Excel files ready for clustering and prediction.

## 📊 2. Clustering Analysis

Perform clustering to identify patterns or operational scenarios.

### K-Means Clustering

python clustering/kmeans_clustering.py


### GMM Clustering

python clustering/gmm_clustering.py


### FCM (Fuzzy C-Means) Clustering

python clustering/fcm_clustering.py


Evaluation metrics: Silhouette Coefficient (SC) and Davies-Bouldin Index (DBI).
Outputs: cluster labels for each sample.

## 🧠 3. DPSTimesNet Prediction

Multi-output real-time prediction model

8 inputs, 2 outputs (customizable)

Sliding window FFT self-attention + ProbSparse attention

Hyperparameters optimized using Kepler Optimization Algorithm (KOA)

Example:
from models.dpstimesnet import DPSTimesNet
import pandas as pd

df = pd.read_excel("data/preprocessed/IPR_selected.xlsx")

model = DPSTimesNet(input_dim=8, output_dim=2, ...)
model.train(df, epochs=50)
predictions = model.predict(df)


Evaluation metrics:  
RMSE  
MAE  
MAPE  
R²  

## 🔍 4. SHAP Feature Importance Analysis

Use SHAP to explain feature contributions:

python analysis/shap_analysis.py

Generates feature importance bar plot

Generates summary plot showing positive/negative impact

Saves SHAP values per sample in shap_values.xlsx
