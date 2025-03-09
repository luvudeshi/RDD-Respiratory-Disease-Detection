import numpy as np
import os
import pandas as pd
import glob
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import seaborn as sns

def perform_statistical_analysis(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    b = fp
    c = fn
    mcnemar_stat = (abs(b - c) - 1)**2 / (b + c)
    mcnemar_p = stats.chi2.sf(mcnemar_stat, df=1)
    kappa = cohen_kappa_score(y_true, y_pred)
    n = len(y_true)
    accuracy = (tp + tn) / n
    z = 1.96
    ci_width = z * np.sqrt((accuracy * (1 - accuracy)) / n)
    ci_lower = accuracy - ci_width
    ci_upper = accuracy + ci_width
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.subplot(2, 2, 2)
    metrics = [sensitivity, specificity, ppv, npv]
    metric_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
    plt.bar(metric_names, metrics)
    plt.title('Performance Metrics')
    plt.ylim(0, 1)
    plt.subplot(2, 2, 3)
    plt.errorbar(x=['Accuracy'], y=[accuracy], yerr=[[ci_width], [ci_width]], fmt='o', capsize=5)
    plt.title('Accuracy with 95% CI')
    plt.ylim(0.8, 1.0)
    plt.tight_layout()
    plt.savefig('statistical_analysis.png')
    plt.close()
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'mcnemar_p': mcnemar_p,
        'kappa': kappa,
        'ci': (ci_lower, ci_upper)
    }

def pad_or_truncate(feature, target_shape=(128, 64)):
    original_shape = feature.shape
    if original_shape[0] < target_shape[0]:
        padding = target_shape[0] - original_shape[0]
        feature = np.pad(feature, ((0, padding), (0, 0)), mode='constant', constant_values=0)
    elif original_shape[0] > target_shape[0]:
        feature = feature[:target_shape[0], :]
    return feature

def load_features_and_labels(features_csv, clinical_csv, noisy_csv, file_pattern):
    features, labels, clinical_data = [], [], []
    noise_types = []
    features_df = pd.read_csv(features_csv)
    clinical_df = pd.read_csv(clinical_csv)
    noisy_df = pd.read_csv(noisy_csv)
    categorical_columns = ['sex', 'tb_prior', 'tb_prior_Pul', 'tb_prior_Extrapul', 'tb_prior_Unknown', 'hemoptysis', 'weight_loss', 'smoke_lweek', 'fever', 'night_sweats']
    clinical_df = pd.get_dummies(clinical_df, columns=categorical_columns)
    sound_scores = dict(zip(features_df['filename'], features_df['sound_prediction_score']))
    files = glob.glob(file_pattern)
    for file_path in tqdm(files, desc="Loading Features"):
        filename = os.path.basename(file_path)
        filename_without_npy = filename.replace('.npy', '')
        if 'noise' in filename:
            matching_row = noisy_df[noisy_df['feature_path'].str.contains(filename)]
            if not matching_row.empty:
                feature = np.load(file_path)
                feature = pad_or_truncate(feature)
                features.append(feature)
                labels.append(int(matching_row['tb_status'].values[0]))
                clinical_features = matching_row[['age', 'height', 'weight', 'reported_cough_dur', 'heart_rate', 'temperature']].values[0]
                clinical_features = np.append(clinical_features, matching_row['sound_prediction_score'].values[0])
                clinical_data.append(clinical_features)
                noise_types.append(matching_row['noise_type'].values[0])
        else:
            matching_row = features_df[features_df['filename'] == filename_without_npy]
            if not matching_row.empty:
                participant = matching_row['participant'].values[0]
                clinical_row = clinical_df[clinical_df['participant'] == participant]
                if not clinical_row.empty:
                    feature = np.load(file_path)
                    feature = pad_or_truncate(feature)
                    features.append(feature)
                    labels.append(int(clinical_row['tb_status'].values[0]))
                    clinical_features = clinical_row[['age', 'height', 'weight', 'reported_cough_dur', 'heart_rate', 'temperature']].values[0]
                    sound_score = sound_scores.get(filename_without_npy, 0)
                    clinical_features = np.append(clinical_features, sound_score)
                    clinical_data.append(clinical_features)
                    noise_types.append('none')
    return np.array(features), np.array(labels), np.array(clinical_data), noise_types

features_csv = "data/raw/solicited_coughs/metadata/CODA_TB_Solicited_Meta_Info.csv"
clinical_csv = "data/raw/solicited_coughs/metadata/CODA_TB_Clinical_Meta_Info.csv"
noisy_csv = "data/processed/solicited_metadata_with_noise.csv"
feature_files = "data/processed/features/*.npy"
X_audio, y, X_clinical, noise_types = load_features_and_labels(features_csv, clinical_csv, noisy_csv, feature_files)
print(f"Loaded {len(X_audio)} samples")
