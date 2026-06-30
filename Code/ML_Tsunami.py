# main_tsunami.py
import os
import time
import joblib
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing your tsunami data loader and configuration
from Tools.data import load_dataset_tsunami
from Models.Models_Tsunami import get_trained_classifiers


def main(data_dir):
    file_path = os.path.join(data_dir, "dataset_earthquake.csv")

    print("1. Loading and preparing Tsunami dataset...")
    X, y = load_dataset_tsunami(file_path)

    # Calculate class imbalance ratio for XGBoost scale_pos_weight
    # ratio = negative_instances / positive_instances
    num_neg = len(y) - sum(y)
    num_pos = sum(y)
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    # 2. Stratified Train/Test Split (Crucial for rare classification events)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standard Scaling for the MLP Classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"   -> Dataset split with Imbalance Weight Rule: {pos_weight:.2f}")
    print(f"   -> Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

    print("\n2. Training and evaluating Classifiers...")
    classifiers = get_trained_classifiers(scale_pos_weight_value=pos_weight)
    resultats = []

    for nom, model in classifiers.items():
        print(f"   - Processing: {nom}...")

        # Direct the correct data type to the model
        if nom in ["MLP Classifier"]:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test

        # Measure Training Time
        start_train = time.time()
        model.fit(X_tr, y_train)
        end_train = time.time()
        t_train = end_train - start_train

        # Measure Prediction Time
        start_pred = time.time()
        y_pred = model.predict(X_te)
        end_pred = time.time()
        t_pred = end_pred - start_pred

        # Calculate Probability Metrics for ROC-AUC
        try:
            y_proba = model.predict_proba(X_te)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except (AttributeError, NotImplementedError):
            roc_auc = 0.5  # Default baseline value if probabilities can't be computed

        # Calculate Classification Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        resultats.append({
            "Modèle": nom,
            "Accuracy": round(acc, 3),
            "F1-Score": round(f1, 3),
            "ROC-AUC": round(roc_auc, 3),
            "Recall": round(recall, 3),
            "Tps Train (s)": round(t_train, 4),
            "Tps Pred (s)": round(t_pred, 4)
        })

    # 3. Output comparative analysis
    df_performance = pd.DataFrame(resultats)
    print("\n=== TSUNAMI CLASSIFICATION PERFORMANCE TABLE ===")
    print(df_performance.to_string(index=False))
    print("================================================\n")

    # save xgboost model
    xgb_model = classifiers.get("XGBoost Classifier")
    if xgb_model:
        joblib.dump(xgb_model, os.path.join(data_dir, "xgboost_tsunami_model.pkl"))
        print("XGBoost model saved as 'xgboost_tsunami_model.pkl' in the dataset directory.")


if __name__ == "__main__":
    main("./dataset/")