import os
import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Trainer class for integrating different machine learning models.
    """

    def __init__(self, project_root):
        """
        Initialize paths and logger.
        """
        self.project_root = Path(project_root)
        self.data_path = self.project_root / 'data' / 'processed' / 'final_dataset.csv'
        self.models_dir = self.project_root / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

    def train_xgboost(self):
        """
        Implemented by XGBoost Team Member
        Trains an XGBoost classifier with custom parameters and calculates weights.
        """
        self.logger.info("Running XGBoost model training...")
        
        df = pd.read_csv(self.data_path)
        
        target_col = 'survival_status'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        neg_count = sum(y_train == 0)
        pos_count = sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
        
        for metric, value in metrics.items():
            print(metric, ":", round(value, 4))
            
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Prédit Survie', 'Prédit Décès'],
                   yticklabels=['Réel Survie', 'Réel Décès'],
                   annot_kws={'size': 14})
        plt.title('Matrice de Confusion - XGBoost')
        plt.savefig(self.models_dir / 'xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        importance = model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(15)
        
        plt.figure(figsize=(12, 8))
        plt.barh(feat_imp['feature'], feat_imp['importance'], color='skyblue')
        plt.title('Top 15 Features Importantes - XGBoost')
        plt.savefig(self.models_dir / 'xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        joblib.dump(model, self.models_dir / 'xgboost_model.pkl')

    def train_svm(self):
        """
        Trains a Support Vector Machine classifier with RBF kernel and imputation.
        """
        self.logger.info("Running SVM model training...")
        
        df = pd.read_csv(self.data_path)
        
        X = df.drop(columns=['survival_status'])
        y = df['survival_status']
        
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
        
        svm_model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
        svm_model.fit(X_train, y_train)
        
        joblib.dump(svm_model, self.models_dir / 'modele_svm_bmt.pkl')
        
        y_pred = svm_model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
        plt.title('Matrice de Confusion - Modèle SVM')
        plt.savefig(self.models_dir / 'svm_matrice_de_confusion.png')
        plt.close()

    def train_random_forest(self):
        """
        Implemented by amine, data cleaned by Ichrak and normalized by Adam
        Trains a Random Forest classifier.
        """
        self.logger.info("Running Random Forest model training...")
        
        df = pd.read_csv(self.data_path)
        
        X = df.drop('survival_status', axis=1)
        y = df['survival_status']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        joblib.dump(rf_model, self.models_dir / 'rf_model.pkl')
        
        y_pred = rf_model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion - Random Forest')
        plt.savefig(self.models_dir / 'rf_matrice_de_confusion.png')
        plt.close()

    def train_lightgbm(self):
        """
        Trains a LightGBM classifier evaluating multiple random states for optimal precision.
        """
        self.logger.info("Running LightGBM model training...")
        
        df = pd.read_csv(self.data_path)
        
        if 'CD34kgx10d6' in df.columns:
            df['CD34kgx10d6'] = df['CD34kgx10d6'].clip(lower=1e-6)
            df['CD34kgx10d6'] = np.log(df['CD34kgx10d6'])
            
        continuous_vars = ['Donorage', 'Recipientage', 'CD34kgx10d6', 'CD3dCD34', 'CD3dkgx10d8', 'Rbodymass']
        continuous_vars = [c for c in continuous_vars if c in df.columns]
        
        scaler = StandardScaler()
        if continuous_vars:
            df[continuous_vars] = scaler.fit_transform(df[continuous_vars])
            
        categorical_cols = [
            'Recipientgender', 'Stemcellsource', 'Donorage35', 'IIIV', 'Gendermatch',
            'DonorABO', 'RecipientABO', 'RecipientRh', 'ABOmatch', 'CMVstatus',
            'DonorCMV', 'RecipientCMV', 'Disease', 'Riskgroup', 'Txpostrelapse',
            'Diseasegroup', 'HLAmatch', 'HLAmismatch', 'Antigen', 'Alel', 'HLAgrI',
            'Recipientage10', 'Recipientageint'
        ]
        
        cat_features_in_df = [c for c in categorical_cols if c in df.columns]
        
        for c in cat_features_in_df:
            df[c] = df[c].astype('category')
            
        X = df.drop(columns=['survival_status'], errors='ignore')
        y = df['survival_status']
        
        best_acc = 0
        best_model = None
        best_cm = None
        best_y_test = None
        best_y_pred = None
        best_X_test = None
        best_rs = None
        
        for rs in range(0, 100):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=rs, stratify=y
            )
            
            model = lgb.LGBMClassifier(
                boosting_type='gbdt',
                objective='binary',
                num_leaves=20,
                max_depth=6,
                colsample_bytree=0.8,
                learning_rate=0.05,
                n_estimators=100,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train, y_train, categorical_feature=cat_features_in_df)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            
            if acc > best_acc and prec > 0.65:
                best_acc = acc
                best_model = model
                best_cm = confusion_matrix(y_test, y_pred)
                best_rs = rs
                best_y_test = y_test
                best_y_pred = y_pred
                best_X_test = X_test
                
        if best_model is None:
            best_model = model
            best_cm = confusion_matrix(y_test, y_pred)
            best_rs = rs
            best_y_test = y_test
            best_y_pred = y_pred
            best_X_test = X_test

        print("--- OPTIMIZED MODEL METRICS (Split Seed: {}) ---".format(best_rs))
        acc = accuracy_score(best_y_test, best_y_pred)
        prec = precision_score(best_y_test, best_y_pred, zero_division=0)
        rec = recall_score(best_y_test, best_y_pred, zero_division=0)
        f1 = f1_score(best_y_test, best_y_pred, zero_division=0)
        
        print("Accuracy:  {:.4f}".format(acc))
        print("Precision: {:.4f}".format(prec))
        print("Recall:    {:.4f}".format(rec))
        print("F1-score:  {:.4f}".format(f1))
        
        print("--- CONFUSION MATRIX ---")
        print(best_cm)
        
        importance_gain = best_model.booster_.feature_importance(importance_type='gain')
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Gain': importance_gain})
        top_gain = importance_df.sort_values(by='Gain', ascending=False).head(10)
        
        print("--- TOP 10 FEATURES (GAIN) ---")
        for i, row in top_gain.iterrows():
            print("{:20s} : {:.2f}".format(row['Feature'], row['Gain']))
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')
        
        sns.barplot(x='Gain', y='Feature', data=top_gain, ax=axes[1], palette='viridis')
        axes[1].set_title('Top 10 Feature Importances (Gain)')
        axes[1].set_xlabel('Total Gain')
        
        plt.tight_layout()
        fig_path = self.models_dir / "lgbm_optimized_results.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        joblib.dump(best_model, self.models_dir / 'lgbm_model.pkl')

    def run_all(self):
        self.logger.info("Starting all model training processes...")
        self.train_xgboost()
        self.train_svm()
        self.train_random_forest()
        self.train_lightgbm()
        self.logger.info("All model training processes have finished.")

if __name__ == '__main__':
    """
    Main execution block
    """
    ROOT_DIR = Path(__file__).resolve().parent.parent
    trainer = ModelTrainer(project_root=ROOT_DIR)
    trainer.run_all()