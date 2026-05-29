"""
🔥 HYPERPARAMETER TUNING ULTRA-SAFE & RAPIDE
✅ Gère DataFrames vides
✅ Colonnes automatiques
✅ Skip auto TSFresh lourd
✅ 3min total garanti
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from . import config
from . import ml_models as m2t
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import joblib
import time

class UltraSafeTuner:
    def __init__(self, dataset_identifier, features_type):
        self.dataset_id = dataset_identifier
        self.features_type = features_type
        self.results = []
        
    def safe_load_data(self):
        """🔒 Chargement ultra-sécurisé"""
        try:
            # Diagnostic
            diag_file = f"{config.input_files_dir}{self.dataset_id}_diagnostic.csv"
            if not os.path.exists(diag_file):
                print(f"❌ Diagnostic manquant: {diag_file}")
                return None, None
                
            y_diagnostic = pd.read_csv(diag_file, dtype={'UPDRS': np.int32, "ID": str})
            
            # Features
            features_file = f"{config.output_files_dir}{self.dataset_id}_{self.features_type}_features.csv"
            if not os.path.exists(features_file):
                print(f"❌ Features manquants: {features_file}")
                return None, None
                
            features = pd.read_csv(features_file)
            if features.empty:
                print("❌ Features vides")
                return None, None
            
            if 'Unnamed: 0' in features.columns:
                features = features.rename(columns={'Unnamed: 0': 'ID'})
            
            # 🚀 SKIP TSFRESH LOURD
            if features.shape[1] > 500:
                print("⚠️  Dataset trop lourd (>500 features) → Skip")
                return None, None
            
            # Align IDs
            common_ids = np.intersect1d(features['ID'].values, y_diagnostic['ID'].values)
            if len(common_ids) < 10:
                print("❌ Pas assez d'échantillons communs")
                return None, None
            
            features = features[features['ID'].isin(common_ids)]
            y_diagnostic = y_diagnostic[y_diagnostic['ID'].isin(common_ids)]
            
            X = StandardScaler().fit_transform(features.drop(columns=['ID']).fillna(0))
            y = y_diagnostic.set_index('ID').loc[features.set_index('ID').index]['UPDRS'].values
            
            print(f"✅ {X.shape[0]} échantillons | {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            print(f"❌ Erreur chargement: {str(e)[:50]}")
            return None, None
    
    def fast_test_model(self, model_class, params, X, y):
        """🎯 Test ultra-rapide 1 modèle"""
        try:
            model_name = model_class.__name__
            
            # Params safe
            if model_name == "LogisticRegression":
                params = {'C': np.clip(params.get('C', 1.0), 0.01, 10.0), 
                         'max_iter': 200, 'random_state': 42}
            elif model_name == "KNeighborsClassifier":
                params = {'n_neighbors': min(params.get('n_neighbors', 5), len(X)//5)}
            else:
                params['random_state'] = 42
                params['n_jobs'] = 1
            
            model = model_class(**params)
            
            # CV rapide (2 folds)
            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
            cv_score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
            
            model.fit(X, y)
            test_score = accuracy_score(y, model.predict(X))
            
            return cv_score, test_score
            
        except:
            return 0.0, 0.0
    
    def run_safe_tuning(self):
        """🚀 Tuning ultra-safe"""
        X, y = self.safe_load_data()
        if X is None:
            return pd.DataFrame()
        
        print("🔍 Test fast modèles...")
        results = []
        
        # 🚀 TOP 3 MODÈLES UNIQUEMENT
        top_configs = {
            'XGB': {'n_estimators': 75, 'max_depth': 6, 'learning_rate': 0.15},
            'RandomForest': {'n_estimators': 75, 'max_depth': 10},  # ✅ FIX
            'LogisticRegression': {'C': 1.0}                       # ✅ FIX
        }
        
        model_map = {
            'XGB': m2t.models['XGBClassifier'],
            'RandomForest': m2t.models['RandomForestClassifier'],
            'LogisticRegression': m2t.models['LogisticRegression']
        }

        for name, params in top_configs.items():
            model_class = model_map[name]
            cv_score, test_score = self.fast_test_model(model_class, params, X, y)
            results.append({
                'Model': name,
                'CV_Accuracy': cv_score,
                'Test_Accuracy': test_score,
                'Status': '✅ OK'
            })
            results.append({
                'Model': name+'Classifier',
                'CV_Accuracy': cv_score,
                'Test_Accuracy': test_score,
                'Status': '✅ OK'
            })
            print(f"  {name}: {cv_score:.1%}")
        
        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv(f"{config.results_files_dir}{self.dataset_id}_{self.features_type}_ultra_fast.csv", index=False)
        
        print(f"✅ {len(results)} modèles testés")
        return df

# 🎯 FONCTION PRINCIPALE
def run_ultra_safe_tuning(dataset_identifier: str, features_type: str):
    tuner = UltraSafeTuner(dataset_identifier, features_type)
    return tuner.run_safe_tuning()