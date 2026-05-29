"""
🧬 DATA AUGMENTATION & CLASS BALANCING - VERSION FINALE
✅ FIX UPDRS column + TOUS les bugs
✅ 100% Compatible pipeline existant
"""

from . import config
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DataAugmentor:
    def __init__(self, dataset_identifier):
        self.dataset_id = dataset_identifier
        self.original_data = None
        self.diagnostic = None
        self.feature_columns = None
        self.augmented_data = None
        
    def load_original_data(self):
        """📂 Charge et nettoie données"""
        ts_file = f"{config.output_files_dir}{self.dataset_id}_data_time_series.csv"
        diag_file = f"{config.input_files_dir}{self.dataset_id}_diagnostic.csv"
        
        # Vérification fichiers
        if not all(os.path.exists(f) for f in [ts_file, diag_file]):
            missing = [f for f in [ts_file, diag_file] if not os.path.exists(f)]
            raise FileNotFoundError(f"❌ Fichiers manquants: {missing}")
        
        # Chargement
        self.original_data = pd.read_csv(ts_file, dtype={"ID": str})
        self.diagnostic = pd.read_csv(diag_file, dtype={'UPDRS': 'int32', "ID": str})
        
        # 🔧 Nettoyage features numériques
        numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns.drop(['index'], errors='ignore')
        self.feature_columns = [col for col in numeric_cols if col != 'ID']
        
        # Conversion float
        for col in self.feature_columns:
            self.original_data[col] = pd.to_numeric(self.original_data[col], errors='coerce').fillna(0)
        
        # Alignement IDs
        valid_ids = np.intersect1d(self.original_data['ID'].unique(), self.diagnostic['ID'].unique())
        self.original_data = self.original_data[self.original_data['ID'].isin(valid_ids)]
        self.diagnostic = self.diagnostic[self.diagnostic['ID'].isin(valid_ids)]
        
        print(f"✅ {len(valid_ids)} vidéos | Features: {self.feature_columns}")
        return self
    
    def analyze_class_distribution(self):
        """📊 Distribution UPDRS"""
        class_counts = self.diagnostic['UPDRS'].value_counts().sort_index()
        self.target_count = 100
        
        print("\n📊 DISTRIBUTION ORIGINALE:")
        print("-" * 35)
        for cls in range(4):
            count = class_counts.get(cls, 0)
            pct = count/len(self.diagnostic)*100
            print(f"  UPDRS {cls}: {count:3d} ({pct:5.1f}%)")
        print(f"\n🎯 Cible: {self.target_count} par classe")
    
    def augment_video_features(self, features_array):
        """🔧 Augmente SEULEMENT les features numériques"""
        if len(features_array) == 0:
            return features_array
            
        features = features_array.astype(np.float64)
        method = np.random.choice(['noise', 'smooth', 'mix'])
        
        try:
            if method == 'noise':
                noise = np.random.normal(0, 0.05, features.shape)
                features += noise
                
            elif method == 'smooth':
                for i in range(features.shape[1]):
                    features[:, i] = gaussian_filter1d(features[:, i], sigma=1.0)
                    
            elif method == 'mix':
                noise = np.random.normal(0, 0.03, features.shape)
                features += noise
                for i in range(features.shape[1]):
                    features[:, i] = gaussian_filter1d(features[:, i], sigma=0.7)
            
            return np.clip(features,
                            np.percentile(features,1),
                            np.percentile(features,99))
        except:
            return features + np.random.normal(0, 0.01, features.shape)
    



    def create_augmented_videos(self, n_augmentations=5):
        """🎬 ÉQUILIBRAGE PARFAIT - EXACTEMENT target_count par classe"""
        print(f"\n🔄 Équilibrage parfait → {self.target_count} vidéos/classe...")
        
        video_list = []
        label_dict = {}
        
        # 1️⃣ TOUTES LES ORIGINALES d'abord
        print("✅ 1/3 Originales conservées...")
        for _, diag_row in self.diagnostic.iterrows():
            vid_id = diag_row['ID']
            video_df = self.original_data[self.original_data['ID'] == vid_id].copy()
            video_df['ID'] = vid_id
            video_list.append(video_df)
            label_dict[vid_id] = diag_row['UPDRS']
        
        # 2️⃣ AUGMENTATIONS pour chaque classe jusqu'à target_count EXACT
        print("✅ 2/3 Augmentations ciblées...")
        for cls in range(4):
            # Compter actuelles pour cette classe
            current_count = sum(1 for label in label_dict.values() if label == cls)
            needed = self.target_count - current_count
            
            print(f"  UPDRS {cls}: {current_count} → besoin {needed}")
            
            if needed <= 0:
                print(f"  UPDRS {cls}: OK (déjà {current_count})")
                continue
            
            # Sources disponibles pour cette classe
            class_sources = self.diagnostic[self.diagnostic['UPDRS'] == cls]['ID'].tolist()
            
            for i in range(needed):
                # Source aléatoire de cette classe
                source_id = np.random.choice(class_sources)
                source_df = self.original_data[self.original_data['ID'] == source_id]
                
                # 🔧 AUGMENTATION
                features_orig = source_df[self.feature_columns].values
                features_aug = self.augment_video_features(features_orig)
                
                # Nouveau ID unique
                aug_count = sum(1 for vid in label_dict if vid.startswith(source_id + '_aug'))
                new_id = f"{source_id}_aug{cls}_{i+1:03d}"
                
                # Nouveau DataFrame
                aug_df = source_df.copy()
                aug_df[self.feature_columns] = features_aug
                aug_df['ID'] = new_id
                aug_df['index'] = range(len(aug_df))
                
                video_list.append(aug_df)
                label_dict[new_id] = cls
            
            final_count = sum(1 for label in label_dict.values() if label == cls)
            print(f"  UPDRS {cls}: FINAL {final_count} ✅")
        
        # 3️⃣ FUSION
        print("✅ 3/3 Fusion finale...")
        self.augmented_data = pd.concat(video_list, ignore_index=True)
        self.label_dict = label_dict
        
        # VÉRIFICATION FINALE
        final_counts = {cls: sum(1 for v in label_dict.values() if v == cls) for cls in range(4)}
        print("\n📊 VÉRIFICATION ÉQUILIBRAGE:")
        for cls in range(4):
            status = "✅" if final_counts[cls] == self.target_count else "❌"
            print(f"  UPDRS {cls}: {final_counts[cls]:3d} {status}")
        
        print(f"🎉 TOTAL: {len(label_dict)} vidéos")
        return self
    
    def save_balanced_dataset(self):
        """💾 Sauvegarde COMPLÈTE - TOUS les fichiers nécessaires"""
        # Fichiers time series + diagnostic
        ts_file = f"{config.output_files_dir}{self.dataset_id}_balanced_data_time_series.csv"
        diag_file = f"{config.input_files_dir}{self.dataset_id}_balanced_diagnostic.csv"
        
        # Fichiers FPS et frame rate (obligatoires pour extract_features)
        fps_file = f"{config.output_files_dir}{self.dataset_id}_balanced_fps_videos.csv"
        frame_file = f"{config.log_files_dir}{self.dataset_id}_balanced_frame_rate_processed_videos.csv"
        
        # 1. Time series
        ts_save = self.augmented_data.drop(columns=['UPDRS'], errors='ignore')
        ts_save.to_csv(ts_file, index=False)
        
        # 2. Diagnostic
        diag_save = pd.DataFrame([
            {'ID': vid_id, 'UPDRS': label} 
            for vid_id, label in self.label_dict.items()
        ]).sort_values('ID')
        diag_save.to_csv(diag_file, index=False)
        
        # 3. FPS (copie des originaux + dummy pour augmentés)
        original_fps = pd.read_csv(f"{config.output_files_dir}{self.dataset_id}_fps_videos.csv")
        fps_df = original_fps.copy()
        
        # Ajout FPS dummy pour vidéos augmentées (même FPS que source)
        aug_fps = []
        for vid_id, label in self.label_dict.items():
            if '_aug' in vid_id and vid_id not in fps_df['ID'].values:
                # Trouver source FPS
                source_id = vid_id.split('_aug')[0]
                source_fps = fps_df[fps_df['ID'] == source_id]['FPS'].iloc[0]
                aug_fps.append({'ID': vid_id, 'FPS': source_fps})
        
        aug_fps_df = pd.DataFrame(aug_fps)
        fps_df = pd.concat([fps_df, aug_fps_df], ignore_index=True)
        fps_df.to_csv(fps_file, index=False)
        
        # 4. Frame rate (95% pour toutes)
        frame_df = pd.DataFrame({
            'ID': list(self.label_dict.keys()),
            'Percentage': 95.0
        })
        frame_df.to_csv(frame_file, index=False)
        
        print(f"\n✅ 4 FICHIERS COMPLÈTS:")
        print(f"   📈 Time series: {os.path.basename(ts_file)}")
        print(f"   🏷️  Diagnostic:  {os.path.basename(diag_file)}")
        print(f"   ⚡ FPS:          {os.path.basename(fps_file)}")
        print(f"   📊 Frame rate:   {os.path.basename(frame_file)}")

        return ts_file, diag_file, fps_file, frame_file
    
    def run_full_pipeline(self, n_augmentations=5):
        """🚀 EXÉCUTION COMPLÈTE"""
        print(f"🎯 DATA AUGMENTATION {self.dataset_id}")
        print("=" * 50)
        
        self.load_original_data()
        self.analyze_class_distribution()
        self.create_augmented_videos(n_augmentations)
        return self.save_balanced_dataset()

def augment_dataset(dataset_identifier="fis", n_augmentations=5):
    """🎯 UNE LIGNE - Retourne SEULEMENT 2 fichiers principaux"""
    augmenter = DataAugmentor(dataset_identifier)
    augmenter.run_full_pipeline(n_augmentations)
    # Retourne SEULEMENT les 2 principaux
    ts_file = f"{config.output_files_dir}{dataset_identifier}_balanced_data_time_series.csv"
    diag_file = f"{config.input_files_dir}{dataset_identifier}_balanced_diagnostic.csv"
    return ts_file, diag_file  # ← UNIQUEMENT 2 fichiers                                                                      Donne-moi un script Python complet et compatible avec mon pipeline de classification vidéo , et comment le implémanter pas a pas. et Le code doit être clair, bien structuré.
