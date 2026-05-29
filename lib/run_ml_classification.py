import pandas as pd
import numpy as np
from . import config
from . import ml_models as m2t
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')


def classify_video_kfold(dataset_identifier, features_type, n_splits=5):
    
    """🚀 MAIN FUNCTION: K-Fold CV"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.my_random_state)
    _classify_video_base(dataset_identifier, features_type, skf, f"kfold_{n_splits}")


def _classify_video_base(dataset_identifier, features_type, cv_splitter, cv_name):
    """Core classification logic"""
    os.makedirs(f"{config.results_files_dir}confusion_matrix", exist_ok=True)

    # Load diagnostic
    print(f"📂 Loading diagnostic...")
    y_diagnostic = pd.read_csv(
        config.input_files_dir + dataset_identifier + "_diagnostic.csv",
        dtype={'UPDRS': np.int32, "ID": str}
    ).sort_values(by=['ID']).drop_duplicates().reset_index(drop=True)

    # Load features
    features_file = config.output_files_dir + dataset_identifier + "_" + features_type + "_features.csv"
    print(f"📂 Loading: {features_file}")

    if not os.path.exists(features_file):
        print(f"❌ ERROR: {features_file} missing")
        return

    features = pd.read_csv(features_file)
    print(f"📊 Features loaded: {features.shape}")

    if 'Unnamed: 0' in features.columns:
        features = features.rename(columns={"Unnamed: 0": "ID"})

    if features.empty or 'ID' not in features.columns:
        print(f"❌ ERROR: Invalid features for {features_type}")
        return

    # Filter matching IDs
    list_id_y_diagnostic = np.unique(y_diagnostic["ID"])
    list_id_features = np.unique(features["ID"])
    features = features[features.ID.isin(list_id_y_diagnostic)]
    y_diagnostic = y_diagnostic[y_diagnostic.ID.isin(list_id_features)]

    print(f"🔗 Final data: Features={len(features)}, Diagnostic={len(y_diagnostic)}")

    if len(features) == 0 or features.drop(columns=['ID']).shape[1] == 0:
        print(f"❌ No valid features for {features_type}")
        return

    # Prepare data
    X = StandardScaler().fit_transform(features.drop(columns=['ID']))
    y = y_diagnostic["UPDRS"].values
    unique_labels = np.unique(y)

    print(f"✅ {X.shape} | {len(unique_labels)} classes | {cv_name}")

    # Results storage
    final_result = pd.DataFrame()
    conf_matrix_data = {m: {'y_true': [], 'y_pred': []} for m in m2t.models.keys()}

    # Cross-validation
    for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
        print(f"  Fold {fold+1}/{cv_splitter.get_n_splits()}", end=' ')
        for model_name, model_class in m2t.models.items():
            model_params = m2t.fast_model_params.get(model_class, {})

            # ✅ Correction: filtrer default_acc
            model_params_safe = {k: v for k, v in model_params.items() if k != "default_acc"}

            if model_name != "KNeighborsClassifier":
                model_params_safe['random_state'] = config.my_random_state

            model = model_class(**model_params_safe)

            # Fit & predict
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])

            # Metrics
            acc = accuracy_score(y[test_idx], y_pred)
            substraction = np.abs(y[test_idx] - y_pred)
            acceptable_acc = np.mean((substraction <= 1).astype(float))

            try:
                kappa = cohen_kappa_score(y[test_idx], y_pred, labels=unique_labels)
                f1 = f1_score(y[test_idx], y_pred, average='weighted', labels=unique_labels)
                mcc = matthews_corrcoef(y[test_idx], y_pred)
            except:
                kappa = f1 = mcc = 0.0

            # Store
            data = {
                'Fold': fold, 'Model': model_name, 'CV': cv_name,
                'Accuracy': acc, 'Acceptable_Acc': acceptable_acc,
                'Kappa': kappa, 'F1': f1, 'MCC': mcc
            }
            final_result = pd.concat([final_result, pd.DataFrame([data])], ignore_index=True)

            conf_matrix_data[model_name]['y_true'].extend(y[test_idx])
            conf_matrix_data[model_name]['y_pred'].extend(y_pred)

    # Confusion matrices
    for model_name, data in conf_matrix_data.items():
        y_true_total = np.array(data['y_true'])
        y_pred_total = np.array(data['y_pred'])
        cm = confusion_matrix(y_true_total, y_pred_total, labels=unique_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f"{dataset_identifier}_{features_type}_{model_name}_{cv_name}")
        plt.tight_layout()
        plt.savefig(
            f"{config.results_files_dir}confusion_matrix/{dataset_identifier}_{features_type}_{model_name}_{cv_name}.png",
            dpi=150, bbox_inches='tight'
        )
        plt.close()

    # Save results
    summary_df = final_result.groupby(['Model']).agg({
        'Accuracy': 'mean', 'Acceptable_Acc': 'mean',
        'Kappa': 'mean', 'F1': 'mean', 'MCC': 'mean'
    }).round(4)

    summary_filename = f"{config.results_files_dir}{dataset_identifier}_{features_type}_{cv_name}_summary.csv"
    summary_df.to_csv(summary_filename, sep=';')

    print(f"\n✅ {cv_name} COMPLETE!")
    print(f"📊 Summary: {summary_filename}")
    print("\n🏆 TOP 3:")
    print(summary_df.nlargest(3, 'Accuracy')[['Accuracy', 'Acceptable_Acc', 'Kappa']].round(3))
