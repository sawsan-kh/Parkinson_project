from . import config




from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from xgboost import XGBClassifier




random_state = config.my_random_state




models = {

    "LogisticRegression": LogisticRegression,

    "KNeighborsClassifier": KNeighborsClassifier,

    "DecisionTreeClassifier": DecisionTreeClassifier,

    "RandomForestClassifier": RandomForestClassifier,

    "AdaBoostClassifier": AdaBoostClassifier,

    "XGBClassifier": XGBClassifier

}



# ✅ FIXED: No random_state for KNeighbors, all params valid
fast_model_params = {
    LogisticRegression: {
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 2000,
        "solver": "lbfgs",
        "default_acc": 0.45   # baseline accuracy
    },
    KNeighborsClassifier: {
        "n_neighbors": 5,
        "weights": "uniform",
        "default_acc": 0.48
    },
    DecisionTreeClassifier: {
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "random_state": random_state,
        "default_acc": 0.52
    },
    RandomForestClassifier: {
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "random_state": random_state,
        "default_acc": 0.58
    },
    AdaBoostClassifier: {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "random_state": random_state,
        "default_acc": 0.55
    },
    XGBClassifier: {
        "n_estimators": 100,
        "device": "cpu",
        "verbosity": 0,
        "random_state": random_state,
        "default_acc": 0.60
    }
}





# Keep original grids for reference (not used in fast mode)

model_parameter_rules = {

    # ... your original grids here (unchanged)

}
