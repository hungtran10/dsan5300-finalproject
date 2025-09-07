LOGREG_GRID = {
    "weight_decay": [0, 1e-4, 1e-3, 1e-2],
    "lr":           [1e-3, 3e-3, 1e-2],
}

SVM_GRID = {
    "C":            [0.01, 0.1, 1, 10],
    "weight_decay": [0, 1e-4, 1e-3],
    "lr":           [1e-3, 3e-3],
}

MLP_GRID = {
    "hidden":       [32, 64, 128],
    "dropout":      [0.0, 0.2, 0.4],
    "weight_decay": [0, 1e-4, 1e-3],
    "lr":           [1e-3, 3e-3],
}

XGB_GRID = {
    "n_estimators":  [300, 600],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample":     [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_lambda":    [0, 1, 5],
    "reg_alpha":     [0, 0.5, 1],
}

GRIDS = {
    "logreg": LOGREG_GRID,
    "svm":    SVM_GRID,
    "mlp":    MLP_GRID,
    "xgb":    XGB_GRID,
}
