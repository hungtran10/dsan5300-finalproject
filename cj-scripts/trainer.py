"""
Train all models on the FULL inspection data using tuner-selected hyper-params.

Creates
───────
Visualizations/
    ├─ {model}_loss.png          
    ├─ {model}_roc.png
    └─ {model}_confusion.png
Weights/
    ├─ {model}.pth   (Torch nets)
    └─ {model}.joblib
summary.csv

python cj-scripts/trainer.py --data ./Datasets/Encoded_Food_Inspections_v2.csv --device cuda
"""
# ───────────────────────── imports ───────────────────────────
from pathlib import Path
import json, argparse, joblib, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import InspectionsDataset
from nets    import LogisticNet, LinearSVMNet, MLPNet
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# ───────────────────── directories ───────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent            # …/cj-scripts
ROOT_DIR   = SCRIPT_DIR.parent                          # project root
VIS_DIR    = ROOT_DIR / "Visualizations"
VIS_DIR.mkdir(exist_ok=True, parents=True)
WEIGHTS_DIR    = ROOT_DIR / "Weights"
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)

# ───────────────────── misc helpers ──────────────────────────
class SquaredHingeLoss(nn.Module):
    def forward(self, margin, y):
        y_signed = y * 2 - 1
        return torch.mean(torch.clamp(1 - y_signed * margin, min=0) ** 2)

class EarlyStopping:
    def __init__(self, patience:int=5):
        self.patience = patience
        self.best     = float("inf")
        self.count    = 0
    def step(self, metric:float) -> bool:
        if metric < self.best - 1e-4:
            self.best  = metric
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience

def acc_auc_from_prob(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, (y_prob > 0.5).astype(int))
    return acc, auc

def _log_epoch(model_name: str, epoch: int, tr_loss: float, val_auc: float) -> None:
    """Pretty one-liner progress report."""
    print(f"[{model_name}] epoch {epoch:02d} | tr-loss {tr_loss:.3f} • val-AUC {val_auc:.3f}")

# ───────────────────── training per model ────────────────────
def train_full(model:str, ds:InspectionsDataset, hp:dict, device:str):
    """
    Train 1 model.  
    Returns (fitted_model, test_acc, test_auc, y_true, y_prob, loss_hist)
    loss_hist is (train_losses, val_losses) or (None, None) for sklearn/XGB.
    """
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

    # 70 / 15 / 15 split
    n = len(ds); n_tr = int(.70*n); n_va = int(.15*n); n_te = n - n_tr - n_va
    tr_ds, va_ds, te_ds = random_split(
        ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42))

    dl_kw = dict(batch_size=8192, num_workers=4, pin_memory=(dev.type=="cuda"))
    tr_dl = DataLoader(tr_ds, shuffle=True,  **dl_kw)
    va_dl = DataLoader(va_ds, shuffle=False, **dl_kw)
    te_dl = DataLoader(te_ds, shuffle=False, **dl_kw)

    # convenience numpy slices for classical models
    Xtr_np = ds.X[tr_ds.indices].numpy()
    ytr_np = ds.y[tr_ds.indices].numpy().ravel()

    # ───── build model ────────────────────────────────────────
    if model == "logreg":
        net  = LogisticNet(ds.X.shape[1]).to(dev)
        crit = nn.BCEWithLogitsLoss()
        opt  = optim.Adam(net.parameters(), lr=hp["lr"],
                          weight_decay=hp.get("weight_decay",0))
    elif model == "svm":
        net  = LinearSVMNet(ds.X.shape[1]).to(dev)
        crit = SquaredHingeLoss()
        opt  = optim.Adam(net.parameters(), lr=hp["lr"],
                          weight_decay=hp.get("weight_decay",0))
    elif model == "mlp":
        net = MLPNet(
            ds.X.shape[1],
            hidden = int(hp.get("hidden", 64)),   # default 64 units
            dropout= float(hp.get("dropout", 0.0))
            ).to(dev)
        crit = nn.BCEWithLogitsLoss()
        opt  = optim.Adam(net.parameters(), lr=hp.get("lr",0),
                          weight_decay=hp.get("weight_decay",0))
    # --- inside train_full -------------------------------------------------
    elif model == "rf":
        int_params = {k: int(v) for k in ("n_estimators",
                                        "max_depth",
                                        "min_samples_leaf",
                                        "min_samples_split")
                    if k in hp}
        rf_params = hp.copy(); rf_params.update(int_params)

        net = RandomForestClassifier(**rf_params,
                                    n_jobs=-1,
                                    random_state=42)
        net.fit(Xtr_np, ytr_np)
        return _eval_sklearn(net, ds, te_ds, model)
    elif model == "xgb":
        # ------------------------------------------------------------------
        # Cast the integer-like hyper-parameters back to int ----------------
        # ------------------------------------------------------------------
        int_keys = ("n_estimators", "max_depth")
        xgb_params = hp.copy()                       # hp comes from best_params.json
        for k in int_keys:
            if k in xgb_params:
                xgb_params[k] = int(xgb_params[k])   # 600.0 → 600, 5.0 → 5

        # If you later add other integer params (e.g. 'max_leaves',
        # 'min_child_weight'), just include them in int_keys.

        net = XGBClassifier(
            tree_method      = "gpu_hist" if dev.type == "cuda" else "hist",
            use_label_encoder= False,
            eval_metric      = "logloss",
            random_state     = 42,
            **xgb_params                       # now all ints where required
        )
        net.fit(Xtr_np, ytr_np)
        return _eval_sklearn(net, ds, te_ds, model)

    else:
        raise ValueError(model)

    # ─────── Torch training loop ───────────────────────────────
    tr_losses, va_losses = [], []
    stopper = EarlyStopping(5)

    for epoch in range(1, 12):                       # up-to 100 epochs
        # 1) -------- train phase --------
        net.train(); running = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward(); opt.step()
            running += loss.item() * len(xb)
        tr_epoch_loss = running / len(tr_ds)
        tr_losses.append(tr_epoch_loss)

        # 2) -------- validation phase --------
        net.eval(); preds, true = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                prob = torch.sigmoid(net(xb))
                preds.append(prob.cpu()); true.append(yb.cpu())
        y_prob = torch.vstack(preds).numpy()
        y_true = torch.vstack(true ).numpy()
        val_auc = roc_auc_score(y_true, y_prob)
        va_losses.append(1 - val_auc)

        # ---- live progress line ----
        _log_epoch(model, epoch, tr_epoch_loss, val_auc)

        # 3) -------- early-stop check --------
        if stopper.step(1 - val_auc):
            break

    # ─────── evaluate on test split ────────────────────────────
    net.eval(); preds, true = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            prob   = torch.sigmoid(net(xb))
            preds.append(prob.cpu()); true.append(yb.cpu())
    y_prob = torch.vstack(preds).numpy().ravel()
    y_true = torch.vstack(true ).numpy().ravel()
    acc, auc = acc_auc_from_prob(y_true, y_prob)
    return net, acc, auc, y_true, y_prob, (tr_losses, va_losses)

# ─────────── helper for classical models ─────────────────────
def _eval_sklearn(mdl, ds, te_ds, model_name):
    Xte = ds.X[te_ds.indices].numpy()
    yte = ds.y[te_ds.indices].numpy().ravel()
    ypr = mdl.predict_proba(Xte)[:,1]
    acc, auc = acc_auc_from_prob(yte, ypr)
    return mdl, acc, auc, yte, ypr, (None, None)

# ────────────────────────── main CLI ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   required=True)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    args = ap.parse_args()

    # hyper-params chosen by tuner
    with open("best_params.json") as fh:
        best = json.load(fh)

    ds = InspectionsDataset(args.data)
    summary = []

    for model, hp in best.items():
        print(f"\n {model}  params={hp}")
        mdl, acc, auc, y_true, y_prob, losses = train_full(model, ds, hp, args.device)
        summary.append({"model":model, "accuracy":acc, "auc_roc":auc})

        # ── save ROC ──
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(4,4))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'k--',alpha=.4)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"{model.upper()} ROC"); plt.legend()
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"{model}_pre_roc.png", dpi=120); plt.close()

        # ── save Confusion Matrix ──
        cm = confusion_matrix(y_true, (y_prob>0.5).astype(int))
        ConfusionMatrixDisplay(cm).plot(colorbar=False)
        plt.title(f"{model.upper()} Confusion")
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"{model}_pre_confusion.png", dpi=120); plt.close()

        # ── save Loss curve (Torch models only) ──
        tr_l, va_l = losses
        if tr_l is not None:
            plt.figure(figsize=(4,3))
            plt.plot(tr_l, label="train")
            plt.plot(va_l, label="val")
            plt.xlabel("epoch"); plt.ylabel("loss / (1-AUC)")
            plt.title(f"{model.upper()} loss"); plt.legend()
            plt.tight_layout()
            plt.savefig(VIS_DIR / f"{model}_pre_loss.png", dpi=120); plt.close()

        # ── save weights ──
        if model in {"rf","xgb"}:
            joblib.dump(mdl, WEIGHTS_DIR / f"{model}_pre.joblib")
        else:
            torch.save(mdl.state_dict(), WEIGHTS_DIR / f"{model}_pre.pth")

        print(f"   → acc={acc:.3f}  auc={auc:.3f}")

    pd.DataFrame(summary).round(4).to_csv("summary_pre.csv", index=False)
    print("\nsummary_pre.csv written")

if __name__ == "__main__":
    main()
