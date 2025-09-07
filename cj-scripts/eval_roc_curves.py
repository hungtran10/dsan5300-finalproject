"""
Evaluate saved ‘pre’ and ‘post’ models, draw combined ROC curves **and**
save a feature-importance bar chart for every model.

Creates
───────
Visualizations/
    pre_combined_roc.png          post_combined_roc.png
    logreg_pre_importance.png     logreg_post_importance.png
    svm_pre_importance.png        …

python cj-scripts/eval_roc_curves.py --pre-data ./Datasets/Encoded_PreInspectionOnly_v2.csv --post-data ./Datasets/Encoded_Food_Inspections_v2.csv --device cuda
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import torch, torch.nn as nn
from torch.utils.data import random_split, DataLoader
import joblib, xgboost as xgb
from dataset import InspectionsDataset
from nets    import LogisticNet, LinearSVMNet, MLPNet

ROOT        = Path(__file__).resolve().parent.parent
VIS_DIR     = ROOT / "Visualizations"
WEIGHTS_DIR = ROOT / "Weights"
VIS_DIR.mkdir(exist_ok=True,  parents=True)

# ───────────────────────── helpers ──────────────────────────
def get_test_loader(ds, batch=8192, device="cpu"):
    n = len(ds); n_tr = int(.70*n); n_va = int(.15*n); n_te = n - n_tr - n_va
    _, _, te_ds = random_split(
        ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42)
    )
    dl = DataLoader(te_ds, batch_size=batch, shuffle=False,
                    num_workers=4, pin_memory=(device=="cuda"))
    return dl, te_ds

@torch.no_grad()
def predict_torch(net, loader, device):
    net.eval(); out = []
    for xb, _ in loader:
        out.append(torch.sigmoid(net(xb.to(device))).cpu())
    return torch.vstack(out).numpy().ravel()

# ------------------------------- importance extractors -----------------------
def importances_linear(weight: np.ndarray) -> np.ndarray:
    """|w| for logistic / linear-SVM (shape = n_features)"""
    return np.abs(weight.squeeze())

def importances_mlp(net: nn.Module) -> np.ndarray:
    """|W| of first dense layer"""
    first_linear = next(m for m in net.modules() if isinstance(m, nn.Linear))
    return np.abs(first_linear.weight.detach().cpu().numpy()).mean(axis=0)

# ───────────────────── evaluate one model  ───────────────────
def evaluate(model_name:str, weight_file:Path, ds:InspectionsDataset,
             loader, te_subset, device, group_suffix:str):
    Xte_np = ds.X[te_subset.indices].numpy()
    y_true = ds.y[te_subset.indices].numpy().ravel()
    feats  = ds.feature_names

    # ------------------------------------------------------------------ Torch
    if model_name == "logreg":
        net = LogisticNet(ds.X.shape[1]).to(device)
        net.load_state_dict(torch.load(weight_file, map_location=device))
        y_prob = predict_torch(net, loader, device)

        # OLD (raises RuntimeError)
        # imp = importances_linear(net.fc.weight.cpu().numpy())

        # NEW
        imp = importances_linear(net.fc.weight.detach().cpu().numpy())

    elif model_name == "svm":
        net = LinearSVMNet(ds.X.shape[1]).to(device)
        net.load_state_dict(torch.load(weight_file, map_location=device))
        y_prob = predict_torch(net, loader, device)

        # imp needs .detach() as well
        imp = importances_linear(net.fc.weight.detach().cpu().numpy())

    elif model_name == "mlp":
        with open(ROOT / "best_params.json") as fh:
            hp = json.load(fh)[model_name]
        net = MLPNet(
            ds.X.shape[1],
            hidden = int(hp.get("hidden",64)),
            dropout= float(hp.get("dropout",0.0))
        ).to(device)
        net.load_state_dict(torch.load(weight_file, map_location=device))
        y_prob = predict_torch(net, loader, device)
        imp    = importances_mlp(net)

    # ------------------------------------------------------------- XGB / RF
    else:                                 # xgb or rf (joblib)
        mdl = joblib.load(weight_file)
        y_prob = mdl.predict_proba(Xte_np)[:,1]
        imp    = mdl.feature_importances_

    # ---------- ROC
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    # ---------- save importance bar
    top_k = 20
    idx   = np.argsort(imp)[::-1][:top_k]
    plt.figure(figsize=(6,4))
    plt.barh(range(top_k), imp[idx][::-1])
    plt.yticks(range(top_k), [feats[i] for i in idx][::-1], fontsize=7)
    plt.xlabel("importance"); plt.title(f"{model_name} importance ({group_suffix})")
    plt.tight_layout()
    imp_path = VIS_DIR / f"{model_name}_{group_suffix}_importance.png"
    plt.savefig(imp_path, dpi=120); plt.close()
    print(f"   ↳ feature-importance saved → {imp_path.name}")

    return auc, fpr, tpr

# ─────────────────────────── CLI ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-data",  required=True)
    ap.add_argument("--post-data", required=True)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cpu")
    args = ap.parse_args()
    dev  = "cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu"

    setups = {
        "pre":  (args.pre_data,  "*_pre.*"),
        "post": (args.post_data, "*_post.*")
    }

    summary_rows = []
    for tag, (csv_path, glob_pat) in setups.items():
        ds          = InspectionsDataset(csv_path)
        loader, te  = get_test_loader(ds, device=dev)

        plt.figure(figsize=(5,5))
        for wfile in sorted(WEIGHTS_DIR.glob(glob_pat)):
            mdl_name = wfile.stem.replace(f"_{tag}","")
            auc,fpr,tpr = evaluate(mdl_name, wfile, ds, loader, te, dev, tag)
            summary_rows.append({"group":tag,"model":mdl_name,"auc":auc})
            plt.plot(fpr,tpr, lw=1.6, label=f"{mdl_name} (AUC={auc:.3f})")

        plt.plot([0,1],[0,1],'k--',alpha=.4)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"{tag.upper()} combined ROC")
        plt.legend(fontsize=8); plt.tight_layout()
        out_png = VIS_DIR / f"{tag}_combined_roc.png"
        plt.savefig(out_png, dpi=130); plt.close()
        print(f"combined ROC saved → {out_png.name}")

    print("\nAUC summary")
    print(pd.DataFrame(summary_rows).sort_values(["group","auc"],ascending=[True,False]).to_string(index=False))

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
