"""
Random hyper-parameter search on a 10-fold CV (60 k stratified sample).

Outputs per model
─────────────────
    • Visualizations/{model}_trial<N>_curve{_pre|_post}.png
    • Visualizations/{model}_trial<N>_violin{_pre|_post}.png
    • Visualizations/{model}_search{_pre|_post}.png
    • Visualizations/{model}_cv{_pre|_post}.csv
    • best_params.json                       (for trainer.py)

Example
~~~~~~~
python cj-scripts/tuner.py \
    --data ./Datasets/Encoded_Food_Inspections_v2.csv \
    --models mlp xgb --n-iter 20 --no-preinspection
"""
from __future__ import annotations
from pathlib import Path
import json, argparse, random, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch, torch.nn as nn, torch.optim as optim

sns.set_style("whitegrid")

# ─────────────── project paths ──────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent            # …/cj-scripts
ROOT_DIR   = SCRIPT_DIR.parent                          # project root
VIS_DIR    = ROOT_DIR / "Visualizations"
VIS_DIR.mkdir(exist_ok=True, parents=True)

# ─────────────── local modules ─────────────────────────────────
from dataset import InspectionsDataset
from nets    import LogisticNet, LinearSVMNet, MLPNet
from grids   import GRIDS

# ──────────────────── plotting helpers ─────────────────────────
def save_learning_curve(fold_losses, model, trial, params, suffix):
    n_ep     = len(fold_losses[0][0])
    tr_avg   = np.mean([tl for tl, _ in fold_losses], axis=0)
    vl_avg   = np.mean([vl for _, vl in fold_losses], axis=0)

    plt.figure(figsize=(4, 3))
    plt.plot(range(1, n_ep+1), tr_avg, label="train")
    plt.plot(range(1, n_ep+1), vl_avg, label="val")
    plt.xlabel("epoch");  plt.ylabel("loss")
    plt.title(f"{model} trial {trial}\n{params}", fontsize=8)
    plt.legend(); plt.tight_layout()

    fname = VIS_DIR / f"{model}_trial{trial:02d}_curve{suffix}.png"
    plt.savefig(fname, dpi=120); plt.close()
    print("   ↳ saved", fname.relative_to(ROOT_DIR))

def save_violin(fold_scores, model, trial, params, suffix):
    plt.figure(figsize=(3, 3))
    sns.violinplot(y=fold_scores, inner=None, color="skyblue")
    sns.boxplot(y=fold_scores, width=.15, color="grey")
    plt.ylabel("AUC");  plt.ylim(0, 1)
    plt.title(f"{model} trial {trial}", fontsize=9)
    plt.tight_layout()

    fname = VIS_DIR / f"{model}_trial{trial:02d}_violin{suffix}.png"
    plt.savefig(fname, dpi=120); plt.close()
    print("   ↳ saved", fname.relative_to(ROOT_DIR))

# ─────────────────── evaluation helpers ────────────────────────
def evaluate_torch(net_cls, X, y, params, cv, model, trial, suffix):
    aucs, fold_losses = [], []
    for tr_idx, vl_idx in cv.split(X, y):
        net = net_cls(X.shape[1],
                      **{k: params[k] for k in ("hidden", "dropout") if k in params})
        opt  = optim.Adam(net.parameters(), lr=params["lr"],
                          weight_decay=params.get("weight_decay", 0))
        crit = nn.BCEWithLogitsLoss()

        Xtr = torch.tensor(X[tr_idx]).float()
        ytr = torch.tensor(y[tr_idx]).float().unsqueeze(1)
        Xvl = torch.tensor(X[vl_idx]).float()
        yvl = torch.tensor(y[vl_idx]).float().unsqueeze(1)

        tr_l, vl_l = [], []
        for _ in range(6):                       # mini budget
            net.train();  opt.zero_grad()
            loss = crit(net(Xtr), ytr); loss.backward(); opt.step()
            tr_l.append(loss.item())

            net.eval()
            with torch.no_grad():
                vl_logits = net(Xvl)
                vl_l.append(crit(vl_logits, yvl).item())

        fold_losses.append((tr_l, vl_l))
        with torch.no_grad():
            proba = torch.sigmoid(vl_logits).numpy()
        aucs.append(roc_auc_score(yvl, proba))

    save_learning_curve(fold_losses, model, trial, params, suffix)
    return float(np.mean(aucs))

def evaluate_sklearn(model_cls, X, y, params, cv, model, trial, suffix):
    aucs = []
    for tr_idx, vl_idx in cv.split(X, y):
        mdl = model_cls(**params)
        mdl.fit(X[tr_idx], y[tr_idx])
        proba = mdl.predict_proba(X[vl_idx])[:, 1]
        aucs.append(roc_auc_score(y[vl_idx], proba))

    save_violin(aucs, model, trial, params, suffix)
    return float(np.mean(aucs))

# ─────────────────────────── main ──────────────────────────────
def main(args):
    suffix = "_pre" if args.preinspection else "_post"

    ds   = InspectionsDataset(args.data, verbose=args.verbose)
    Xnp  = ds.X.numpy()
    ynp  = ds.y.numpy().ravel()

    # balanced 60 k sample
    idx0 = np.random.choice(np.where(ynp == 0)[0], 30_000, replace=False)
    idx1 = np.random.choice(np.where(ynp == 1)[0], 30_000, replace=False)
    samp = np.concatenate([idx0, idx1]);  np.random.shuffle(samp)
    Xs, ys = Xnp[samp], ynp[samp]

    cv = StratifiedKFold(10, shuffle=True, random_state=42)
    best_params = {}

    for model in args.models:
        grid     = GRIDS[model]
        if model == "xgb":
            sampler  = list(ParameterSampler(grid, n_iter=5, random_state=42))
        else:    
            sampler  = list(ParameterSampler(grid, n_iter=args.n_iter, random_state=42))
        results  = []

        print(f"\n{model} — {len(sampler)} trials")

        for trial_id, hp in enumerate(sampler, 1):
            if model == "logreg":
                auc = evaluate_torch(LogisticNet,  Xs, ys, hp, cv, model, trial_id, suffix)
            elif model == "svm":
                auc = evaluate_torch(LinearSVMNet, Xs, ys, hp, cv, model, trial_id, suffix)
            elif model == "mlp":
                auc = evaluate_torch(MLPNet,       Xs, ys, hp, cv, model, trial_id, suffix)
            elif model == "rf":
                auc = evaluate_sklearn(RandomForestClassifier, Xs, ys, hp, cv, model, trial_id, suffix)
            else:  # xgb
                auc = evaluate_sklearn(XGBClassifier,          Xs, ys, hp, cv, model, trial_id, suffix)

            results.append({**hp, "auc": auc})
            print(f"   trial {trial_id:02d}/{len(sampler)}  AUC={auc:.3f}")

        # ---------- per-model summary files ----------
        df = pd.DataFrame(results).sort_values("auc", ascending=False)
        csv_path = VIS_DIR / f"{model}_cv{suffix}.csv"
        df.to_csv(csv_path, index=False)
        print("   ↳ saved", csv_path.relative_to(ROOT_DIR))

        plt.figure(figsize=(4, 3))
        plt.plot(df["auc"].values, marker="o")
        plt.title(f"{model} CV-AUC");  plt.ylabel("AUC");  plt.xlabel("trial")
        plt.tight_layout()
        png_path = VIS_DIR / f"{model}_search{suffix}.png"
        plt.savefig(png_path, dpi=120);  plt.close()
        print("   ↳ saved", png_path.relative_to(ROOT_DIR))

        best_params[model] = df.iloc[0].drop("auc").to_dict()
        print(f"{model} best AUC = {df.iloc[0]['auc']:.3f}")

    # ---------- JSON for trainer.py ----------
    with open("best_params.json", "w") as fh:
        json.dump(best_params, fh, indent=2)
    print("\nbest_params.json written")

# ───────────────────────── CLI ─────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    required=True)
    ap.add_argument("--models",  nargs="+", default=["logreg","svm","mlp","xgb"])
    ap.add_argument("--n-iter",  type=int, default=15)
    ap.add_argument("--verbose", action="store_true")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--preinspection",     dest="preinspection", action="store_true")
    grp.add_argument("--no-preinspection",  dest="preinspection", action="store_false")
    ap.set_defaults(preinspection=True)
    main(ap.parse_args())
