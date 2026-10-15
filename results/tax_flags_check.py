# One-off check (15 Oct 2026): Random Forest on tax + filing + abatement features only, and stock returns of simple tax/filing red flags in the test years. Run: python results/tax_flags_check.py
import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import config
from train_models import TAX, FILING, ABATEMENT, design_matrix, fit, make_model, tune
df = pd.read_csv(config.DATA_DIR / "modeling_table.csv")
tr = (df.split=="train").to_numpy(); y = df.beat_sp500.to_numpy()
X,_ = design_matrix(df, TAX+FILING+ABATEMENT, "tree", tr)
Xtr = X[tr].reset_index(drop=True)
params, vauc, _ = tune("rf", Xtr, y[tr], df.fiscal_year[tr].reset_index(drop=True), False)
m = fit(make_model("rf", params, y[tr]), "rf", Xtr, y[tr])
te = df[~tr].copy(); te["p"] = m.predict_proba(X[~tr])[:,1]
print("params", params, "valid", round(vauc,3), "test AUC", round(roc_auc_score(te.beat_sp500, te.p),4))
te["rank"] = te.groupby("fiscal_year").p.rank(pct=True, method="first")
def stats(mask, label):
    g = te[mask].groupby("fiscal_year")
    ex = g.excess_ret_12m.mean()
    rng = np.random.default_rng(42); boots=[]
    for _, part in te[mask].groupby("fiscal_year"):
        e = part.excess_ret_12m.to_numpy(); boots.append(e[rng.integers(0,len(e),(2000,len(e)))].mean(1))
    b = np.mean(boots,0)
    print(f"{label:38s} n/yr {g.size().mean():6.0f} ret {g.ret_12m.mean().mean()*100:6.1f} excess {ex.mean()*100:6.1f} [{np.percentile(b,2.5)*100:.1f},{np.percentile(b,97.5)*100:.1f}] beat {g.beat_sp500.mean().mean()*100:5.1f} by yr {list((ex*100).round(1))}")
stats(te.p.notna(), "All stocks")
stats(te["rank"]<=0.1, "Tax+filing model bottom 10%")
stats(te["rank"]>0.9, "Tax+filing model top 10%")
# red flags
stats(te.nt_10k==1, "Late 10-K (NT 10-K)")
stats(te.nt_count>0, "Any late-filing notice")
stats(te.pretax_loss==1, "Pretax loss")
stats(te.pretax_missing==1, "No pretax income reported")
stats((te.pretax_loss==0)&(te.pretax_missing==0), "Profitable (positive pretax)")
q = te.cash_to_book_tax
stats(q.notna() & (q <= q.quantile(.1)), "Cash tax vs book tax, lowest 10%")
stats(q.notna() & (q >= q.quantile(.9)), "Cash tax vs book tax, highest 10%")
v = te.valuation_allowance_to_assets
stats(v.notna() & (v >= v.quantile(.9)), "Valuation allowance, highest 10%")
c = te.etr_change
stats(c.notna() & (c <= c.quantile(.1)), "ETR fell most (bottom 10%)")
stats(c.notna() & (c >= c.quantile(.9)), "ETR rose most (top 10%)")
# within size: small caps only, does tax model still separate?
for qq in [1,3,5]:
    s = te.size_quintile==qq
    r = te[s].groupby("fiscal_year").p.rank(pct=True)
    lo = te[s][r<=0.2].excess_ret_12m.mean(); hi = te[s][r>0.8].excess_ret_12m.mean()
    print(f"size quintile {qq}: tax model bottom 20% excess {lo*100:.1f}, top 20% {hi*100:.1f}, all {te[s].excess_ret_12m.mean()*100:.1f}")
