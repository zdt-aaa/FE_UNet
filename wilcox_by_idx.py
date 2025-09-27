# -*- coding: utf-8 -*-
import argparse, numpy as np, pandas as pd
from scipy.stats import wilcoxon, rankdata
from pathlib import Path

METRICS = [("DSC","ours_DSC","transunet_DSC"),
           ("IoU","ours_IoU","transunet_IoU"),
           ("PPV","ours_PPV","transunet_PPV"),
           ("TPR","ours_TPR","transunet_TPR")]

def holm(p):
    m=len(p); order=np.argsort(p); ps=np.array(p)[order]; adj=np.empty_like(ps)
    for i,pi in enumerate(ps): adj[i]=min((m-i)*pi,1.0)
    for i in range(1,m): adj[i]=max(adj[i],adj[i-1])  # 单调性
    out=np.empty_like(adj); out[order]=adj; return out

def rank_biserial(d):
    d=d[np.abs(d)>0];
    if d.size==0: return 0.0
    r=rankdata(np.abs(d),method='average')
    rp=r[d>0].sum(); rm=r[d<0].sum(); n=d.size; rmax=n*(n+1)/2.0
    return (rp-rm)/rmax

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="paired_metrics.csv")
    ap.add_argument("--outdir", default="stats_out")
    ap.add_argument("--alternative", choices=["two-sided","greater","less"], default="two-sided")
    args=ap.parse_args()

    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.csv)

    rows, p_raw=[], []
    for name, co, cb in METRICS:
        x=df[co].astype(float).to_numpy(); y=df[cb].astype(float).to_numpy()
        d=x-y
        res=wilcoxon(x, y, zero_method='wilcox', alternative=args.alternative, correction=False, mode='auto')
        rows.append({
            "Metric":name,
            "N_all":len(d),
            "N_used":int(np.count_nonzero(np.abs(d)>0)),
            "Median (TransUNet)":float(np.median(y)),
            "Median (Ours)":float(np.median(x)),
            "Median Δ":float(np.median(d)),
            "Mean Δ":float(np.mean(d)),
            "Wilcoxon W":int(res.statistic),
            "p (raw)":float(res.pvalue),
            "r_rb":float(rank_biserial(d))
        })
        p_raw.append(res.pvalue)

    p_adj=holm(p_raw)
    for i,pv in enumerate(p_adj): rows[i]["p (Holm)"]=float(pv)

    out=pd.DataFrame(rows)
    out.to_csv(outdir/"wilcoxon_results.csv", index=False)
    print(f"[Saved] {outdir/'wilcoxon_results.csv'}")

    # 生成 LaTeX 表
    fmt=out.copy()
    for c in ["Median (TransUNet)","Median (Ours)","Median Δ","Mean Δ","r_rb"]:
        fmt[c]=fmt[c].map(lambda v: f"{v:.4f}")
    for c in ["p (raw)","p (Holm)"]:
        fmt[c]=fmt[c].map(lambda v: f"{v:.3e}")
    latex=fmt[["Metric","N_used","Median (TransUNet)","Median (Ours)","Median Δ","Wilcoxon W","p (Holm)","r_rb"]].to_latex(
        index=False, escape=False, column_format="lrrrrrrr",
        caption="Wilcoxon signed-rank tests comparing our method with TransUNet (per idx).",
        label="tab:wilcoxon")
    (outdir/"wilcoxon_table.tex").write_text(latex, encoding="utf-8")
    print(f"[Saved] {outdir/'wilcoxon_table.tex'}")

if __name__ == "__main__":
    main()