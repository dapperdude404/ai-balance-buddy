"""Stable SHAP feature extraction and plotting utilities."""
import numpy as np
import pandas as pd
def top_contributions(model,features:pd.DataFrame,class_index:int,limit=10):
    import shap
    if features.shape[0]!=1: raise ValueError("Expected exactly one observation")
    if class_index<0: raise ValueError("class_index must be nonnegative")
    values=np.asarray(shap.TreeExplainer(model)(features).values)
    if values.ndim==3:
        if class_index>=values.shape[2]: raise ValueError("Invalid class index")
        selected=values[0,:,class_index]
    elif values.ndim==2:
        selected=values[0]
    else:
        raise ValueError("Unexpected SHAP tensor shape")
    if len(selected)!=len(features.columns): raise ValueError("SHAP feature count mismatch")
    if not np.isfinite(selected).all(): raise ValueError("Non-finite SHAP values")
    return pd.Series(selected,index=features.columns,name="SHAP contribution").sort_values(key=abs,ascending=False).head(limit)
def plot_contributions(contributions):
    import matplotlib.pyplot as plt
    values=contributions.iloc[::-1]
    fig,ax=plt.subplots(figsize=(8,max(3,0.38*len(values)+1)))
    ax.barh([s.replace("_"," ") for s in values.index],values.values,color=["#F59E0B" if v>0 else "#8B5CF6" for v in values.values])
    ax.axvline(0,color="#94A3B8",linewidth=.8)
    ax.set_xlabel("SHAP contribution (model output units)")
    ax.set_title("Local explanation — not causal evidence")
    fig.tight_layout()
    return fig
