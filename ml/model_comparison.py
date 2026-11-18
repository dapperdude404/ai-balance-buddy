"""Synthetic-only model comparison. Not external validation."""
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from features import encode, dependency_index, NUMERIC
from train import generate

def build_synthetic_benchmark(n=400,seed=42):
    raw=generate(n=n,seed=seed)
    X=encode(raw)
    cluster_input=X.copy()
    cluster_input[NUMERIC]=StandardScaler().fit_transform(cluster_input[NUMERIC])
    clusters=KMeans(n_clusters=3,random_state=seed,n_init=20).fit_predict(cluster_input)
    raw["cluster"]=clusters
    raw["dependency_index"]=dependency_index(raw)
    order=raw.groupby("cluster")["dependency_index"].mean().sort_values().index
    mapping={int(cluster):i for i,cluster in enumerate(order)}
    y=np.array([mapping[int(c)] for c in clusters])
    return X,y,{"silhouette":float(silhouette_score(cluster_input,clusters)),"sample_count":n,"label_origin":"synthetic K-Means pseudo-labels"}

def compare_models(n=400,seed=42,folds=5):
    X,y,metadata=build_synthetic_benchmark(n,seed)
    models={
        "Logistic Regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,random_state=seed)),
        "Random Forest":RandomForestClassifier(n_estimators=120,random_state=seed,n_jobs=2),
        "SVM (RBF)":make_pipeline(StandardScaler(),SVC(kernel="rbf",probability=True,random_state=seed)),
        "XGBoost":XGBClassifier(n_estimators=120,max_depth=3,learning_rate=.1,objective="multi:softprob",num_class=3,random_state=seed,n_jobs=2)
    }
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    results=[]
    for name,model in models.items():
        scores=cross_validate(model,X,y,cv=cv,scoring={"macro_f1":"f1_macro","accuracy":"accuracy"},n_jobs=1)
        results.append({"model":name,"macro_f1_mean":round(float(scores["test_macro_f1"].mean()),4),"macro_f1_std":round(float(scores["test_macro_f1"].std()),4),"accuracy_mean":round(float(scores["test_accuracy"].mean()),4)})
    return sorted(results,key=lambda item:item["macro_f1_mean"],reverse=True),metadata

if __name__=="__main__":
    rows,metadata=compare_models()
    print(json.dumps({"results":rows,"metadata":metadata},indent=2))
