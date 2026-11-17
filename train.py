"""Synthetic-only reproducible research training pipeline."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from features import LEVELS, DISCIPLINES, NUMERIC, COLUMNS, encode, dependency_index
ROOT=Path(__file__).resolve().parent

def generate(n=400,seed=42):
    r=np.random.default_rng(seed)
    latent=r.uniform(0,1,n)
    clamp=lambda x:np.clip(np.rint(x),1,5).astype(int)
    return pd.DataFrame({
        'Age':r.integers(18,46,n),
        'Academic_Level':r.choice(LEVELS,n,p=[.6,.3,.1]),
        'Discipline':r.choice(DISCIPLINES,n),
        'Weekly_GenAI_Hours':np.clip(2+21*latent+r.normal(0,3,n),0,40),
        'Task_Brainstorming':r.binomial(1,.7,n),
        'Task_Drafting':r.binomial(1,np.clip(.15+.7*latent,0,1)),
        'Task_Coding':r.binomial(1,.5,n),
        'Quota_Enforced':r.binomial(1,np.clip(.7-.55*latent,0,1)),
        'Cognitive_Isolation_Mode':r.binomial(1,np.clip(.7-.55*latent,0,1)),
        'Prompt_Complexity':clamp(4.5-2.8*latent+r.normal(0,.8,n)),
        'Fact_Check_Freq':clamp(4.7-3.4*latent+r.normal(0,.7,n)),
        'Content_Mod_Rate':clamp(4.6-3.2*latent+r.normal(0,.7,n)),
        'Blank_Page_Paralysis':clamp(1+3.7*latent+r.normal(0,.7,n)),
        'System_Outage_Anxiety':clamp(1+3.6*latent+r.normal(0,.8,n)),
        'Skill_Atrophy_Sense':clamp(1+3.6*latent+r.normal(0,.8,n)),
        'Late_Night_Dependency':clamp(1+3.5*latent+r.normal(0,.8,n))
    })

def train(save=True):
    raw=generate()
    X=encode(raw)
    clustered=X.copy()
    clustered[NUMERIC]=StandardScaler().fit_transform(X[NUMERIC])
    clusters=KMeans(n_clusters=3,n_init=20,random_state=42).fit_predict(clustered)
    raw['Raw_Cluster']=clusters
    raw['DI']=dependency_index(raw)
    ranking={int(k):i for i,k in enumerate(raw.groupby('Raw_Cluster')['DI'].mean().sort_values().index)}
    y=pd.Series(clusters).map(ranking)
    a,b,ya,yb=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model=XGBClassifier(n_estimators=200,max_depth=3,learning_rate=.1,reg_lambda=1,objective='multi:softprob',num_class=3,random_state=42,n_jobs=2)
    model.fit(a,ya)
    report={'data_type':'synthetic','sample_count':len(raw),'silhouette':float(silhouette_score(clustered,clusters)),'classification_report':classification_report(yb,model.predict(b),output_dict=True),'confusion_matrix':confusion_matrix(yb,model.predict(b)).tolist()}
    if save:
        model.save_model(str(ROOT/'xgb_balance_model.json'))
        (ROOT/'feature_columns.json').write_text(json.dumps(COLUMNS,indent=2))
        (ROOT/'training_report.json').write_text(json.dumps(report,indent=2))
    return model,report

if __name__=='__main__':
    _,report=train()
    print('Synthetic evaluation accuracy:',report['classification_report']['accuracy'])
