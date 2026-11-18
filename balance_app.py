"""AI Balance Buddy: synthetic research demo dashboard."""
import os
import numpy as np
import pandas as pd
import streamlit as st
from features import LEVELS,DISCIPLINES,encode,balance_score
from train import train
from survey_schema import encode_survey, SurveyValidationError, SCHEMA_VERSION
from ml.explainability import top_contributions, plot_contributions
st.set_page_config(page_title="AI Balance Buddy",page_icon="⚖️",layout="wide")
st.markdown("<style>.block-container{max-width:1200px;padding-top:2rem}[data-testid=stMetric]{background:#172038;padding:16px;border-radius:12px}</style>",unsafe_allow_html=True)
@st.cache_resource(show_spinner="Preparing synthetic research model...")
def model_and_report():
    return train(save=False)
st.title("⚖️ AI Balance Buddy 2.0")
st.caption("Sprint 1 research preview · schema "+SCHEMA_VERSION)
st.caption("Explainable machine learning for balanced Gen-AI use among university students")
st.info("Research demonstration using 400 **synthetic** student profiles. Scores are illustrative and not validated diagnoses. Survey responses are not stored.")
with st.sidebar:
    st.header("Student assessment")
    age=st.slider("Age",18,60,22)
    level=st.selectbox("Academic level",LEVELS)
    discipline=st.selectbox("Discipline",DISCIPLINES)
    hours=st.slider("Weekly Gen-AI hours",0.,40.,8.,.5)
    st.subheader("Task delegation")
    brainstorm=st.checkbox("Brainstorming",True)
    drafting=st.checkbox("Full text drafting")
    coding=st.checkbox("Coding / debugging")
    st.subheader("Self-regulation")
    quota=st.checkbox("Daily AI usage quota")
    isolation=st.checkbox("AI-free task initiation")
    st.subheader("Behavioral indicators (1–5)")
    prompt=st.slider("Iterative prompt complexity",1,5,3)
    fact=st.slider("Independent fact checking",1,5,3)
    mod=st.slider("Manual editing of AI content",1,5,3)
    blank=st.slider("Difficulty starting without AI",1,5,3)
    outage=st.slider("Anxiety during AI outages",1,5,3)
    atrophy=st.slider("Perceived skill decline",1,5,3)
    night=st.slider("Late-night AI reliance",1,5,3)
record={'Age':age,'Academic_Level':level,'Discipline':discipline,'Weekly_GenAI_Hours':hours,'Task_Brainstorming':int(brainstorm),'Task_Drafting':int(drafting),'Task_Coding':int(coding),'Quota_Enforced':int(quota),'Cognitive_Isolation_Mode':int(isolation),'Prompt_Complexity':prompt,'Fact_Check_Freq':fact,'Content_Mod_Rate':mod,'Blank_Page_Paralysis':blank,'System_Outage_Anxiety':outage,'Skill_Atrophy_Sense':atrophy,'Late_Night_Dependency':night}
try:
    X=encode_survey(record)
except SurveyValidationError as exc:
    st.error("Survey validation error: "+str(exc))
    st.stop()
try:
    model,report=model_and_report()
except Exception as exc:
    st.error("Model initialization failed: "+str(exc))
    st.stop()
probs=model.predict_proba(X)[0]
score=balance_score(probs)
labels=['Healthy augmentation','Moderate reliance','High dependency pattern']
class_id=int(np.argmax(probs))
st.metric("AI Balance Score",f"{score}/100")
st.metric("Predicted cohort",labels[class_id])
st.metric("High-reliance probability",f"{probs[2]*100:.1f}%")
st.progress(int(score))
st.subheader("Model cohort probabilities")
st.bar_chart(pd.DataFrame({'Probability':probs},index=labels))
st.subheader("Explainable AI · SHAP")
try:
    top=top_contributions(model,X,class_id,limit=10)
    fig=plot_contributions(top)
    st.pyplot(fig,use_container_width=True)
    import matplotlib.pyplot as plt
    plt.close(fig)
    st.caption("Model attribution, not causal evidence.")
except Exception as exc:
    st.warning("SHAP visualization unavailable: "+type(exc).__name__)
st.subheader("Personalized self-regulation coaching")
drivers=[]
if blank>=4: drivers.append("difficulty initiating independent work")
if night>=4: drivers.append("late-night AI reliance")
if fact<=2: drivers.append("limited fact checking")
if mod<=2: drivers.append("limited editing of AI output")
if hours>=15: drivers.append("high weekly AI use")
if not drivers: drivers=["maintaining independent reasoning"]
fallback="1. Start with a 15-minute independent outline before opening AI tools.\n\n2. Verify AI claims using independent primary sources.\n\n3. Set an AI usage window and review habits weekly."
if st.button("Generate coaching plan",type="primary"):
    key=os.getenv("ANTHROPIC_API_KEY")
    if not key:
        try: key=st.secrets.get("ANTHROPIC_API_KEY")
        except Exception: key=None
    if key:
        try:
            import anthropic
            response=anthropic.Anthropic(api_key=key).messages.create(model=os.getenv("ANTHROPIC_MODEL","claude-sonnet-4-5"),max_tokens=450,messages=[{"role":"user","content":f"Act as a supportive academic study coach. Provide exactly 3 practical nonjudgmental tips. The student's illustrative synthetic-model balance score is {score}/100. Concerns: {', '.join(drivers)}. Never diagnose addiction or claim clinical validity."}])
            st.write(response.content[0].text)
        except Exception:
            st.warning("Online coaching unavailable; showing offline guidance.")
            st.markdown(fallback)
    else: st.markdown(fallback)
with st.expander("Methods, evaluation and limitations"):
    st.write("K-Means K=3 pseudo-labels, XGBoost classification, TreeSHAP explanations, 400 synthetic profiles. Held-out accuracy measures reproduction of pseudo-labels, not validity on real students.")
    st.json(report)
