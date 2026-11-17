"""Shared survey feature schema for training and inference."""
import pandas as pd
LEVELS=['UG','Masters','Doctoral']
DISCIPLINES=['STEM','Humanities','Business','Law','Medicine']
NUMERIC=['Age','Weekly_GenAI_Hours','Prompt_Complexity','Fact_Check_Freq','Content_Mod_Rate','Blank_Page_Paralysis','System_Outage_Anxiety','Skill_Atrophy_Sense','Late_Night_Dependency']
BINARY=['Task_Brainstorming','Task_Drafting','Task_Coding','Quota_Enforced','Cognitive_Isolation_Mode']
COLUMNS=NUMERIC+BINARY+[f'Academic_Level_{x}' for x in LEVELS]+[f'Discipline_{x}' for x in DISCIPLINES]
def encode(frame):
    result=frame.drop(columns=['Academic_Level','Discipline'],errors='ignore').copy()
    for value in LEVELS:
        result[f'Academic_Level_{value}']=(frame['Academic_Level']==value).astype(int)
    for value in DISCIPLINES:
        result[f'Discipline_{value}']=(frame['Discipline']==value).astype(int)
    return result.reindex(columns=COLUMNS).astype(float)
def dependency_index(df):
    return (df['Blank_Page_Paralysis']+df['System_Outage_Anxiety']+df['Skill_Atrophy_Sense']+df['Late_Night_Dependency']+(6-df['Fact_Check_Freq'])+(6-df['Content_Mod_Rate']))/6
def balance_score(probabilities):
    return round(100*(1-(.5*float(probabilities[1])+float(probabilities[2]))),1)
