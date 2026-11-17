import pandas as pd
from features import encode, COLUMNS, balance_score, dependency_index
def test_encoding():
    sample={'Age':22,'Academic_Level':'Masters','Discipline':'STEM','Weekly_GenAI_Hours':8,'Task_Brainstorming':1,'Task_Drafting':0,'Task_Coding':1,'Quota_Enforced':0,'Cognitive_Isolation_Mode':0,'Prompt_Complexity':3,'Fact_Check_Freq':4,'Content_Mod_Rate':4,'Blank_Page_Paralysis':2,'System_Outage_Anxiety':2,'Skill_Atrophy_Sense':2,'Late_Night_Dependency':2}
    result=encode(pd.DataFrame([sample]))
    assert list(result.columns)==COLUMNS
    assert result['Academic_Level_Masters'].iloc[0]==1
    assert result['Discipline_STEM'].iloc[0]==1
def test_score():
    assert balance_score([1,0,0])==100
    assert balance_score([0,0,1])==0
    assert balance_score([0,1,0])==50
