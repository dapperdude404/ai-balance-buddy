import pytest
from survey_schema import SCHEMA_VERSION,validate_survey,encode_survey,SurveyValidationError
from train import generate
def sample():
    row=generate(n=1).iloc[0].to_dict()
    row["Age"]=int(row["Age"])
    for key in ["Prompt_Complexity","Fact_Check_Freq","Content_Mod_Rate","Blank_Page_Paralysis","System_Outage_Anxiety","Skill_Atrophy_Sense","Late_Night_Dependency"]:
        row[key]=int(row[key])
    for key in ["Task_Brainstorming","Task_Drafting","Task_Coding","Quota_Enforced","Cognitive_Isolation_Mode"]:
        row[key]=int(row[key])
    return row
def test_valid():
    assert encode_survey(sample()).shape[0]==1
def test_missing():
    x=sample();del x["Age"]
    with pytest.raises(SurveyValidationError):validate_survey(x)
def test_out_of_range():
    x=sample();x["Age"]=100
    with pytest.raises(SurveyValidationError):validate_survey(x)
def test_version():
    x=sample();x["schema_version"]="1.0"
    with pytest.raises(SurveyValidationError):validate_survey(x)
def test_extra():
    x=sample();x["secret"]="unexpected"
    with pytest.raises(SurveyValidationError):validate_survey(x)
