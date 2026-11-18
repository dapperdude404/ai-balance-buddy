"""Versioned, validated survey contract (v2.0.0)."""
from dataclasses import dataclass
from typing import Mapping
import pandas as pd
from features import LEVELS, DISCIPLINES, encode
SCHEMA_VERSION="2.0.0"
RANGES={"Age":(18,60),"Weekly_GenAI_Hours":(0,80)}
LIKERT=["Prompt_Complexity","Fact_Check_Freq","Content_Mod_Rate","Blank_Page_Paralysis","System_Outage_Anxiety","Skill_Atrophy_Sense","Late_Night_Dependency"]
BINARY=["Task_Brainstorming","Task_Drafting","Task_Coding","Quota_Enforced","Cognitive_Isolation_Mode"]
REQUIRED=set(RANGES)|set(LIKERT)|set(BINARY)|{"Academic_Level","Discipline"}
@dataclass
class SurveyValidationError(ValueError):
    errors: list
    def __str__(self):
        return "; ".join(self.errors)
def validate_survey(data: Mapping)->pd.DataFrame:
    errors=[]
    missing=REQUIRED-set(data)
    if missing: errors.append("Missing fields: "+", ".join(sorted(missing)))
    extra=set(data)-REQUIRED-{"schema_version"}
    if extra: errors.append("Unknown fields: "+", ".join(sorted(extra)))
    if data.get("schema_version",SCHEMA_VERSION)!=SCHEMA_VERSION:
        errors.append("Unsupported survey schema version")
    for field,values in (("Academic_Level",LEVELS),("Discipline",DISCIPLINES)):
        if field in data and data[field] not in values: errors.append(field+" must be one of "+str(values))
    for field,(low,high) in RANGES.items():
        if field in data and (isinstance(data[field],bool) or not isinstance(data[field],(int,float)) or not low<=data[field]<=high):
            errors.append(field+" must be within "+str((low,high)))
    for field in LIKERT:
        if field in data and (type(data[field]) is not int or data[field] not in range(1,6)):
            errors.append(field+" must be an integer 1–5")
    for field in BINARY:
        if field in data and data[field] not in (0,1,False,True):
            errors.append(field+" must be binary")
    if errors: raise SurveyValidationError(errors)
    return pd.DataFrame([{key:data[key] for key in sorted(REQUIRED)}])
def encode_survey(data:Mapping)->pd.DataFrame:
    return encode(validate_survey(data))
