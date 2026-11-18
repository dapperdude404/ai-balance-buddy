import numpy as np
import pandas as pd
import pytest
from ml.explainability import top_contributions,plot_contributions
from train import train,generate
from features import encode
def test_shap_top_contributions():
    model,_=train(save=False)
    X=encode(generate(n=1))
    top=top_contributions(model,X,class_index=1,limit=5)
    assert len(top)==5
    assert np.isfinite(top.values).all()
    fig=plot_contributions(top)
    assert fig is not None
def test_shap_rejects_multiple_rows():
    model,_=train(save=False)
    with pytest.raises(ValueError):
        top_contributions(model,encode(generate(n=2)),class_index=1)
