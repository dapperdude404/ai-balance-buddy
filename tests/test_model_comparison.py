from ml.model_comparison import compare_models
def test_all_models_evaluated():
    results,meta=compare_models(n=90,folds=3)
    assert {x["model"] for x in results}=={"Logistic Regression","Random Forest","SVM (RBF)","XGBoost"}
    assert all(0<=x["macro_f1_mean"]<=1 for x in results)
    assert meta["label_origin"]=="synthetic K-Means pseudo-labels"
