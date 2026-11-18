"""Synthetic research benchmark — no real student data."""
import pandas as pd
import streamlit as st
from ml.model_comparison import compare_models
st.set_page_config(page_title="Research Lab | AI Balance Buddy",page_icon="🔬",layout="wide")
st.title("🔬 Research Lab")
st.info("All comparisons below use 400 synthetic profiles and K-Means-derived pseudo-labels. Results are not real-world validity estimates.")
@st.cache_data(show_spinner="Running 5-fold cross-validation...")
def cached_benchmark():
    return compare_models()
if st.button("Run model comparison",type="primary"):
    results,meta=cached_benchmark()
    st.dataframe(pd.DataFrame(results),hide_index=True,use_container_width=True)
    st.bar_chart(pd.DataFrame(results).set_index("model")["macro_f1_mean"])
    st.caption("Macro-F1 is averaged across stratified folds. Do not interpret as real student classification accuracy.")
    with st.expander("Benchmark metadata"):st.json(meta)
st.markdown("### Experimental protocol")
st.write("Standardized linear and SVM baselines; Random Forest and XGBoost tree ensembles; 5-fold stratified CV with a fixed random seed. Pseudo-labels are constructed before cross-validation; this benchmark measures reconstruction of exploratory synthetic clusters only.")
