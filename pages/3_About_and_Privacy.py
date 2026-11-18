import streamlit as st
from survey_schema import SCHEMA_VERSION
st.set_page_config(page_title="Privacy | AI Balance Buddy",page_icon="🔒",layout="wide")
st.title("🔒 About and privacy")
st.success("Current demo: survey inputs are processed in memory; the app does not intentionally persist submitted survey responses.")
st.write("Survey schema version:",SCHEMA_VERSION)
st.write("No login or database is active in this Sprint 1 prototype. Future consent-based research storage requires an approved protocol, access controls, retention limits, and a deletion workflow.")
st.warning("This is an educational research demonstration using synthetic training data. It is not a diagnostic tool.")
