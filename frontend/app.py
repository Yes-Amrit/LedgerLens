"""
LedgerLens Streamlit Frontend

Run: streamlit run frontend/app.py (requires api/main.py running separately via: uvicorn api.main:app --reload)
"""

import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000/investigate"

st.set_page_config(page_title="LedgerLens", layout="wide")

st.title("LedgerLens Investigation Dashboard")
st.markdown("Agentic AML transaction investigation system")

# Sidebar for example queries
st.sidebar.title("Example Queries")
st.sidebar.markdown("Click to test example queries:")

query_1 = "Which customers made 10+ transactions under $10,000?"
query_2 = "Is customer ID 8724731955 suspicious?"
query_3 = "Show me an overview of transaction patterns"

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

if st.sidebar.button("Query 1: Aggregation"):
    st.session_state.query_input = query_1
if st.sidebar.button("Query 2: Entity Lookup"):
    st.session_state.query_input = query_2
if st.sidebar.button("Query 3: Broad Exploration"):
    st.session_state.query_input = query_3

query = st.text_input("Enter natural language query:", value=st.session_state.query_input)

if st.button("Investigate"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Investigating..."):
            try:
                response = requests.post(API_URL, json={"query": query})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.header("Query Understanding")
                    exec_summary = data.get("execution_summary", {})
                    st.write(f"**Intent:** {exec_summary.get('intent', 'N/A')}")
                    st.write(f"**Target Pattern:** {exec_summary.get('target_pattern', 'N/A')}")
                    st.write(f"**Entity IDs:** {exec_summary.get('entity_ids', 'N/A')}")
                    st.write(f"**Date Filter:** {exec_summary.get('date_filter', 'N/A')}")
                    
                    st.header("Execution Plan")
                    plan = exec_summary.get("plan_executed", [])
                    if plan:
                        st.write(" ➔ ".join(plan))
                    else:
                        st.write("No plan executed.")
                    
                    st.header("Results")
                    flagged = data.get("flagged_transactions", [])
                    if flagged:
                        if isinstance(flagged, list) and isinstance(flagged[0], dict):
                            df = pd.DataFrame(flagged)
                            st.dataframe(df)
                        else:
                            st.write(flagged)
                    else:
                        st.write("No transactions flagged.")
                    
                    st.header("Risk Level")
                    risk = data.get("risk_level", "unknown").upper()
                    if risk == "HIGH":
                        st.error(f"Risk Level: {risk}")
                    elif risk in ["MEDIUM", "MODERATE"]:
                        st.warning(f"Risk Level: {risk}")
                    elif risk == "LOW":
                        st.success(f"Risk Level: {risk}")
                    else:
                        st.info(f"Risk Level: {risk}")
                        
                    st.header("Explanation")
                    st.write(data.get("explanation", "No explanation provided."))
                    
                    st.header("Recommended Action")
                    st.info(data.get("escalation_action", "No action recommended."))
                    
                    if data.get("calibration_warning"):
                        st.warning(data["calibration_warning"])
                        
                else:
                    st.error(f"Error from API: {response.status_code}")
                    try:
                        st.write(response.json())
                    except:
                        st.write(response.text)
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")
