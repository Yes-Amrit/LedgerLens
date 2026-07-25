from agent.graph import graph
from agent.schemas import ExtractedIntent
import agent.intent_extractor as ie
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Mock intent extractor to avoid LLM calls and rate limits
def mock_extract_intent(query):
    return ExtractedIntent(
        intent="aggregation_query",
        date_filter=None,
        entity_ids=[],
        target_pattern=None,
        transaction_type_filter=None
    )
ie.extract_intent = mock_extract_intent

query = "Which customers made 10+ transactions under $10,000?"
inputs = {"raw_query": query}
print("Running query:", query)
try:
    for s in graph.stream(inputs):
        print("--- Node Executed ---")
        node_name = list(s.keys())[0]
        print("Node:", node_name)
        state_after = s[node_name]
        print("Plan:", state_after.get('plan', []))
        if 'anomaly_results' in state_after:
            print("Anomaly results flag count:", len(state_after['anomaly_results'].get('flagged_transactions', [])))
            print("Method used:", state_after['anomaly_results'].get('method_used', ''))
except Exception as e:
    print("Error:", e)
