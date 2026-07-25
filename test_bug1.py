from agent.graph import run_graph
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

query = "Which customers made 10+ transactions under $10,000?"
print("Running query:", query)
try:
    result = run_graph(query)
    print("Result Explanation:", result.get('explanation', ''))
except Exception as e:
    print("Error:", e)
