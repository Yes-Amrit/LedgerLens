from data.loader import load_dataset
from agent.nodes.aggregation_node import aggregation_node

df = load_dataset()
state = {"dataset": df}
result = aggregation_node(state)
print("Aggregation Node Result:", result)
