from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.intent_extractor import extract_intent
from agent.planner import build_plan

from agent.nodes.eda_node import eda_node
from agent.nodes.feature_engineering_node import feature_engineering_node
from agent.nodes.aggregation_node import aggregation_node
from agent.nodes.entity_lookup_node import entity_lookup_node
from agent.nodes.anomaly_node import anomaly_node
from agent.nodes.risk_node import risk_node
from agent.nodes.explanation_node import explanation_node
from agent.nodes.escalation_node import escalation_node

def extract_and_plan(state: AgentState) -> dict:
    raw_query = state["raw_query"]
    extracted = extract_intent(raw_query)
    plan = build_plan(extracted)
    
    return {
        "intent": extracted.intent,
        "date_filter": extracted.date_filter,
        "entity_ids": extracted.entity_ids,
        "target_pattern": extracted.target_pattern,
        "transaction_type_filter": extracted.transaction_type_filter,
        "plan": plan
    }

def route_next(state: AgentState) -> str:
    plan = state.get("plan", [])
    if not plan:
        return END
    return plan[0]

def get_next_node_in_plan(current_node: str):
    def router(state: AgentState) -> str:
        plan = state.get("plan", [])
        try:
            idx = plan.index(current_node)
            if idx + 1 < len(plan):
                return plan[idx + 1]
            return END
        except ValueError:
            return END
    return router

builder = StateGraph(AgentState)

builder.add_node("router", extract_and_plan)
builder.add_node("eda_node", eda_node)
builder.add_node("feature_engineering_node", feature_engineering_node)
builder.add_node("aggregation_node", aggregation_node)
builder.add_node("entity_lookup_node", entity_lookup_node)
builder.add_node("anomaly_node", anomaly_node)
builder.add_node("risk_node", risk_node)
builder.add_node("explanation_node", explanation_node)
builder.add_node("escalation_node", escalation_node)

builder.set_entry_point("router")
builder.add_conditional_edges("router", route_next)

all_nodes = [
    "eda_node", "feature_engineering_node", "aggregation_node", 
    "entity_lookup_node", "anomaly_node", "risk_node", 
    "explanation_node", "escalation_node"
]

for node in all_nodes:
    builder.add_conditional_edges(node, get_next_node_in_plan(node))

graph = builder.compile()

def run_graph(query: str) -> dict:
    state = {"raw_query": query}
    return graph.invoke(state)
