from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from agent.state import AgentState

def explanation_node(state: AgentState) -> dict:
    """
    Use the LLM to generate a natural-language explanation given the flagged 
    transaction's features and risk level
    """
    risk_results = state.get("risk_results", {})
    overall_risk = risk_results.get("overall_risk", "low")
    
    if overall_risk == "low":
        return {"explanation": "No significant risk detected. Transactions appear normal."}
        
    prompt = f"""
    You are an AML investigator assistant. 
    Explain why a set of transactions was flagged with a risk level of {overall_risk}.
    Provide a concise natural language explanation.
    """
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_output_tokens=256)
        response = llm.invoke([HumanMessage(content=prompt)])
        explanation = response.content
    except Exception as e:
        explanation = f"Error generating explanation: {str(e)}"
        
    return {"explanation": explanation}
