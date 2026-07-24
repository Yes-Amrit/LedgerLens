import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.schemas import ExtractedIntent

load_dotenv()

SYSTEM_PROMPT = """You are an intent extraction engine for an AML (Anti-Money Laundering) system.
Your job is to take a user's natural language query and extract the structured intent.

Valid intents are:
- "pattern_search": The user is looking for a specific money laundering pattern (e.g., structuring, layering).
- "aggregation_query": The user is asking for aggregate statistics or a threshold-based query (no ML needed).
- "entity_lookup": The user is investigating a specific entity or list of entities.
- "broad_exploration": The user wants to explore the dataset generally without a specific target.

Examples:
User: "Find structuring patterns in the last 30 days"
Result: intent="pattern_search", date_filter={{"start_date": "...", "end_date": "..."}}, target_pattern="structuring"

User: "Which customers made 10+ transactions under $10,000?"
Result: intent="aggregation_query", target_pattern="none"

User: "Is customer ID 4521 suspicious?"
Result: intent="entity_lookup", entity_ids=["4521"]
"""

def extract_intent(query: str) -> ExtractedIntent:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_output_tokens=256)
    structured_llm = llm.with_structured_output(ExtractedIntent)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{query}")
    ])
    
    chain = prompt | structured_llm
    return chain.invoke({"query": query})
