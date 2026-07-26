import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
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

Respond with ONLY the structured output. Do not include explanations, markdown formatting, or code fences.
"""

def extract_intent(query: str) -> ExtractedIntent:
    from agent.config import MODEL_NAME

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0,
        max_output_tokens=1024,  # was 256 — too small, was truncating JSON mid-object
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{query}")
    ])

    # method="json_mode" forces Gemini's native JSON output mode rather than
    # relying on prompted function-calling, which is more failure-prone here
    structured_llm = llm.with_structured_output(ExtractedIntent, method="json_mode")
    chain = prompt | structured_llm

    try:
        return chain.invoke({"query": query})
    except Exception as e:
        # Fallback: retry once with function-calling mode instead of json_mode,
        # in case json_mode itself was the failure point for this query
        try:
            fallback_llm = llm.with_structured_output(ExtractedIntent, method="function_calling")
            fallback_chain = prompt | fallback_llm
            return fallback_chain.invoke({"query": query})
        except Exception:
            raise RuntimeError(
                f"Intent extraction failed for query: {query!r}. Original error: {e}"
            )