from typing import Optional, List
from pydantic import BaseModel, Field

class ExtractedIntent(BaseModel):
    intent: str = Field(description='One of: "pattern_search", "aggregation_query", "entity_lookup", "broad_exploration"')
    date_filter: Optional[dict] = Field(default=None, description='Start and end date, e.g. {"start_date": "...", "end_date": "..."}')
    entity_ids: Optional[List[str]] = Field(default=None, description='List of specific entity IDs extracted from the query.')
    target_pattern: Optional[str] = Field(default=None, description='One of: "structuring", "layering", "cash_out", or "none"')
    transaction_type_filter: Optional[str] = Field(default=None, description='Filter for specific transaction types if mentioned.')
