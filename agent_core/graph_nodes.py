import json
import calendar
from typing import TypedDict, Optional, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from agent_core.data_loader import (
    compare_assets_by_price,
    get_single_asset,
    compute_total_pnl,
    summarize_asset_record,
    query_data_flexible,
    DatasetConfigError,
)


class AgentState(TypedDict, total=False):
    user_query: str
    intent: Optional[str]
    extracted_params: Dict[str, Any]
    retrieved_data: Dict[str, Any]
    answer: Optional[str]
    error: Optional[str]
    strategy: Dict[str, Any]
    llm: BaseChatModel


# ---------- Node: Intent detection ----------

def detect_intent(state: AgentState) -> AgentState:
    """
    Classifies user query intent using LLM.
    
    IMPORTANT: System prompt MUST be configured in Django Admin 
    (agent_app.models.Prompt with type='intent').
    The prompt is passed via state['strategy']['intent_system_prompt'].
    """
    llm: BaseChatModel = state["llm"]
    query = state["user_query"].lower()

    # Get system prompt from Django (REQUIRED)
    strategy_prompts = state.get("strategy", {})
    system_prompt = strategy_prompts.get("intent_system_prompt")
    
    if not system_prompt:
        raise ValueError(
            "Intent system prompt not configured. "
            "Please configure it in Django Admin (Prompt model with type='intent')."
        )

    try:
        system = SystemMessage(content=system_prompt)
        human = HumanMessage(content=state["user_query"])
        resp = llm.invoke([system, human])
        label = resp.content.strip().lower()
    except Exception:
        label = "general_qa"

    # Failsafe - only if LLM returns general_qa
    if label == "general_qa":
        # Check if this is actually a data query
        keywords_data_explicit = [
            "show all", "list all", "show rows", "list rows", "display rows",
            "get all", "all rows", "all data", "filter by", "where month", 
            "where year", "for month", "for year", "records for"
        ]
        keywords_pnl = ["total profit", "total loss", "total pnl", "sum of", "aggregate"]
        keywords_asset = ["building", "property", "tenant", "rent", "asset"]
        
        # Only switch to data_query if explicitly asking for data
        if any(k in query for k in keywords_data_explicit):
            label = "data_query"
        elif any(k in query for k in keywords_pnl):
            label = "total_pnl"
        elif any(k in query for k in keywords_asset):
            label = "asset_details"
        # Otherwise, keep it as general_qa (default for conversational queries)

    valid_intents = {"price_comparison", "total_pnl", "asset_details", "general_qa", "data_query"}
    if label not in valid_intents:
        # Default to general_qa for unknown intents (conversational fallback)
        label = "general_qa"

    state["intent"] = label
    return state


# ---------- Node: Parameter extraction ----------

def extract_params(state: AgentState) -> AgentState:
    """
    Extracts structured parameters from user query using LLM.
    """
    import re
    
    llm: BaseChatModel = state["llm"]
    intent = state.get("intent")

    if intent == "general_qa":
        state["extracted_params"] = {"addresses": [], "year": None, "month": None}
        return state

    # Get system prompt from Django (REQUIRED)
    strategy_prompts = state.get("strategy", {})
    system_prompt = strategy_prompts.get("extract_system_prompt")
    
    if not system_prompt:
        raise ValueError(
            "Extract system prompt not configured. "
            "Please configure it in Django Admin (Prompt model with type='extract')."
        )

    system = SystemMessage(content=system_prompt)
    human = HumanMessage(content=state["user_query"])
    resp = llm.invoke([system, human])

    try:
        content = resp.content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)
        addresses = parsed.get("addresses") or []
        year = parsed.get("year", None)
        month = parsed.get("month", None)
        filters = parsed.get("filters", {})
        action = parsed.get("action", "show")  # "show", "aggregate", "count"
    except (json.JSONDecodeError, AttributeError):
        addresses = []
        year = None
        month = None
        filters = {}
        action = "show"
        # Fallback extraction (simple regex)
        
        # Try to extract month pattern like "2025-M01" (exact format from CSV)
        month_pattern_match = re.search(r"(202[0-9]-M\d{2})", state["user_query"])
        if month_pattern_match:
            filters["month"] = month_pattern_match.group(1)
            year_from_month = int(month_pattern_match.group(1).split("-")[0])
            year = year_from_month
        else:
            # Try year extraction
            year_match = re.search(r"202[0-9]", state["user_query"])
            if year_match:
                year = int(year_match.group(0))
        
        # Extract property/building names (for both addresses and filters)
        building_matches = re.findall(r"[Bb]uilding\s+(\d+)", state["user_query"])
        if building_matches:
            # Add all buildings to addresses (for comparison and asset details)
            addresses = [f"Building {num}" for num in building_matches]
            # Also add the first one to filters (for data_query)
            filters["property_name"] = f"Building {building_matches[0]}"
        
        # Extract tenant
        tenant_match = re.search(r"[Tt]enant\s+(\d+)", state["user_query"])
        if tenant_match:
            filters["tenant_name"] = f"Tenant {tenant_match.group(1)}"
        
        # Detect action from keywords
        query_lower = state["user_query"].lower()
        if any(word in query_lower for word in ["show", "list", "display", "all rows", "get"]):
            action = "show"
        elif any(word in query_lower for word in ["sum", "total", "aggregate"]):
            action = "aggregate"
        elif any(word in query_lower for word in ["count", "how many"]):
            action = "count"

    # Fix: If LLM put property_name in filters but addresses is empty, copy to addresses
    if not addresses and filters.get("property_name"):
        # Extract all Building names from the property_name filter
        prop_name = filters["property_name"]
        if "Building" in prop_name:
            # Check if there are multiple buildings in the original query
            all_buildings = re.findall(r"[Bb]uilding\s+(\d+)", state["user_query"])
            if all_buildings:
                addresses = [f"Building {num}" for num in all_buildings]
            else:
                addresses = [prop_name]

    state["extracted_params"] = {
        "addresses": addresses,
        "year": year,
        "month": month,
        "filters": filters,
        "action": action
    }
    return state


# ---------- Node: Data retrieval ----------

def retrieve_data(state: AgentState) -> AgentState:
    intent = state.get("intent")
    params = state.get("extracted_params") or {}
    
    addresses = params.get("addresses") or []
    year = params.get("year")
    month = params.get("month") # Extract month

    result: Dict[str, Any] = {}

    try:
        if intent == "price_comparison":
            if len(addresses) < 2:
                state["error"] = "To compare, I need two property names."
                return state
            
            data = compare_assets_by_price(addresses[0], addresses[1])
            result["price_comparison"] = data

        elif intent == "asset_details":
            if not addresses:
                state["error"] = "I couldn't identify the property name."
                return state

            record = get_single_asset(addresses[0])
            if record is None:
                state["error"] = f"Property '{addresses[0]}' not found."
                return state

            result["asset_details"] = record

        elif intent == "total_pnl":
            # Pass both year and month
            total = compute_total_pnl(year=year, month=month)
            result["total_pnl"] = {"year": year, "month": month, "value": total}

        elif intent == "data_query":
            # Flexible data query
            filters = params.get("filters", {})
            action = params.get("action", "show")
            
            # Add year/month to filters if provided
            if year is not None and "year" not in filters:
                filters["year"] = year
            if month is not None and "month" not in filters:
                # Convert month number to string format if needed
                if isinstance(month, int):
                    filters["month"] = f"{year}-M{month:02d}"
                else:
                    filters["month"] = month
            
            query_result = query_data_flexible(filters=filters, action=action, limit=200)
            result["data_query"] = query_result

        elif intent == "general_qa":
            result["general_qa"] = {}

        else:
            state["error"] = f"Unknown intent: {intent}"
            return state

    except DatasetConfigError as e:
        state["error"] = f"Data Config Error: {e}"
        return state
    except ValueError as e:
        # This message will be sent to the user (e.g., "No data for 2023")
        state["error"] = str(e)
        return state
    except Exception as e:
        state["error"] = f"System Error: {e}"
        return state

    state["retrieved_data"] = result
    return state


# ---------- Node: Answer computation ----------

def compute_answer(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    llm: BaseChatModel = state["llm"]
    intent = state.get("intent")
    data = state.get("retrieved_data") or {}

    if intent == "price_comparison":
        comp = data.get("price_comparison", {})
        a1_name = comp['asset_1']['property_name']
        a1_val = comp['asset_1']['value']
        a2_name = comp['asset_2']['property_name']
        a2_val = comp['asset_2']['value']
        diff = comp['difference']
        
        better = a1_name if diff > 0 else a2_name
        answer = (
            f"**Comparison Result:**\n\n"
            f"* **{a1_name}**: {a1_val:,.2f} profit\n"
            f"* **{a2_name}**: {a2_val:,.2f} profit\n\n"
            f"The property **{better}** generated {abs(diff):,.2f} more profit."
        )
        state["answer"] = answer

    elif intent == "asset_details":
        record = data.get("asset_details", {})
        summary_text = summarize_asset_record(record)
        state["answer"] = f"Here is the financial summary for the requested property:\n\n{summary_text}"

    elif intent == "total_pnl":
        pnl = data.get("total_pnl", {})
        val = pnl.get("value", 0.0)
        yr = pnl.get("year")
        mn = pnl.get("month")
        
        # Format a nice time description
        if yr and mn:
            month_name = calendar.month_name[mn] # 1 -> January
            time_frame = f"for {month_name} {yr}"
        elif yr:
            time_frame = f"for the full year {yr}"
        else:
            time_frame = "across all recorded dates"
            
        state["answer"] = f"The total recorded **Profit/Loss** {time_frame} is: **{val:,.2f}**"

    elif intent == "data_query":
        query_result = data.get("data_query", {})
        
        if query_result.get("status") == "no_data":
            state["answer"] = query_result.get("message", "No data found.")
        else:
            # Format the result nicely
            answer_parts = []
            
            # Header
            filters = query_result.get('filters', {})
            if filters:
                filter_desc = ", ".join([f"{k}={v}" for k, v in filters.items()])
                answer_parts.append(f"**Query Results** (Filters: {filter_desc})\n")
            else:
                answer_parts.append(f"**Query Results**\n")
            
            # Summary
            answer_parts.append(f"📊 **Summary:**")
            answer_parts.append(f"- Total matching records: **{query_result.get('count', 0):,}**")
            answer_parts.append(f"- Total Profit/Loss: **{query_result.get('total_profit', 0):,.2f}**\n")
            
            # Show rows if available
            if "rows" in query_result:
                rows = query_result["rows"]
                showing = query_result.get('showing', 0)
                total = query_result.get('total_rows', 0)
                
                answer_parts.append(f"📋 **Data Rows** (showing {showing} of {total}):\n")
                
                # Convert rows to markdown table
                if rows:
                    # Get all unique keys from all rows
                    all_keys = set()
                    for row in rows:
                        all_keys.update(row.keys())
                    
                    # Filter out less important columns for display
                    important_cols = ["property_name", "tenant_name", "ledger_type", "ledger_description", 
                                     "month", "year", "profit", "entity_name", "ledger_category"]
                    display_cols = [col for col in important_cols if col in all_keys]
                    
                    # Add any remaining columns not in important_cols
                    for col in all_keys:
                        if col not in display_cols:
                            display_cols.append(col)
                    
                    # Build markdown table
                    header = "| " + " | ".join(display_cols) + " |"
                    separator = "| " + " | ".join(["---"] * len(display_cols)) + " |"
                    answer_parts.append(header)
                    answer_parts.append(separator)
                    
                    for row in rows:
                        row_values = []
                        for col in display_cols:
                            val = row.get(col, "")
                            # Format profit with 2 decimal places
                            if col == "profit" and val != "":
                                try:
                                    val = f"{float(val):,.2f}"
                                except:
                                    pass
                            # Handle None/NaN values
                            if val is None or str(val) == "nan":
                                val = "-"
                            row_values.append(str(val))
                        answer_parts.append("| " + " | ".join(row_values) + " |")
                
                if total > showing:
                    answer_parts.append(f"\n*...and {total - showing} more rows*")
            
            # Show aggregations if available
            if "aggregations" in query_result:
                aggs = query_result["aggregations"]
                answer_parts.append(f"\n📈 **Aggregations:**")
                
                if aggs.get("by_ledger_type"):
                    answer_parts.append("\n**By Ledger Type:**")
                    for ledger, value in sorted(aggs["by_ledger_type"].items(), key=lambda x: x[1], reverse=True):
                        answer_parts.append(f"- {ledger}: {value:,.2f}")
                
                if aggs.get("by_property"):
                    answer_parts.append("\n**By Property:**")
                    for prop, value in sorted(aggs["by_property"].items(), key=lambda x: x[1], reverse=True)[:10]:
                        answer_parts.append(f"- {prop}: {value:,.2f}")
            
            # Show counts if available
            if "counts" in query_result:
                counts = query_result["counts"]
                answer_parts.append(f"\n📊 **Counts:**")
                answer_parts.append(f"- Total records: {counts.get('total', 0):,}")
                
                if counts.get("by_ledger_type"):
                    answer_parts.append("\n**By Ledger Type:**")
                    for ledger, count in sorted(counts["by_ledger_type"].items(), key=lambda x: x[1], reverse=True):
                        answer_parts.append(f"- {ledger}: {count:,}")
            
            state["answer"] = "\n".join(answer_parts)

    elif intent == "general_qa":
        # Get general Q&A prompt from Django (REQUIRED)
        strategy_prompts = state.get("strategy", {})
        general_qa_prompt = strategy_prompts.get("general_qa_system_prompt")
        
        if not general_qa_prompt:
            raise ValueError(
                "General Q&A system prompt not configured. "
                "Please configure it in Django Admin (Prompt model with type='general_qa')."
            )
        
        system = SystemMessage(content=general_qa_prompt)
        human = HumanMessage(content=state["user_query"])
        resp = llm.invoke([system, human])
        state["answer"] = resp.content

    return state


def route_after_retrieval(state: AgentState) -> str:
    if state.get("error"):
        return "end_with_error"
    return "compute_answer"


def end_with_error(state: AgentState) -> AgentState:
    if state.get("error") and not state.get("answer"):
        error_msg = state["error"]
        state["answer"] = f"{error_msg}"
    return state