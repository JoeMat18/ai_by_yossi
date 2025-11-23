from typing import Dict, Any, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from agent_core.graph_nodes import AgentState
from agent_core.graph_builder import app as graph_app

def build_llm_from_strategy(strategy_config: Dict[str, Any]) -> BaseChatModel:
    """
    Create a LangChain chat model based on the strategy configuration

    strategy_config should contain:
    - provider
    - model_name
    - (optional) llm_kwargs
    """
    provider = strategy_config.get("provider")
    model_name = strategy_config.get("model_name")
    llm_kwargs = strategy_config.get("llm_kwargs") or {}

    llm: BaseChatModel = init_chat_model(
        model=model_name,
        model_provider=provider,
        temperature=0.0,
        **llm_kwargs,
    )

    return llm

def run_agent(
    user_query: str,
    strategy_config: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:

    llm = build_llm_from_strategy(strategy_config)

    strategy_prompts = {
        "intent_system_prompt": strategy_config.get("intent_system_prompt"),
        "extract_system_prompt": strategy_config.get("extract_system_prompt"),
        "general_qa_system_prompt": strategy_config.get("general_qa_system_prompt")
    }

    initial_state: AgentState = {
        "user_query": (user_query or "").strip(),
        "intent": None,
        "extracted_params": {},
        "retrieved_data": {},
        "answer": None,
        "error": None,
        "strategy": strategy_prompts,
        "llm": llm,
    }

    result = graph_app.invoke(initial_state)

    answer = result.get("answer") or ""
    error = result.get("error")
    if not answer and error:
        answer = f"Sorry, I couldn't complete your request: {error}"

    debug_state: Dict[str, Any] = {
        "intent": result.get("intent"),
        "extracted_params": result.get("extracted_params"),
        "retrieved_data": result.get("retrieved_data"),
        "error": result.get("error"),
    }

    return answer, debug_state