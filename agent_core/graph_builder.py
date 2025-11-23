from langgraph.graph import StateGraph, START, END

from agent_core.graph_nodes import (
    AgentState,
    detect_intent,
    extract_params,
    retrieve_data,
    compute_answer,
    end_with_error,
    route_after_retrieval,
)

def build_graph() -> "CompiledGraph":
    """
    Creates and returns a compiled LangGraph graph
    
    Called from the orchestrator during initialization.
    """
    builder = StateGraph(AgentState)
    builder.add_node("detect_intent", detect_intent)
    builder.add_node("extract_params", extract_params)
    builder.add_node("retrieve_data", retrieve_data)
    builder.add_node("compute_answer", compute_answer)
    builder.add_node("end_with_error", end_with_error)

    builder.add_edge(START, "detect_intent")
    builder.add_edge("detect_intent", "extract_params")
    builder.add_edge("extract_params", "retrieve_data")


    builder.add_conditional_edges(
        "retrieve_data",
        route_after_retrieval,
        {
            "compute_answer": "compute_answer",
            "end_with_error": "end_with_error",
        },
    )

    builder.add_edge("compute_answer", END)
    builder.add_edge("end_with_error", END)

    app = builder.compile()
    return app


# For convenience: ready app immediately on import
app = build_graph()