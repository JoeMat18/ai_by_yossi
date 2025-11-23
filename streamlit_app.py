import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
from dotenv import load_dotenv
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "real_estate_manager.settings")
import django  # noqa: E402

django.setup()

from agent_app.models import StrategyPromptMapping  # noqa: E402
from agent_core.orchestrator import run_agent  # noqa: E402
from agent_core import data_loader  # noqa: E402
from agent_core.config import DATA_PATH  # noqa: E402


@st.cache_data(show_spinner=False)
def load_strategy_options() -> List[Dict[str, Any]]:
    """
    Pull active strategies from Django and convert them into orchestrator configs.
    Cached to avoid hitting the database on every rerun.
    """
    strategies = (
        StrategyPromptMapping.objects.filter(is_active=True)
        .select_related("intent_prompt", "extract_prompt", "general_qa_prompt")
        .order_by("strategy_name")
    )

    options: List[Dict[str, Any]] = []
    for strategy in strategies:
        options.append(
            {
                "label": strategy.strategy_name,
                "config": {
                    "provider": strategy.provider,
                    "model_name": strategy.model_name,
                    "llm_kwargs": {},
                    "intent_system_prompt": (
                        strategy.intent_prompt.content if strategy.intent_prompt else None
                    ),
                    "extract_system_prompt": (
                        strategy.extract_prompt.content if strategy.extract_prompt else None
                    ),
                    "general_qa_system_prompt": (
                        strategy.general_qa_prompt.content
                        if strategy.general_qa_prompt
                        else None
                    ),
                },
            }
        )
    return options


def build_user_query(question: str) -> str:
    return (question or "").strip()


def persist_uploaded_dataset(uploaded_file) -> Tuple[int, str]:
    """
    Save user-uploaded CSV/Parquet into DATA_PATH expected by the agent.
    Returns (row_count, format) for display.
    """
    if not uploaded_file:
        raise ValueError("File is not uploaded.")

    suffix = Path(uploaded_file.name).suffix.lower()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if suffix in (".parquet", ".pq"):
        DATA_PATH.write_bytes(uploaded_file.getvalue())
        df = pd.read_parquet(DATA_PATH)
        fmt = "parquet"
    elif suffix in (".csv", ".txt"):
        df = pd.read_csv(uploaded_file)
        df.to_parquet(DATA_PATH, index=False)
        fmt = "csv→parquet"
    else:
        raise ValueError("Only CSV and Parquet files are supported.")

    data_loader.get_assets_df.cache_clear()
    return len(df), fmt


def main():
    st.set_page_config(page_title="Cortex Agent", layout="wide")
    st.title("Cortex Real Estate Agent")
    st.caption("Choose an LLM strategy, upload a file, and submit your query.")

    strategy_options = load_strategy_options()
    if not strategy_options:
        st.error("No active strategies. Create them in Django Admin.")
        return

    option_labels = [item["label"] for item in strategy_options]
    selected_label = st.selectbox("Strategy", option_labels, index=0)
    selected_config = next(
        item["config"] for item in strategy_options if item["label"] == selected_label
    )

    with st.expander("Strategy settings", expanded=False):
        st.json(selected_config)

    st.subheader("Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV or Parquet with asset data",
        type=["csv", "parquet"],
        help="The file will be saved to data/ and used by the agent.",
    )

    if uploaded_file:
        try:
            rows, fmt = persist_uploaded_dataset(uploaded_file)
            st.success(
                f"File uploaded ({fmt}). Rows saved: {rows:,}. "
                f"Path: {DATA_PATH}"
            )
            with st.expander("Preview"):
                df_preview = pd.read_parquet(DATA_PATH).head(5)
                st.dataframe(df_preview)
        except Exception as exc:
            st.error(f"Failed to save dataset: {exc}")

    user_question = st.text_area(
        "Query text",
        placeholder="Describe the task for the agent...",
        help="The LLM will see this text along with the uploaded data.",
    )

    if st.button("Run agent"):
        user_query = build_user_query(user_question)
        if not user_query:
            st.warning("Enter a query to run the agent.")
            return

        with st.spinner("Running LangGraph agent..."):
            try:
                answer, debug_state = run_agent(user_query, selected_config)
            except Exception as exc:
                st.error(f"Error while running the agent: {exc}")
                return

        st.success("Agent finished running.")
        st.subheader("Answer")
        st.write(answer or "Answer is empty.")

        with st.expander("Diagnostics"):
            st.json(debug_state)


if __name__ == "__main__":
    main()

