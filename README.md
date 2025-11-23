# 🏢 Real Estate AI Agent

An intelligent AI-powered real estate financial analysis system built with **LangGraph**, **Django**, and **Streamlit**. This multi-agent system helps analyze real estate financial data, including revenue, expenses, and profit/loss calculations across properties, tenants, and time periods.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2.8-green.svg)](https://www.djangoproject.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.3-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-red.svg)](https://streamlit.io/)


## 🚀 Features

- 💬 **Conversational Interface**: Ask questions in natural language 
- 📊 **Data Query**: Show filtered data in tables with flexible filtering
- 💰 **P&L Analysis**: Calculate total Profit & Loss for specific time periods
- 🏢 **Property Comparison**: Compare financial performance between properties
- 🔍 **Asset Details**: Get comprehensive financial summaries for specific properties
- 🎯 **Intent Detection**: Automatically classifies user queries into appropriate workflows
- 🔄 **Multi-LLM Support**: Switch between OpenAI, Anthropic, Google Gemini providers
- ⚙️ **Customizable Prompts**: Manage system prompts through Django Admin without code changes
- 📁 **Dataset Upload**: Upload CSV or Parquet files with financial data

---

## 🏗️ Solution Architecture

### System Overview

The system consists of three main layers:

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                     │
│              (User Interface & Interaction)              │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Django Backend                          │
│        (Strategy & Prompt Configuration Management)      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  LangGraph Agent Core                    │
│           (Multi-Agent Workflow Orchestration)           │
└──────────────────────────────────────────────────────────┘
```

### Component Breakdown

1. **Streamlit Frontend** (`streamlit_app.py`)
   - Interactive web interface for users
   - Strategy selection and dataset upload
   - Query input and result display

2. **Django Backend** (`agent_app/`, `real_estate_manager/`)
   - Database management (SQLite/PostgreSQL)
   - Prompt versioning and management
   - Strategy configuration (LLM provider, model, prompts)
   - Admin panel for non-technical configuration

3. **Agent Core** (`agent_core/`)
   - LangGraph-based workflow orchestration
   - Intent detection and parameter extraction
   - Data retrieval and processing
   - Answer generation and formatting

---

## 🤖 Multi-Agent Workflow with LangGraph

### LangGraph State Machine

The agent operates as a **state machine** with 5 specialized nodes, each handling a specific task:

```
                            ┌─────────┐
                            │  START  │
                            └────┬────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ detect_intent │  ← Classifies user query
                         └───────┬───────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │ extract_params   │  ← Extracts filters & params
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ retrieve_data    │  ← Fetches filtered data
                       └────────┬─────────┘
                                │
                        ┌───────┴────────┐
                        │                │
                        ▼                ▼
               ┌────────────┐    ┌──────────────┐
               │ compute    │    │ end_with     │
               │ _answer    │    │ _error       │
               └─────┬──────┘    └──────┬───────┘
                     │                  │
                     └────────┬─────────┘
                              ▼
                         ┌─────────┐
                         │   END   │
                         └─────────┘
```

### Node Descriptions

#### 1. **detect_intent** 🎯
- **Purpose**: Classifies user query into one of 5 intents
- **Intents**:
  - `general_qa`: Conversational queries ("Hello", "How are you?")
  - `data_query`: Show filtered data ("Show all rows for month 2025-M01")
  - `total_pnl`: Calculate totals ("Total profit for 2025")
  - `asset_details`: Property info ("Tell me about Building 17")
  - `price_comparison`: Compare properties ("Compare Building 17 and Building 160")

#### 2. **extract_params** 🔍
- **Purpose**: Extracts structured parameters from user query
- **Extracted Data**:
  - Property names/addresses
  - Year and month filters
  - Filter conditions (property, tenant, ledger type)
  - Action type (show, aggregate, count)

#### 3. **retrieve_data** 📊
- **Purpose**: Fetches data from dataset based on extracted parameters
- **Data Sources**: Pandas DataFrames (Parquet/CSV)
- **Functions**:
  - `query_data_flexible()`: Flexible filtering
  - `compute_total_pnl()`: P&L calculations
  - `get_single_asset()`: Asset details
  - `compare_assets_by_price()`: Property comparison

#### 4. **compute_answer** ✍️
- **Purpose**: Generates natural language response
- **Formatting**:
  - Markdown tables for data query results
  - Statistical summaries
  - Formatted numbers with thousands separators
  - Icons and visual enhancements

#### 5. **end_with_error** ⚠️
- **Purpose**: Handles error states gracefully
- **Error Types**:
  - No data found
  - Invalid parameters
  - Dataset configuration errors

### Agent State

The agent maintains a typed state throughout the workflow:

```python
class AgentState(TypedDict, total=False):
    user_query: str                    # Original user question
    intent: Optional[str]              # Classified intent
    extracted_params: Dict[str, Any]   # Extracted parameters
    retrieved_data: Dict[str, Any]     # Fetched data
    answer: Optional[str]              # Final answer
    error: Optional[str]               # Error message if any
    strategy: Dict[str, Any]           # LLM configuration
    llm: BaseChatModel                 # LLM instance
```

---

## 📋 Setup Instructions

### Prerequisites

- Python 3.12 
- pip (Python package manager)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/JoeMat18/ai_by_yossi.git
cd cortex-real-estate-agent
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# LLM API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here

```

> See `env-example.txt` for reference.

### 5. Initialize Django Database

```bash
python manage.py migrate
python manage.py createsuperuser
```

Follow prompts to create an admin user.

### 6. Configure Prompts in Django Admin

```bash
python manage.py runserver
```

1. Visit `http://localhost:8000/admin`
2. Log in with your superuser credentials
3. Navigate to **Prompts** and create 3 prompts:
   - **Intent Prompt** (type: `intent`)
   - **Extract Prompt** (type: `extract`)
   - **General Q&A Prompt** (type: `general_qa`)

4. Navigate to **Strategy Prompt Mappings** and create a strategy:
   - Choose LLM provider (e.g., OpenAI)
   - Select model (e.g., gpt-4)
   - Link the 3 prompts you created
   - Mark as active ✅

> 📄 See `RECOMMENDED_PROMPTS.md` for ready-to-use prompt templates.

### 7. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

---

## 🎯 Usage

### Example Queries

#### Conversational
```
"Hello"
"How are you?"
"What can you do?"
```

#### Data Query (Show Rows)
```
"Show all rows for month 2025-M01"
"List all data for Building 17"
"Display all revenue records for January 2025"
```

#### Total P&L (Sum)
```
"Total profit for 2025"
"What is the total P&L for January?"
"Sum of all revenue in 2024"
```

#### Property Comparison
```
"Compare Building 17 and Building 160"
"Which is better: Building 17 or Building 180?"
```

#### Asset Details
```
"Tell me about Building 17"
"Show me information about Building 160"
```
