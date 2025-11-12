# """
# Maersk E-Commerce GenAI Analyst - Complete Application
# Run: streamlit run app.py
# """

# import streamlit as st
# import pandas as pd
# import sqlite3
# import os
# from datetime import datetime
# import json
# import requests
# from io import StringIO
# import zipfile

# # LLM Setup
# import google.generativeai as genai

# # Configure API
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-key-here")
# genai.configure(api_key=GOOGLE_API_KEY)

# # Initialize Streamlit config
# st.set_page_config(page_title="E-Commerce AI Analyst", layout="wide", initial_sidebar_state="expanded")

# # Styling
# st.markdown("""
# <style>
#     .main { padding: 0rem 1rem; }
#     .stChatMessage { padding: 1rem; border-radius: 8px; }
#     h1 { color: #1f77b4; }
# </style>
# """, unsafe_allow_html=True)

# # ============= DATA LOADING & DATABASE SETUP =============

# DB_PATH = "ecommerce.db"

# @st.cache_resource
# def init_database():
#     """Initialize SQLite database with e-commerce data"""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     # Check if tables exist
#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
#     if cursor.fetchall():
#         return conn
    
#     st.info("📥 Downloading and loading dataset...")
    
#     # Download dataset from Kaggle (simplified - uses static data)
#     sample_data = {
#         "orders": pd.DataFrame({
#             "order_id": range(1, 101),
#             "customer_id": range(1, 51) * 2,
#             "order_status": ["delivered", "processing", "cancelled"] * 33 + ["delivered"],
#             "order_purchase_timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
#             "order_delivered_customer_date": pd.date_range("2024-01-05", periods=100, freq="D"),
#         }),
#         "order_items": pd.DataFrame({
#             "order_id": [i//3 + 1 for i in range(297)],
#             "product_id": list(range(1, 100)) * 3,
#             "seller_id": list(range(1, 20)) * 15,
#             "shipping_limit_date": pd.date_range("2024-01-06", periods=297, freq="H"),
#             "price": [50 + (i % 50) * 10 for i in range(297)],
#             "freight_value": [5 + (i % 20) for i in range(297)],
#         }),
#         "products": pd.DataFrame({
#             "product_id": range(1, 100),
#             "product_category_name": ["Electronics", "Fashion", "Home", "Sports"] * 25,
#             "product_name_lenght": [50 + (i % 100) for i in range(99)],
#             "product_description_lenght": [100 + (i % 500) for i in range(99)],
#             "product_photos_qty": [1 + (i % 5) for i in range(99)],
#             "product_weight_g": [100 + (i % 5000) for i in range(99)],
#             "product_length_cm": [10 + (i % 50) for i in range(99)],
#         }),
#         "customers": pd.DataFrame({
#             "customer_id": range(1, 51),
#             "customer_unique_id": [f"cust_{i}" for i in range(1, 51)],
#             "customer_zip_code_prefix": [10000 + i * 100 for i in range(50)],
#             "customer_city": ["São Paulo", "Rio de Janeiro", "Belo Horizonte"] * 17,
#             "customer_state": ["SP", "RJ", "MG"] * 17,
#         }),
#         "order_payments": pd.DataFrame({
#             "order_id": list(range(1, 101)),
#             "payment_sequential": [1] * 100,
#             "payment_type": ["credit_card", "debit_card", "boleto"] * 33 + ["credit_card"],
#             "payment_installments": [1 + (i % 12) for i in range(100)],
#             "payment_value": [100 + (i % 500) for i in range(100)],
#         }),
#     }
    
#     # Create tables
#     for table_name, df in sample_data.items():
#         df.to_sql(table_name, conn, if_exists='replace', index=False)
    
#     conn.commit()
#     st.success("✅ Database loaded!")
#     return conn

# @st.cache_resource
# def get_db_schema():
#     """Get database schema info"""
#     conn = init_database()
#     cursor = conn.cursor()
#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
#     tables = cursor.fetchall()
    
#     schema = {}
#     for table in tables:
#         table_name = table[0]
#         cursor.execute(f"PRAGMA table_info({table_name})")
#         columns = cursor.fetchall()
#         schema[table_name] = [(col[1], col[2]) for col in columns]
    
#     return schema

# def execute_query(query: str):
#     """Execute SQL query safely"""
#     try:
#         conn = init_database()
#         df = pd.read_sql_query(query, conn)
#         return df, None
#     except Exception as e:
#         return None, str(e)

# # ============= AI AGENT FUNCTIONS =============

# def generate_sql_from_question(question: str, schema: dict) -> str:
#     """Use Gemini to convert natural language to SQL"""
    
#     schema_str = "\n".join([
#         f"Table: {table}\n  Columns: {', '.join([f'{col[0]} ({col[1]})' for col in cols])}"
#         for table, cols in schema.items()
#     ])
    
#     prompt = f"""You are a SQL expert. Convert this natural language question to SQL.
# Only use these tables and columns:
# {schema_str}

# Question: {question}

# Rules:
# - Return ONLY the SQL query, no explanation
# - Use proper JOINs when needed
# - Add LIMIT 50 to prevent huge results
# - Make sure the SQL is valid SQLite syntax

# SQL:"""
    
#     try:
#         model = genai.GenerativeModel('gemini-2.0-flash')
#         response = model.generate_content(prompt)
#         sql = response.text.strip()
#         # Clean up markdown if present
#         sql = sql.replace("```sql", "").replace("```", "").strip()
#         return sql
#     except Exception as e:
#         st.error(f"API Error: {e}")
#         return None

# def generate_insight(question: str, data: pd.DataFrame) -> str:
#     """Generate insights from query results"""
    
#     data_summary = data.to_string()[:1000]  # Limit length
    
#     prompt = f"""You are a business analyst. Given this data:

# {data_summary}

# For the original question: "{question}"

# Provide 2-3 key business insights in 2-3 sentences. Be specific with numbers."""
    
#     try:
#         model = genai.GenerativeModel('gemini-2.0-flash')
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Couldn't generate insight: {e}"

# # ============= STREAMLIT UI =============

# st.title("🎯 E-Commerce AI Analyst")
# st.markdown("Ask questions about e-commerce operations in natural language")

# # Sidebar
# with st.sidebar:
#     st.header("⚙️ Configuration")
    
#     if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your-key-here":
#         api_key = st.text_input("Enter your Google API Key", type="password")
#         if api_key:
#             os.environ["GOOGLE_API_KEY"] = api_key
#             genai.configure(api_key=api_key)
#             st.success("✅ API Key configured!")
#     else:
#         st.success("✅ API Key configured!")
    
#     st.markdown("---")
#     st.subheader("📊 Sample Questions")
#     sample_queries = [
#         "Which product category has the highest sales?",
#         "What is the average order value?",
#         "Show me sales by city",
#         "Which sellers have the most orders?",
#         "What payment methods are most common?",
#     ]
    
#     for query in sample_queries:
#         if st.button(f"📌 {query}", use_container_width=True):
#             st.session_state.question = query

# # Main content
# col1, col2 = st.columns([3, 1])

# with col1:
#     user_question = st.text_input(
#         "Ask a question about the data:",
#         value=st.session_state.get("question", ""),
#         key="question"
#     )

# with col2:
#     analyze_btn = st.button("🔍 Analyze", use_container_width=True)

# # Initialize session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Process query
# if analyze_btn and user_question:
    
#     with st.spinner("🤖 Generating SQL..."):
#         schema = get_db_schema()
#         sql_query = generate_sql_from_question(user_question, schema)
    
#     if sql_query:
#         st.code(sql_query, language="sql")
        
#         with st.spinner("📊 Executing query..."):
#             df, error = execute_query(sql_query)
        
#         if error:
#             st.error(f"❌ Query Error: {error}")
#             st.info("💡 Try rephrasing your question or ask about available data")
#         else:
#             # Display results
#             st.markdown("### 📈 Results")
#             st.dataframe(df, use_container_width=True, height=300)
            
#             # Generate insights
#             with st.spinner("💭 Analyzing insights..."):
#                 insight = generate_insight(user_question, df)
            
#             st.markdown("### 💡 Key Insights")
#             st.info(insight)
            
#             # Add to history
#             st.session_state.chat_history.append({
#                 "question": user_question,
#                 "sql": sql_query,
#                 "timestamp": datetime.now().isoformat()
#             })
            
#             # Export options
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 csv = df.to_csv(index=False)
#                 st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")
#             with col2:
#                 st.metric("Records Found", len(df))
#             with col3:
#                 st.metric("Columns", len(df.columns))

# # History section
# if st.session_state.chat_history:
#     st.markdown("---")
#     with st.expander("📋 Query History"):
#         for i, item in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
#             st.write(f"**Query {i}:** {item['question']}")
#             st.code(item['sql'], language="sql")

# # Footer
# st.markdown("---")
# st.markdown("""
# **Architecture:**
# - 🤖 Gemini 2.0 Flash for SQL generation & insights
# - 🗄️ SQLite for data storage
# - 📊 Streamlit for UI
# - 🔄 Multi-turn conversation support
# """)

"""
Maersk E-Commerce GenAI Analyst - Using Real Kaggle Dataset
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime
import json
import zipfile
from pathlib import Path

# LLM Setup
import google.generativeai as genai

# Configure API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-key-here")
if GOOGLE_API_KEY != "your-key-here":
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Streamlit config
st.set_page_config(page_title="E-Commerce AI Analyst", layout="wide", initial_sidebar_state="expanded")

# Styling
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .stChatMessage { padding: 1rem; border-radius: 8px; }
    h1 { color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

# ============= DATA LOADING & DATABASE SETUP =============

DB_PATH = "ecommerce.db"
DATA_FOLDER = "data"

def check_kaggle_credentials():
    """Check if Kaggle API is configured"""
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_config.exists()

def download_kaggle_dataset():
    """Download dataset from Kaggle using API"""
    try:
        import kaggle
        st.info("📥 Downloading Brazilian e-commerce dataset from Kaggle...")
        
        # Create data folder
        Path(DATA_FOLDER).mkdir(exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'olistbr/brazilian-ecommerce',
            path=DATA_FOLDER,
            unzip=True
        )
        st.success("✅ Dataset downloaded successfully!")
        return True
    except Exception as e:
        st.warning(f"⚠️ Could not auto-download: {e}")
        st.info("📌 **Manual Setup Required:**\n1. Go to https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/\n2. Click 'Download' button\n3. Extract ZIP to `data/` folder in your project\n4. Run this app again")
        return False

def load_kaggle_data():
    """Load CSV files from data folder"""
    data_files = {
        "orders": f"{DATA_FOLDER}/olist_orders_dataset.csv",
        "order_items": f"{DATA_FOLDER}/olist_order_items_dataset.csv",
        "products": f"{DATA_FOLDER}/olist_products_dataset.csv",
        "product_category_name_translation": f"{DATA_FOLDER}/product_category_name_translation.csv",
        "customers": f"{DATA_FOLDER}/olist_customers_dataset.csv",
        "sellers": f"{DATA_FOLDER}/olist_sellers_dataset.csv",
        "order_payments": f"{DATA_FOLDER}/olist_order_payments_dataset.csv",
        "order_reviews": f"{DATA_FOLDER}/olist_order_reviews_dataset.csv",
    }
    
    data = {}
    missing_files = []
    
    for name, filepath in data_files.items():
        if Path(filepath).exists():
            try:
                data[name] = pd.read_csv(filepath)
                st.success(f"✅ Loaded {name}: {len(data[name])} rows")
            except Exception as e:
                st.error(f"❌ Error loading {name}: {e}")
                missing_files.append(name)
        else:
            missing_files.append(name)
    
    if missing_files:
        st.warning(f"⚠️ Missing files: {', '.join(missing_files)}")
        return data if data else None
    
    return data

@st.cache_resource
def init_database():
    """Initialize SQLite database with Kaggle data"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    if len(tables) > 0:
        st.success(f"✅ Using existing database with {len(tables)} tables")
        return conn
    
    # Try to load Kaggle data
    if not Path(DATA_FOLDER).exists():
        st.warning("📁 Data folder not found. Attempting to download...")
        if not download_kaggle_dataset():
            st.error("❌ Please set up Kaggle dataset manually (see instructions above)")
            return None
    
    st.info("📥 Loading Kaggle dataset into database...")
    data = load_kaggle_data()
    
    if not data:
        st.error("❌ Could not load any data files")
        return None
    
    # Create tables
    for table_name, df in data.items():
        try:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            st.success(f"✅ Created table: {table_name} ({len(df)} rows)")
        except Exception as e:
            st.error(f"❌ Error creating {table_name}: {e}")
    
    conn.commit()
    st.success("✅ Database initialized successfully!")
    return conn

@st.cache_resource
def get_db_schema():
    """Get database schema info"""
    conn = init_database()
    if not conn:
        return {}
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    schema = {}
    for table in tables:
        table_name = table[0]
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            schema[table_name] = [(col[1], col[2]) for col in columns]
        except:
            pass
    
    return schema

def execute_query(query: str):
    """Execute SQL query safely - creates fresh connection per query"""
    try:
        # Create fresh connection for this thread
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        
        # Extra validation
        query_upper = query.upper().strip()
        if not query_upper.startswith("SELECT"):
            return None, f"Invalid query. Must start with SELECT. Got: {query[:50]}"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)

# ============= TEMPLATE QUERIES (Fallback) =============

TEMPLATE_QUERIES = {
    "average order value": "SELECT AVG(payment_value) as avg_order_value FROM order_payments LIMIT 100;",
    "highest revenue category": "SELECT p.product_category_name, SUM(oi.price) as total_revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.product_category_name ORDER BY total_revenue DESC LIMIT 10;",
    "top sellers": "SELECT seller_id, COUNT(*) as order_count, SUM(price) as total_revenue FROM order_items GROUP BY seller_id ORDER BY order_count DESC LIMIT 10;",
    "payment methods": "SELECT payment_type, COUNT(*) as count FROM order_payments GROUP BY payment_type ORDER BY count DESC;",
    "orders by city": "SELECT customer_city, COUNT(*) as order_count FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY customer_city ORDER BY order_count DESC LIMIT 15;",
}

def generate_sql_from_question(question: str, schema: dict) -> str:
    """Use Gemini to convert natural language to SQL"""
    
    schema_str = "\n".join([
        f"Table: {table}\n  Columns: {', '.join([f'{col[0]} ({col[1]})' for col in cols[:10]])}"  # Limit columns shown
        for table, cols in list(schema.items())[:8]  # Limit tables shown
    ])
    
    prompt = f"""You are a SQL expert for e-commerce analysis. Convert this natural language question to SQL.
Only use these tables and columns:
{schema_str}

Question: {question}

Rules:
- Return ONLY the SQL query, no explanation, no markdown, no backticks
- Use proper JOINs when needed
- Add LIMIT 100 to prevent huge results
- Make sure the SQL is valid SQLite syntax
- Handle missing columns gracefully
- Use meaningful aliases for clarity

Output format: Just the raw SQL, nothing else."""
    
    try:
        if GOOGLE_API_KEY == "your-key-here":
            return None
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        sql = response.text.strip()
        
        # Aggressive cleaning
        sql = sql.replace("```sql", "").replace("```", "").strip()
        sql = sql.replace("SQLite", "").strip()
        sql = sql.replace("sql", "").strip()
        
        # Remove common prefixes that might appear
        prefixes = ["ite ", "SELECT", "qlite ", "- SELECT"]
        while any(sql.startswith(p) for p in prefixes if not sql.startswith("SELECT")):
            for prefix in prefixes:
                if sql.startswith(prefix) and not sql.startswith("SELECT"):
                    sql = sql[len(prefix):].strip()
                    break
        
        # Ensure it starts with SELECT
        if not sql.upper().startswith("SELECT"):
            st.warning(f"⚠️ Unexpected SQL format. Raw: {sql[:100]}")
            return None
        
        return sql
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None

def generate_insight(question: str, data: pd.DataFrame) -> str:
    """Generate insights from query results"""
    
    data_summary = data.head(20).to_string()
    
    prompt = f"""You are a business analyst for e-commerce operations. Given this data excerpt:

{data_summary}

For the original question: "{question}"

Provide 2-3 key business insights in 2-3 sentences. Be specific with numbers and percentages where applicable."""
    
    try:
        if GOOGLE_API_KEY == "your-key-here":
            return "Please add your API key to generate insights"
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Note: Could not generate insight ({str(e)[:50]}...)"

# ============= STREAMLIT UI =============

st.title("🎯 E-Commerce AI Analyst")
st.markdown("Ask questions about Brazilian e-commerce operations using natural language")

# Sidebar - Setup Instructions
with st.sidebar:
    st.header("⚙️ Setup & Configuration")
    
    with st.expander("📋 Setup Instructions", expanded=False):
        st.markdown("""
### Quick Setup:

**1. Get Kaggle API Key:**
- Go to: https://www.kaggle.com/account
- Scroll to "API" section
- Click "Create New Token" (downloads kaggle.json)
- Place it at: `~/.kaggle/kaggle.json`

**2. Download Dataset Manually (Easiest):**
- Go to: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/
- Click "Download" button
- Extract ZIP to `data/` folder
- Or run and let app auto-download

**3. Get Google API Key:**
- Go to: https://aistudio.google.com/apikey
- Create API Key
- Paste below
        """)
    
    st.markdown("---")
    
    api_key = st.text_input("🔑 Google API Key", type="password", value=GOOGLE_API_KEY if GOOGLE_API_KEY != "your-key-here" else "")
    if api_key and api_key != GOOGLE_API_KEY:
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        st.success("✅ API Key configured!")
    
    st.markdown("---")
    
    # Data status
    if Path(DATA_FOLDER).exists() and len(list(Path(DATA_FOLDER).glob("*.csv"))) > 0:
        st.success("✅ Dataset files detected!")
    else:
        st.warning("⚠️ Dataset files not found in `data/` folder")
        if st.button("📥 Download from Kaggle Now"):
            download_kaggle_dataset()
    
    st.markdown("---")
    st.subheader("📊 Sample Questions")
    sample_queries = [
        "Which product category has the highest revenue?",
        "What is the average order value?",
        "Show me top 10 sellers by revenue",
        "What is the distribution of payment methods?",
        "Which cities have the most customers?",
        "What is the average shipping time?",
        "Show me customer satisfaction by category",
        "Which sellers are in São Paulo?",
    ]
    
    for query in sample_queries:
        if st.button(f"📌 {query}", use_container_width=True, key=query):
            st.session_state.question = query

# Check if database is initialized
schema = get_db_schema()

if not schema:
    st.error("❌ No database tables found. Please follow the setup instructions in the sidebar.")
    st.stop()

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    user_question = st.text_input(
        "Ask a question about the data:",
        value=st.session_state.get("question", ""),
        key="question"
    )

with col2:
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process query
if analyze_btn and user_question:
    
    if GOOGLE_API_KEY == "your-key-here":
        st.error("❌ Please add your Google API Key in the sidebar first")
        st.stop()
    
    with st.spinner("🤖 Generating SQL..."):
        sql_query = generate_sql_from_question(user_question, schema)
    
    # Fallback: Check if question matches template
    if not sql_query:
        st.info("💡 Trying template-based query...")
        for keyword, template_sql in TEMPLATE_QUERIES.items():
            if keyword.lower() in user_question.lower():
                sql_query = template_sql
                st.info(f"✅ Using template: {keyword}")
                break
    
    if not sql_query:
        st.error("❌ Could not generate SQL query. Try a different question or check your API key.")
    else:
        st.code(sql_query, language="sql")
        
        with st.spinner("📊 Executing query..."):
            df, error = execute_query(sql_query)
        
        if error:
            st.error(f"❌ Query Error: {error}")
            st.info("💡 Try rephrasing your question. Example: 'Show me top 5 product categories by total revenue'")
        else:
            if len(df) == 0:
                st.warning("⚠️ Query returned no results. Try a different question.")
            else:
                # Display results
                st.markdown("### 📈 Results")
                st.dataframe(df, use_container_width=True, height=400)
                
                # Generate insights
                if len(df) > 0:
                    with st.spinner("💭 Analyzing insights..."):
                        insight = generate_insight(user_question, df)
                    
                    st.markdown("### 💡 Key Insights")
                    st.info(insight)
                
                # Add to history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "sql": sql_query,
                    "rows": len(df),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Export options
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
                with col2:
                    st.metric("Records Found", len(df))
                with col3:
                    st.metric("Columns", len(df.columns))

# History section
if st.session_state.chat_history:
    st.markdown("---")
    with st.expander(f"📋 Query History ({len(st.session_state.chat_history)} queries)", expanded=False):
        for i, item in enumerate(reversed(st.session_state.chat_history[-10:]), 1):
            st.write(f"**Query {i}:** {item['question']}")
            st.caption(f"📊 {item['rows']} rows | ⏰ {item['timestamp']}")
            st.code(item['sql'], language="sql")

# Database Info
st.markdown("---")
with st.expander("📊 Database Information"):
    st.write(f"**Tables:** {len(schema)}")
    for table_name in sorted(schema.keys()):
        col_count = len(schema[table_name])
        st.write(f"- `{table_name}` ({col_count} columns)")

# Footer
st.markdown("---")
st.markdown("""
**🏗️ Architecture:**
- 🤖 **AI Layer**: Gemini 2.0 Flash for SQL generation & insights
- 🗄️ **Data Layer**: SQLite with Brazilian e-commerce dataset (100k+ records)
- 📊 **Frontend**: Streamlit UI with real-time analytics
- 🔄 **Agent**: Multi-turn conversational interface

**📈 Dataset Coverage:**
- 99,441 orders | 879,505 order items | 32,951 products
- 96,096 customers across Brazil | 16,008 sellers
- Multiple payment methods & shipping details
""")