

import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime
import json
import zipfile
from pathlib import Path


import google.generativeai as genai


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-key-here")
if GOOGLE_API_KEY != "your-key-here":
    genai.configure(api_key=GOOGLE_API_KEY)


st.set_page_config(page_title="E-Commerce AI Analyst", layout="wide", initial_sidebar_state="expanded")


st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .stChatMessage { padding: 1rem; border-radius: 8px; }
    h1 { color: #1f77b4; }
</style>
""", unsafe_allow_html=True)


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
        
        Path(DATA_FOLDER).mkdir(exist_ok=True)
        
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
        f"Table: {table}\n  Columns: {', '.join([f'{col[0]} ({col[1]})' for col in cols[:10]])}"  
        for table, cols in list(schema.items())[:8] 
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
        

        sql = sql.replace("```sql", "").replace("```", "").strip()
        sql = sql.replace("SQLite", "").strip()
        sql = sql.replace("sql", "").strip()
        
        prefixes = ["ite ", "SELECT", "qlite ", "- SELECT"]
        while any(sql.startswith(p) for p in prefixes if not sql.startswith("SELECT")):
            for prefix in prefixes:
                if sql.startswith(prefix) and not sql.startswith("SELECT"):
                    sql = sql[len(prefix):].strip()
                    break
        
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
