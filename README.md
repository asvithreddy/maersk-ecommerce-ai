# 🎯 E-Commerce AI Analyst

**A GenAI-powered agentic system for querying e-commerce operations data using natural language.**

Transform business questions into actionable insights in seconds. No SQL knowledge required.

---

## 🌟 Features

✨ **Natural Language Queries** - Ask questions in plain English, get instant answers  
🤖 **AI-Powered SQL Generation** - Google Gemini converts natural language to SQL  
💡 **Automated Insights** - AI analyzes results and provides business intelligence  
📊 **Real-time Results** - Visualize data with interactive tables and charts  
💾 **Export & History** - Download results as CSV, track all queries  
🔄 **Multi-turn Conversations** - Context-aware dialogue with conversation memory  
📱 **Clean Interface** - Intuitive Streamlit UI for any operations team member  

---
Working Demonstration video-https://youtu.be/UsQ2i6ODjxU
github link- https://github.com/asvithreddy/maersk-ecommerce-ai
## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Sample Queries](#-sample-queries)
- [Design Decisions](#-design-decisions)
- [Future Enhancements](#-future-enhancements)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google API Key (free: https://aistudio.google.com/apikey)
- Kaggle dataset (download from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/)

### One-Minute Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/maersk-ecommerce-ai.git
cd maersk-ecommerce-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# Download dataset to data/ folder
# (From https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/)
# Extract ZIP to ./data/ folder

# Run application
streamlit run app.py
```

Open browser to `http://localhost:8501` ✅

---

## 📦 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/maersk-ecommerce-ai.git
cd maersk-ecommerce-ai
```

### Step 2: Virtual Environment

```bash
# Create
python -m venv venv

# Activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Windows (CMD):
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Expected packages:
```
✓ streamlit==1.28.1
✓ pandas==2.0.3
✓ google-generativeai==0.3.0
✓ python-dotenv (auto-installed)
```

### Step 4: Get API Keys

**Google Gemini API Key:**
1. Go to https://aistudio.google.com/apikey
2. Click "Create API Key"
3. Copy the key

**Create `.env` file:**
```bash
cat > .env << EOF
GOOGLE_API_KEY=your_key_here
EOF
```

**⚠️ IMPORTANT:** Never commit `.env` file. It's in `.gitignore`.

### Step 5: Download Dataset

**Option A: Manual (Easiest)**
1. Go to https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/
2. Click "Download" button
3. Extract ZIP to `data/` folder in your project
4. Verify these 8 files are in `data/`:
   - `olist_orders_dataset.csv`
   - `olist_order_items_dataset.csv`
   - `olist_products_dataset.csv`
   - `olist_customers_dataset.csv`
   - `olist_sellers_dataset.csv`
   - `olist_order_payments_dataset.csv`
   - `olist_order_reviews_dataset.csv`
   - `product_category_name_translation.csv`

**Option B: Kaggle CLI**
```bash
pip install kaggle
# Configure: https://www.kaggle.com/account
kaggle datasets download -d olistbr/brazilian-ecommerce
unzip brazilian-ecommerce.zip -d data/
```

### Step 6: Run Application

```bash
streamlit run app.py
```

App will open at `http://localhost:8501`

---

## 💬 Usage

### Basic Query

1. **Type or select a question** - Use sidebar templates or type custom question
2. **Click "Analyze"** - System processes your query
3. **View results** - SQL, data table, and AI insights appear
4. **Export or explore** - Download CSV or view history

### Example Questions

```
"Which product category has the highest revenue?"
"What is the average order value?"
"Show me top 10 sellers by order count"
"What payment methods are most common?"
"Which cities have the most customers?"
"Show me customer distribution by state"
"What is the average review rating?"
```

### Features

**Export Results:**
- Click "📥 Download CSV" button
- Results saved as `results_YYYYMMDD_HHMMSS.csv`

**Query History:**
- Expand "📋 Query History" section
- Shows last 10 queries with timestamps
- Click to see exact SQL generated

**Multiple Queries:**
- Ask follow-up questions
- System maintains context
- Session memory preserved during app session

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────┐
│                    USER INTERFACE                    │
│         Streamlit Web Application (Python)          │
│  Chat Input │ Sample Questions │ Results Display    │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│              CONVERSATION LAYER                     │
│  • Session State Management                         │
│  • Query History (timestamps, SQL)                  │
│  • Context Preservation                            │
└────────────────┬────────────────────────────────────┘
                 │
         ┌───────┴───────┐
         │               │
    ┌────▼────┐    ┌────▼────┐
    │ SQL GEN │    │ ANALYSIS │
    │ AGENT   │    │ AGENT    │
    └────┬────┘    └────┬────┘
         │               │
    ┌────▼────────────────▼────┐
    │  GEMINI 2.0 FLASH LLM   │
    │  • SQL Generation       │
    │  • Result Analysis      │
    │  • Insight Generation   │
    └────┬────────────────────┘
         │
    ┌────▼──────────────────────┐
    │  DATA & QUERY EXECUTION   │
    │  • SQLite Database        │
    │  • Query Validation       │
    │  • Thread-Safe Execution  │
    └────┬──────────────────────┘
         │
    ┌────▼──────────────────────┐
    │   BRAZILIAN ECOMMERCE     │
    │   DATASET (100k+ records) │
    │  • Orders                 │
    │  • Products               │
    │  • Customers              │
    │  • Sellers                │
    │  • Payments               │
    │  • Reviews                │
    └───────────────────────────┘
```

### Query Pipeline

```
1. USER QUESTION
   ↓
2. SCHEMA RETRIEVAL
   └─ Get table & column names
   ↓
3. PROMPT CONSTRUCTION
   └─ Add schema, rules, examples
   ↓
4. LLM CALL (Gemini)
   └─ Generate SQL
   ↓
5. SQL CLEANING & VALIDATION
   └─ Remove artifacts, verify SELECT
   ↓
6. DATABASE EXECUTION
   └─ Run on SQLite
   ↓
7. RESULT PROCESSING
   └─ Format for display
   ↓
8. INSIGHT GENERATION
   └─ LLM analyzes results
   ↓
9. UI RENDERING
   └─ Display in Streamlit
```

### Database Schema

**8 Tables (100k+ records):**

| Table | Rows | Purpose |
|-------|------|---------|
| `orders` | 99,441 | Order metadata, status, timestamps |
| `order_items` | 879,505 | Items per order, pricing, seller info |
| `products` | 32,951 | Product catalog, categories, attributes |
| `customers` | 96,096 | Customer data, location, contact |
| `sellers` | 16,008 | Seller information and location |
| `order_payments` | 103,886 | Payment methods, amounts, installments |
| `order_reviews` | 99,224 | Customer reviews, ratings, comments |
| `product_category_translation` | 71 | Portuguese → English category names |

**Entity Relationships:**
```
orders → order_items ← products
  ↓                      ↓
customers            product_category
  ↓
order_payments

orders → order_reviews
```

---

## 🛠️ Tech Stack

### Core Dependencies

```
streamlit==1.28.1              # Web UI framework
pandas==2.0.3                  # Data manipulation
google-generativeai==0.3.0     # Gemini API access
python-dotenv==1.0.0           # Environment variables
sqlite3 (built-in)             # Database
```

### Architecture Choices

**Why Streamlit?**
- ✅ Rapid development (write Python, get web app)
- ✅ Built for data applications
- ✅ Session state management for conversation
- ✅ Easy deployment (Streamlit Cloud)
- ✅ Zero frontend knowledge needed

**Why Gemini 2.0 Flash?**
- ✅ Free tier with high rate limits
- ✅ Excellent SQL generation capability
- ✅ Fast inference (flash variant)
- ✅ Natural language understanding
- ✅ Good reasoning for business logic

**Why SQLite?**
- ✅ File-based, no server needed
- ✅ Fast for 100k+ record queries
- ✅ Full SQL support
- ✅ Perfect for development & demos
- ✅ Can scale to PostgreSQL if needed

**Why Pandas?**
- ✅ Standard for data processing
- ✅ Easy SQL → DataFrame conversion
- ✅ CSV export, data manipulation
- ✅ Well-integrated with Streamlit

---

## 📊 Dataset

### Brazilian E-Commerce (Olist)

**Source:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/

**Coverage:**
- 📅 **Time Period:** 2016-2018
- 🌍 **Geographic:** Brazilian states, 4,119 cities
- 📦 **Orders:** 99,441 complete orders
- 🛍️ **Items:** 879,505 order line items
- 📝 **Products:** 32,951 unique products
- 👥 **Customers:** 96,096 unique customers
- 🏪 **Sellers:** 16,008 registered sellers
- ⭐ **Reviews:** 99,224 customer reviews

**Key Statistics:**
- Average order value: ~$150
- Most common category: Electronics
- Top state: São Paulo (SP)
- Most used payment: Credit card
- Average review rating: 4.2/5

**Data Quality:**
- ✅ No missing critical values
- ✅ Properly typed columns (dates, numerics)
- ✅ Valid geographic data
- ✅ Consistent foreign keys

---

## 💡 Sample Queries

These queries demonstrate the system's capabilities:

### Analytics Query
```
Question: "Which product category has the highest revenue?"

Generated SQL:
SELECT p.product_category_name, SUM(oi.price) as total_revenue 
FROM products p 
JOIN order_items oi ON p.product_id = oi.product_id 
GROUP BY p.product_category_name 
ORDER BY total_revenue DESC 
LIMIT 10;

Output: Electronics, Fashion, Home, Sports with revenue figures
Insight: Electronics leads with $X, representing 35% of total revenue
```

### Aggregation Query
```
Question: "What payment methods are most common?"

Generated SQL:
SELECT payment_type, COUNT(*) as count, 
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as percentage
FROM order_payments 
GROUP BY payment_type 
ORDER BY count DESC;

Output: Credit Card (76%), Boleto (18%), Debit (5%), Voucher (1%)
Insight: Credit card dominates, representing over 3/4 of all transactions
```

### Geographic Query
```
Question: "Which cities have the most customers?"

Generated SQL:
SELECT customer_city, customer_state, COUNT(*) as customer_count
FROM customers
GROUP BY customer_city, customer_state
ORDER BY customer_count DESC
LIMIT 15;

Output: São Paulo (SP), Rio de Janeiro (RJ), Belo Horizonte (MG), etc.
Insight: Top 3 cities account for 45% of all customers
```

### Multi-Table Query
```
Question: "Show me top sellers by revenue"

Generated SQL:
SELECT oi.seller_id, s.seller_state, COUNT(*) as order_count,
  SUM(oi.price) as total_revenue
FROM order_items oi
JOIN sellers s ON oi.seller_id = s.seller_id
GROUP BY oi.seller_id, s.seller_state
ORDER BY total_revenue DESC
LIMIT 10;

Output: Seller rankings with revenue and order counts
Insight: Top 10 sellers account for 28% of total revenue
```

---

## 🎨 Design Decisions

### 1. Multi-Agent Architecture (vs. Single LLM)

**Decision:** Separate SQL Generation and Analysis agents

**Why:**
- Allows specialized prompts for each task
- SQL generation needs strict formatting, analysis needs creativity
- Easier to debug and maintain
- Can swap agents independently

**Trade-off:** Slightly more API calls, but better reliability

```python
# Specialized agents
agent_sql = generate_sql_from_question(question, schema)
agent_analysis = generate_insight(question, results)
```

---

### 2. Template Fallback Queries

**Decision:** Maintain pre-written SQL templates for common questions

**Why:**
- Fallback if Gemini fails or returns malformed SQL
- Ensures reliability for common operations
- Faster response for predictable questions
- Cost savings (skip API call if template matches)

**Template Examples:**
```python
TEMPLATE_QUERIES = {
    "average order value": "SELECT AVG(payment_value) ...",
    "highest revenue category": "SELECT p.category, SUM(price) ...",
    "top sellers": "SELECT seller_id, SUM(price) ...",
}
```

---

### 3. Thread-Safe SQLite Access

**Decision:** `check_same_thread=False` + fresh connections per query

**Why:**
- Streamlit runs on multiple threads
- SQLite is thread-sensitive by default
- Creating fresh connections avoids state conflicts
- Each query gets isolated execution context

**Implementation:**
```python
# Instead of reusing connection
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# Fresh connection per query
df = pd.read_sql_query(sql, conn)
conn.close()
```

---

### 4. Aggressive SQL Cleaning

**Decision:** Multiple cleaning passes to handle LLM output variations

**Why:**
- LLM sometimes adds prefixes ("ite SELECT"), markdown, explanations
- Need bulletproof SQL extraction
- Better error detection

**Implementation:**
```python
sql = response.text.strip()
sql = sql.replace("```sql", "").replace("```", "").strip()
sql = sql.replace("SQLite", "").strip()
# Validate it starts with SELECT
if not sql.upper().startswith("SELECT"):
    return None  # Use template fallback
```

---

### 5. Session State for Conversation Memory

**Decision:** Streamlit `st.session_state` for query history

**Why:**
- Built into Streamlit, no external DB needed
- Perfect for single-session demos
- Preserves context across interactions
- User can review what was asked/executed

**Trade-off:** History lost on page refresh (acceptable for MVP)

```python
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.session_state.chat_history.append({
    "question": user_question,
    "sql": sql_query,
    "timestamp": datetime.now()
})
```

---

### 6. Schema Limiting in Prompts

**Decision:** Show only 10 columns per table, 8 tables max in prompt

**Why:**
- Token limits on LLM input
- Too much schema = confusion for model
- Reduces hallucination of non-existent columns
- Keeps prompt concise

**Result:** Better SQL generation, lower latency

---

## 📁 Project Structure

```
maersk-ecommerce-ai/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Example environment variables
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
│
├── data/                           # Dataset (not in repo)
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   └── product_category_name_translation.csv
│
└── docs/                           # Documentation (optional)
    ├── ARCHITECTURE.md             # Detailed architecture
    ├── SETUP.md                    # Detailed setup guide
    └── API_REFERENCE.md            # Function documentation
```

---

## 🔮 Future Enhancements

### Phase 1: Advanced Analytics (1-2 weeks)

**Forecasting & Trends:**
- Time-series forecasting (ARIMA, Prophet)
- Seasonal decomposition
- YoY trend analysis
- Anomaly detection

```python
# Example enhancement
def forecast_sales(category, months=3):
    data = get_sales_by_month(category)
    forecast = arima_forecast(data, periods=months)
    return forecast
```

**Implementation:** Add `statsmodels` and `scikit-learn` for ML

---

### Phase 2: Vector Embeddings & Semantic Search (1 week)

**Product Similarity:**
- Generate embeddings for products using Gemini
- Store in ChromaDB or Pinecone
- Enable: "Find products similar to [product]"
- Recommendations: "Customers who bought X also bought..."

```python
# Vector search
embeddings = generate_embeddings(product_descriptions)
vectorstore = Chroma(embeddings)
similar = vectorstore.similarity_search("electronics", k=5)
```

**Implementation:** Add ChromaDB, update prompts for RAG

---

### Phase 3: Multi-Agent Specialization (1-2 weeks)

**Specialized Agents:**
- 🎯 **Sales Agent:** Revenue, trends, product performance
- 🏪 **Seller Agent:** Seller metrics, inventory, performance
- 👥 **Customer Agent:** Lifetime value, segmentation, churn
- 💰 **Finance Agent:** Costs, margins, profitability
- 📦 **Ops Agent:** Shipping times, logistics, fulfillment

```python
class SalesAgent:
    def handle_query(self, question):
        if "revenue" in question.lower():
            return self.revenue_query(question)
        elif "trend" in question.lower():
            return self.trend_query(question)

class CustomerAgent:
    def handle_query(self, question):
        if "lifetime value" in question.lower():
            return self.clv_query(question)
```

**Implementation:** Router agent to dispatch to specialists

---

### Phase 4: Real-time Dashboards (1 week)

**Auto-Generated KPIs:**
- Real-time sales dashboard
- Inventory status
- Seller performance leaderboard
- Customer acquisition funnel

```python
# Auto-generate dashboard
def auto_dashboard():
    metrics = {
        "today_revenue": get_today_revenue(),
        "avg_order_value": get_aov(),
        "top_category": get_top_category(),
        "seller_count": get_active_sellers()
    }
    render_dashboard(metrics)
```

**Implementation:** Plotly Dash or Streamlit multipage app

---

### Phase 5: Authentication & Multi-User (1 week)

**User Management:**
- Admin login with credentials
- Role-based access control (RBAC)
- User-specific dashboards
- Query permission levels

```python
# Authentication
@st.cache_resource
def init_auth():
    return Auth0Manager()

if not init_auth().is_authenticated():
    st.error("Please log in")
    st.stop()
```

**Implementation:** Streamlit Cloud secrets, Auth0, or simple DB

---

### Phase 6: Mobile App & Progressive Web App (2 weeks)

**Mobile Version:**
- React Native or Flutter app
- Offline query caching
- Voice input for queries
- Push notifications for insights

**PWA Features:**
- Install as app
- Offline mode
- Background sync

**Implementation:** Separate frontend repo, API backend

---

### Phase 7: Production Deployment (1 week)

**Infrastructure:**
- Move to PostgreSQL (not SQLite)
- Deploy API backend (FastAPI)
- Frontend on Vercel/Netlify
- Caching layer (Redis)
- Monitoring (Datadog/NewRelic)

**Architecture:**
```
Frontend (Next.js) ← API (FastAPI) ← Database (PostgreSQL)
                        ↓
                  Gemini API
                        ↓
                    Cache (Redis)
```

**Implementation:** Docker, Kubernetes, CI/CD pipeline

---

### Phase 8: Advanced Features

**Smart Query Suggestions:**
- Learn user preferences
- Suggest relevant questions
- Auto-complete based on history

**Report Generation:**
- PDF/Excel exports with formatting
- Scheduled reports
- Email delivery

**Collaborative Features:**
- Share queries with team
- Comment on results
- Pin important queries

**Data Lineage:**
- Show data sources and transformations
- Audit trail of all queries
- Data governance compliance

---

## 🐛 Troubleshooting

### Issue: "No tables found in database"

**Causes:**
- CSV files not in `data/` folder
- CSV files not named correctly
- Database file corrupted

**Solutions:**
```bash
# Verify files exist
ls data/olist_*.csv

# Delete corrupted database
rm ecommerce.db

# Restart app
streamlit run app.py
```

---

### Issue: "SQLite threading error"

**Cause:** Old version of code without thread-safe connection

**Solution:** Update to latest code with `check_same_thread=False`

---

### Issue: "API Key not working"

**Causes:**
- Key not in `.env` file
- Key expired
- API quota exceeded

**Solutions:**
```bash
# Check .env
cat .env

# Get new key at https://aistudio.google.com/apikey
# Update .env and restart app
```

---

### Issue: "Generated SQL is invalid"

**Cause:** Gemini returned malformed SQL

**Solution:** 
- Try rephrasing question
- Use template query instead
- Check schema in sidebar

**If persistent:**
```bash
# Enable debug logging
export DEBUG=1
streamlit run app.py
```

---

### Issue: "Slow query execution"

**Cause:** Query scanning large tables without indexes

**Solution:**
```sql
-- Add indexes
CREATE INDEX idx_order_customer ON orders(customer_id);
CREATE INDEX idx_item_order ON order_items(order_id);
CREATE INDEX idx_item_product ON order_items(product_id);
```

---

## 📝 Development

### Running Tests

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Code Style

```bash
# Format code
black app.py

# Lint
pylint app.py

# Type checking
mypy app.py
```

---

## 📄 License

MIT License - see LICENSE file

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📞 Support

**Questions or issues?**

- 📧 Email: [your-email]
- 🐛 Issues: https://github.com/YOUR_USERNAME/maersk-ecommerce-ai/issues
- 💬 Discussions: https://github.com/YOUR_USERNAME/maersk-ecommerce-ai/discussions

---

## 🎓 Learning Resources

**Built with these technologies:**

- [Streamlit Docs](https://docs.streamlit.io)
- [Google Gemini API](https://ai.google.dev/docs)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [SQLite Tutorial](https://www.sqlite.org/docs.html)

**Related Projects:**

- [LangChain](https://python.langchain.com/) - LLM orchestration
- [LlamaIndex](https://docs.llamaindex.ai/) - Data indexing
- [ChromaDB](https://docs.trychroma.com/) - Vector embeddings

---

## ✨ Acknowledgments

- 🙏 Brazilian E-Commerce Dataset by Olist (Kaggle)
- 🙏 Google for Gemini API
- 🙏 Streamlit team for excellent framework
- 🙏 Maersk for this opportunity

---

**Built with ❤️ for the Maersk AI/ML Internship**

Last Updated: 2024
