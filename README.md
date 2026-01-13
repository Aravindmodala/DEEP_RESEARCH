# 🏦 Deep Research Agent
## Restaurant Market Analysis for Commercial Banking

A sophisticated **multi-agent AI system** designed for comprehensive restaurant market analysis, built for commercial banking decisions including lending risk assessment, expansion viability, and competitive intelligence.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-purple.svg)

---

## 🎯 Overview

This system implements a **four-node agentic architecture** that autonomously researches, analyzes, and generates decision-grade market intelligence reports suitable for commercial banking professionals.

### Agent Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│   PLANNER   │───▶│  RESEARCHER  │───▶│   CRITIC    │───▶│   REPORT    │
│             │    │  (ReAct)     │    │             │    │             │
│ Parse Query │    │ Execute Plan │    │ QA Review   │    │ Generate    │
│ Create Plan │    │ Use Tools    │    │ Accept/     │    │ Final       │
│             │    │ Synthesize   │    │ Reject      │    │ Report      │
└─────────────┘    └──────────────┘    └──────┬──────┘    └─────────────┘
                          ▲                   │
                          │     REJECT        │
                          └───────────────────┘
```

### Key Features

- 🔍 **Autonomous Research**: ReAct-style agent loop with tool orchestration
- 🗺️ **Competitor Discovery**: Google Maps/Places API integration
- 📊 **Menu Intelligence**: Web scraping for pricing and menu analysis
- 💬 **Sentiment Analysis**: Customer review aggregation and theme extraction
- 📈 **Market Signals**: Foot traffic, saturation, and demand indicators
- ✅ **Quality Assurance**: Bank-grade critic with accept/reject loops
- 📄 **Decision-Grade Reports**: Professional reports for credit officers

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- API Keys (see Configuration below)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd DRA

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and configure
copy env.template .env  # Windows
# cp env.template .env  # macOS/Linux

# Edit .env with your API keys
```

---

## ⚙️ Configuration

### Required API Keys

1. **OpenAI API** (or Anthropic)
   - Get key at: https://platform.openai.com/
   - Used for: LLM reasoning and generation

2. **Google Maps API**
   - Get key at: https://console.cloud.google.com/google/maps-apis
   - Enable: Places API, Geocoding API, Distance Matrix API
   - Used for: Restaurant discovery, ratings, reviews

3. **Tavily Search API**
   - Get key at: https://tavily.com/
   - Used for: Web research, market trends, news

### Environment Variables

```env
# LLM Provider
OPENAI_API_KEY=sk-your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Research Tools
GOOGLE_MAPS_API_KEY=your-google-maps-key
TAVILY_API_KEY=tvly-your-tavily-key

# Optional Configuration
LLM_MODEL=gpt-4o
MAX_RESEARCH_ITERATIONS=3
MIN_QUALITY_SCORE=0.7
```

---

## 🚀 Usage

### Command Line

```bash
# Single query
python main.py "Analyze Chipotle in Austin, TX for lending risk assessment"

# With output file
python main.py "Research Five Guys in Seattle, WA" -o report_five_guys

# Interactive mode
python main.py --interactive

# With debug logging
python main.py "Analyze restaurant" --log-level DEBUG
```

### Programmatic Usage

```python
from orchestrator import DeepResearchOrchestrator
from config import get_config

# Initialize
config = get_config()
orchestrator = DeepResearchOrchestrator(config)

# Run research
result = orchestrator.run(
    "Analyze Olive Garden in Denver, CO for expansion viability"
)

# Access outputs
planner_output = result["planner_output"]
researcher_output = result["researcher_output"]
critic_output = result["critic_output"]
report_output = result["report_output"]

# Format as Markdown
from orchestrator import format_report_as_markdown
markdown = format_report_as_markdown(report_output)
print(markdown)
```

### Streaming Mode

```python
for event in orchestrator.run_streaming(query):
    node_name = list(event.keys())[0]
    print(f"Completed: {node_name}")
```

---

## 🏗️ Architecture Details

### Node 1: PLANNER

**Responsibility**: Convert natural language query into structured research plan.

**Output Schema**:
```json
{
  "target_restaurant": "Chipotle",
  "location": "Austin, TX",
  "intent": "lending risk assessment",
  "search_queries": [
    "Chipotle competitors Mexican restaurants Austin TX",
    "Chipotle menu pricing vs competitors Austin",
    "Chipotle customer reviews satisfaction Austin",
    "Austin TX restaurant market demand foot traffic"
  ]
}
```

### Node 2: RESEARCHER

**Responsibility**: Execute research plan autonomously using ReAct loop.

**Available Tools**:
| Tool | Description |
|------|-------------|
| `find_restaurant` | Locate target restaurant via Google Places |
| `find_competitors` | Discover nearby competitors |
| `get_restaurant_reviews` | Fetch Google reviews |
| `get_restaurant_details` | Get hours, website, phone |
| `web_search` | General web search via Tavily |
| `search_restaurant_market` | Restaurant-specific market search |
| `search_foot_traffic` | Local demand indicators |
| `extract_menu` | Parse menu from website |
| `crawl_restaurant_site` | Crawl restaurant website |

**Output**: Structured research packet with competitors, menus, sentiment, market signals.

### Node 3: CRITIC

**Responsibility**: Bank-grade quality assurance review.

**Evaluation Criteria**:
- ✅ Completeness (30%): All required data present
- ✅ Accuracy (30%): Data grounded in sources
- ✅ Logic (20%): Reasonable inferences
- ✅ Banking Relevance (20%): Suitable for decisions

**Decisions**:
- `ACCEPT`: Quality ≥ 70%, no critical issues → proceed to REPORT
- `REJECT`: Quality < 70% or critical issues → return to RESEARCHER

### Node 4: REPORT

**Responsibility**: Generate client-ready market research report.

**Report Sections**:
1. Executive Summary
2. Competitive Landscape
3. Menu & Pricing Position
4. Customer Sentiment
5. Commercial Banking Relevance
6. Final Recommendation
7. Appendix & Disclaimers

---

## 📊 Sample Output

```markdown
# Market Research Report: Chipotle

**Generated:** 2025-01-07T10:30:00
**Location:** Austin, TX

## Executive Summary

This report presents a comprehensive market analysis of Chipotle 
located in Austin, TX, conducted to support lending risk assessment.

### Key Findings
- 12 direct competitors identified within 2-mile radius
- Market saturation level: moderate
- Target restaurant positioned in the mid-range segment
- Customer sentiment score: 78%
- Average competitor rating: 4.2/5.0

**Lending Implications:** Strong market position supports favorable 
lending consideration.

**Expansion Implications:** Moderate competition indicates selective 
expansion opportunities.

## Final Recommendation

**Outlook:** MODERATE
**Confidence:** MEDIUM

### Recommendation
Standard underwriting recommended. Market position is stable.

### Next Steps
1. Request detailed financial statements
2. Schedule site visit
3. Complete standard underwriting process
```

---

## 🛠️ Development

### Project Structure

```
DRA/
├── main.py              # CLI entry point
├── orchestrator.py      # LangGraph workflow orchestration
├── config.py            # Configuration and system prompts
├── requirements.txt     # Dependencies
├── env.template         # Environment template
│
├── agents/              # Agent node implementations
│   ├── __init__.py
│   ├── planner.py       # PLANNER agent
│   ├── researcher.py    # RESEARCHER agent (ReAct)
│   ├── critic.py        # CRITIC agent
│   └── report.py        # REPORT agent
│
├── models/              # Pydantic schemas
│   ├── __init__.py
│   └── schemas.py       # All data models
│
├── tools/               # Research tool implementations
│   ├── __init__.py
│   ├── google_maps.py   # Google Maps/Places
│   ├── tavily_search.py # Tavily web search
│   └── web_scraper.py   # Web scraping utilities
│
└── logs/                # Log files (auto-created)
```

### Adding New Tools

1. Create tool module in `tools/`
2. Implement tool class with retry logic
3. Create LangChain-compatible wrapper functions
4. Register tools in `ResearcherAgent._initialize_tools()`

### Extending the Critic

Edit `config.py` to modify:
- Quality score thresholds
- Required data fields
- Evaluation criteria

---

## 🔒 Security Notes

- API keys are loaded from environment variables
- No credentials are logged or stored
- Scraping respects robots.txt and rate limits
- All external calls use HTTPS

---

## 📜 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

Built with:
- [LangChain](https://langchain.com/) - LLM framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent orchestration
- [Google Maps Platform](https://developers.google.com/maps) - Location intelligence
- [Tavily](https://tavily.com/) - AI-powered search
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [Rich](https://rich.readthedocs.io/) - Terminal formatting

---

<div align="center">
  <b>Built for Commercial Banking Intelligence</b>
</div>


