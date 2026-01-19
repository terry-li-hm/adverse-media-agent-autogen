# Adverse Media Screening Agent (AutoGen)

Multi-agent adverse media screening system built with Microsoft AutoGen framework.

## Overview

This tool screens entities (people or companies) for adverse media mentions using a 4-agent workflow:

1. **Research Agent** - Generates targeted search queries
2. **Analysis Agent** - Categorizes findings by risk type
3. **Risk Agent** - Calculates risk score (0-100)
4. **Report Agent** - Generates executive summary and detailed report

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI Interface                          │
│          python main.py "Entity Name" --type person         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Research │──▶│ Analysis │──▶│   Risk   │──▶│  Report  │ │
│  │  Agent   │   │  Agent   │   │  Agent   │   │  Agent   │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       │              │                                      │
│       └──────────────┘                                      │
│       (loop if needs deeper research, max 3 iterations)     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│   OpenRouter (LLM)              Tavily (Search)             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Set environment variables
export OPENROUTER_API_KEY=your-key
export TAVILY_API_KEY=your-key

# Run a screening (uv handles dependencies automatically)
uv run main.py "Elizabeth Holmes" --type person

# Or test with mock data (no TAVILY_API_KEY needed)
uv run main.py "Elizabeth Holmes" --type person --mock
```

## Usage

```bash
# Screen a person
uv run main.py "Elizabeth Holmes" --type person

# Screen a company
uv run main.py "Wirecard" --type company

# Save detailed report to file
uv run main.py "Enron" --type company --output report.md

# Output as JSON
uv run main.py "Bernie Madoff" --type person --json

# Quiet mode (no progress output)
uv run main.py "FTX" --type company --quiet

# Mock mode (testing without API keys)
uv run main.py "Elizabeth Holmes" --type person --mock
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key |
| `OPENROUTER_MODEL` | No | Model to use (default: `anthropic/claude-sonnet-4`) |
| `TAVILY_API_KEY` | Yes | Tavily search API key |

## Example Output

```
[1/3] Research phase...
    Generated 4 search queries
    Searching: Elizabeth Holmes fraud scandal...
    Found 12 new articles (total: 12)

[1/3] Analysis phase...
    Found 5 findings
    Needs deeper research: False

[*] Risk assessment phase...
    Risk Level: CRITICAL
    Risk Score: 95/100

[*] Report generation phase...
    Report generated

============================================================
SCREENING COMPLETE: Elizabeth Holmes
============================================================

Risk Level: CRITICAL
Risk Score: 95/100

Key Concerns:
  - Criminal fraud conviction (2022)
  - SEC civil fraud charges
  - Wire fraud affecting investors and patients

Recommendation: Do not proceed with business relationship.

Executive Summary:
Elizabeth Holmes, founder of Theranos, was convicted of wire fraud
in 2022 and sentenced to over 11 years in federal prison...
```

## Risk Categories

- Fraud
- Sanctions
- Litigation
- Regulatory violations
- Financial crime
- Corruption
- Bankruptcy
- Environmental issues
- Human rights
- Terrorism
- PEP (Politically Exposed Person)

## Risk Levels

| Level | Score | Description |
|-------|-------|-------------|
| None | 0-10 | No adverse media found |
| Low | 11-30 | Minor concerns, routine monitoring |
| Medium | 31-60 | Notable concerns, enhanced due diligence |
| High | 61-85 | Significant concerns, senior review required |
| Critical | 86-100 | Severe concerns, potential deal-breaker |

## Tech Stack

- **AutoGen** - Multi-agent orchestration
- **OpenRouter** - LLM gateway (supports Claude, GPT-4, etc.)
- **Tavily** - News search API
- **Pydantic** - Data validation and typed state

## License

MIT
