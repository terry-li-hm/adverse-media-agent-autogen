#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "autogen-agentchat>=0.4",
#     "autogen-ext[openai]>=0.4",
#     "httpx>=0.27",
#     "pydantic>=2.0",
# ]
# ///
"""
Adverse Media Screening Agent using Microsoft AutoGen.

A multi-agent system that screens entities for adverse media mentions.
Demonstrates AutoGen's multi-agent orchestration capabilities.

Usage:
    uv run main.py "Elizabeth Holmes" --type person
    uv run main.py "Wirecard" --type company
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from enum import Enum
from typing import Any

import httpx
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

def get_env(key: str, default: str | None = None) -> str:
    """Get environment variable or raise error."""
    value = os.environ.get(key, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


# =============================================================================
# Models (Typed State)
# =============================================================================

class EntityType(str, Enum):
    PERSON = "person"
    COMPANY = "company"


class Article(BaseModel):
    """A news article from search results."""
    title: str
    url: str
    content: str
    source: str = ""
    published_date: str = ""
    relevance_score: float = 0.0


class Finding(BaseModel):
    """An adverse media finding from analysis."""
    category: str  # fraud, sanctions, litigation, etc.
    severity: str  # low, medium, high, critical
    description: str
    source_urls: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class RiskAssessment(BaseModel):
    """Risk assessment result."""
    risk_level: str  # none, low, medium, high, critical
    risk_score: int  # 0-100
    key_concerns: list[str] = Field(default_factory=list)
    recommendation: str = ""


class ScreeningState(BaseModel):
    """State passed between agents."""
    # Input
    entity_name: str
    entity_type: EntityType

    # Research phase
    search_queries: list[str] = Field(default_factory=list)
    articles: list[Article] = Field(default_factory=list)
    research_iterations: int = 0

    # Analysis phase
    findings: list[Finding] = Field(default_factory=list)
    needs_deeper_research: bool = False
    additional_search_terms: list[str] = Field(default_factory=list)

    # Risk assessment phase
    risk_assessment: RiskAssessment | None = None

    # Report phase
    executive_summary: str = ""
    detailed_report: str = ""


# =============================================================================
# Tools
# =============================================================================

# Mock data for demo/testing without API keys
MOCK_ARTICLES = {
    "elizabeth holmes": [
        {"title": "Elizabeth Holmes Sentenced to 11 Years in Theranos Fraud Case", "url": "https://example.com/1", "content": "Elizabeth Holmes, founder of Theranos, was sentenced to more than 11 years in federal prison for defrauding investors in her blood-testing startup.", "score": 0.95},
        {"title": "SEC Charges Theranos CEO Elizabeth Holmes with Fraud", "url": "https://example.com/2", "content": "The Securities and Exchange Commission charged Elizabeth Holmes with massive fraud for deceiving investors about the company's blood-testing technology.", "score": 0.92},
        {"title": "The Rise and Fall of Elizabeth Holmes and Theranos", "url": "https://example.com/3", "content": "From Stanford dropout to billionaire to convicted felon, Elizabeth Holmes's story is a cautionary tale about fraud in Silicon Valley.", "score": 0.88},
    ],
    "wirecard": [
        {"title": "Wirecard Collapses After $2 Billion Fraud Scandal", "url": "https://example.com/4", "content": "German payments company Wirecard filed for insolvency after admitting that 1.9 billion euros supposedly held in trust accounts probably don't exist.", "score": 0.95},
        {"title": "Wirecard CEO Arrested on Fraud Charges", "url": "https://example.com/5", "content": "Markus Braun, former CEO of Wirecard, was arrested on suspicion of falsifying accounts and market manipulation in one of Germany's biggest fraud cases.", "score": 0.93},
    ],
    "default": [
        {"title": "No significant adverse media found", "url": "https://example.com/none", "content": "Search returned no significant adverse media findings for this entity.", "score": 0.5},
    ],
}


async def search_news(query: str, max_results: int = 10, mock: bool = False) -> list[dict]:
    """
    Search for news articles using Tavily API.

    Args:
        query: Search query string
        max_results: Maximum number of results
        mock: Use mock data instead of real API

    Returns:
        List of article dictionaries
    """
    if mock:
        # Return mock data for testing
        query_lower = query.lower()
        for key, articles in MOCK_ARTICLES.items():
            if key in query_lower:
                return articles[:max_results]
        return MOCK_ARTICLES["default"][:max_results]

    api_key = get_env("TAVILY_API_KEY")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": False,
                "max_results": max_results,
            },
        )
        response.raise_for_status()
        data = response.json()

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "")[:500],  # Truncate for token efficiency
            "score": r.get("score", 0.0),
        }
        for r in data.get("results", [])
    ]


# =============================================================================
# Agent Prompts
# =============================================================================

RESEARCH_SYSTEM_PROMPT = """You are a Research Agent specializing in adverse media screening.

Your task is to generate effective search queries to find news about potential adverse media
for the given entity. Focus on queries that would surface:
- Fraud, scandals, or misconduct
- Legal issues, lawsuits, or regulatory actions
- Sanctions or watchlist mentions
- Financial crimes or money laundering
- Corruption or bribery
- Bankruptcy or financial distress

Return your response as JSON with this structure:
{
    "search_queries": ["query1", "query2", "query3"]
}

Generate 3-5 targeted search queries. Include the entity name in each query."""

ANALYSIS_SYSTEM_PROMPT = """You are an Analysis Agent specializing in adverse media classification.

Your task is to analyze news articles and identify adverse media findings.
Categorize each finding by type and severity.

Categories: fraud, sanctions, litigation, regulatory, financial_crime, corruption,
           bankruptcy, environmental, human_rights, terrorism, pep, other

Severity levels: low, medium, high, critical

Return your response as JSON with this structure:
{
    "findings": [
        {
            "category": "category_name",
            "severity": "severity_level",
            "description": "Brief description of the adverse finding",
            "confidence": 0.0-1.0
        }
    ],
    "needs_deeper_research": true/false,
    "additional_search_terms": ["term1", "term2"] (only if needs_deeper_research is true)
}

Be thorough but avoid false positives. Only flag genuine adverse media."""

RISK_SYSTEM_PROMPT = """You are a Risk Assessment Agent specializing in compliance risk scoring.

Your task is to evaluate findings and calculate an overall risk score.

Risk levels based on score:
- none (0-10): No adverse media found
- low (11-30): Minor concerns, routine monitoring
- medium (31-60): Notable concerns, enhanced due diligence recommended
- high (61-85): Significant concerns, senior review required
- critical (86-100): Severe concerns, potential deal-breaker

Return your response as JSON with this structure:
{
    "risk_level": "level_name",
    "risk_score": 0-100,
    "key_concerns": ["concern1", "concern2"],
    "recommendation": "Your recommendation for next steps"
}

Base your assessment on the severity, recency, and credibility of findings."""

REPORT_SYSTEM_PROMPT = """You are a Report Agent specializing in compliance documentation.

Your task is to generate a clear, professional adverse media screening report.

Return your response as JSON with this structure:
{
    "executive_summary": "2-3 sentence summary of findings and risk level",
    "detailed_report": "Full markdown report with sections for Overview, Findings, Risk Assessment, and Recommendations"
}

The detailed report should be suitable for compliance review and audit purposes."""


# =============================================================================
# Agent Execution
# =============================================================================

def create_model_client() -> OpenAIChatCompletionClient:
    """Create OpenRouter-compatible model client."""
    return OpenAIChatCompletionClient(
        model=get_env("OPENROUTER_MODEL", "anthropic/claude-sonnet-4"),
        api_key=get_env("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "claude",
            "structured_output": True,
        },
    )


async def run_agent(
    agent: AssistantAgent,
    prompt: str,
) -> str:
    """Run an agent and return its response."""
    response = await agent.on_messages(
        [TextMessage(content=prompt, source="orchestrator")],
        CancellationToken(),
    )
    return response.chat_message.content


def parse_json_response(response: str) -> dict[str, Any]:
    """Parse JSON from agent response, handling markdown code blocks."""
    # Remove markdown code blocks if present
    content = response.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last lines (```json and ```)
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"Could not parse JSON from response: {content[:200]}")


# =============================================================================
# Orchestrator
# =============================================================================

async def run_screening(
    entity_name: str,
    entity_type: str,
    max_iterations: int = 3,
    verbose: bool = True,
    mock: bool = False,
) -> ScreeningState:
    """
    Run the full adverse media screening workflow.

    Args:
        entity_name: Name of entity to screen
        entity_type: Type of entity (person/company)
        max_iterations: Maximum research iterations
        verbose: Print progress updates
        mock: Use mock data instead of real API calls

    Returns:
        Final screening state with results
    """
    # Initialize state
    state = ScreeningState(
        entity_name=entity_name,
        entity_type=EntityType(entity_type),
    )

    # Create model client
    model_client = create_model_client()

    # Create agents
    research_agent = AssistantAgent(
        name="ResearchAgent",
        model_client=model_client,
        system_message=RESEARCH_SYSTEM_PROMPT,
    )

    analysis_agent = AssistantAgent(
        name="AnalysisAgent",
        model_client=model_client,
        system_message=ANALYSIS_SYSTEM_PROMPT,
    )

    risk_agent = AssistantAgent(
        name="RiskAgent",
        model_client=model_client,
        system_message=RISK_SYSTEM_PROMPT,
    )

    report_agent = AssistantAgent(
        name="ReportAgent",
        model_client=model_client,
        system_message=REPORT_SYSTEM_PROMPT,
    )

    # Research loop
    for iteration in range(max_iterations):
        state.research_iterations = iteration + 1

        if verbose:
            print(f"\n[{iteration + 1}/{max_iterations}] Research phase...")

        # Generate search queries
        if iteration == 0:
            research_prompt = f"""Generate search queries for adverse media screening.

Entity: {entity_name}
Type: {entity_type}

Generate targeted queries to find any negative news, scandals, legal issues, or regulatory concerns."""
        else:
            research_prompt = f"""Generate additional search queries based on previous findings.

Entity: {entity_name}
Additional search terms to explore: {state.additional_search_terms}

Generate targeted queries to investigate these specific concerns deeper."""

        response = await run_agent(research_agent, research_prompt)
        research_result = parse_json_response(response)

        queries = research_result.get("search_queries", [])
        state.search_queries.extend(queries)

        if verbose:
            print(f"    Generated {len(queries)} search queries")

        # Execute searches
        all_articles = []
        for query in queries:
            if verbose:
                print(f"    Searching: {query[:50]}...")
            try:
                results = await search_news(query, max_results=5, mock=mock)
                for r in results:
                    all_articles.append(Article(
                        title=r["title"],
                        url=r["url"],
                        content=r["content"],
                        relevance_score=r.get("score", 0.0),
                    ))
            except Exception as e:
                if verbose:
                    print(f"    Warning: Search failed - {e}")

        # Deduplicate by URL
        seen_urls = {a.url for a in state.articles}
        new_articles = [a for a in all_articles if a.url not in seen_urls]
        state.articles.extend(new_articles)

        if verbose:
            print(f"    Found {len(new_articles)} new articles (total: {len(state.articles)})")

        # Analysis phase
        if verbose:
            print(f"\n[{iteration + 1}/{max_iterations}] Analysis phase...")

        if not state.articles:
            if verbose:
                print("    No articles to analyze")
            break

        articles_text = "\n\n".join([
            f"Title: {a.title}\nURL: {a.url}\nContent: {a.content}"
            for a in state.articles[-20:]  # Limit to recent 20
        ])

        analysis_prompt = f"""Analyze these news articles for adverse media about {entity_name}.

Articles:
{articles_text}

Identify any adverse findings and assess if deeper research is needed."""

        response = await run_agent(analysis_agent, analysis_prompt)
        analysis_result = parse_json_response(response)

        # Update findings
        for f in analysis_result.get("findings", []):
            state.findings.append(Finding(
                category=f.get("category", "other"),
                severity=f.get("severity", "low"),
                description=f.get("description", ""),
                confidence=f.get("confidence", 0.5),
            ))

        state.needs_deeper_research = analysis_result.get("needs_deeper_research", False)
        state.additional_search_terms = analysis_result.get("additional_search_terms", [])

        if verbose:
            print(f"    Found {len(analysis_result.get('findings', []))} findings")
            print(f"    Needs deeper research: {state.needs_deeper_research}")

        # Check if we should continue
        if not state.needs_deeper_research:
            break

    # Risk assessment phase
    if verbose:
        print("\n[*] Risk assessment phase...")

    findings_text = "\n".join([
        f"- [{f.severity.upper()}] {f.category}: {f.description}"
        for f in state.findings
    ]) or "No adverse findings identified."

    risk_prompt = f"""Assess the risk level for {entity_name} based on these findings:

{findings_text}

Total articles screened: {len(state.articles)}
Research iterations: {state.research_iterations}"""

    response = await run_agent(risk_agent, risk_prompt)
    risk_result = parse_json_response(response)

    state.risk_assessment = RiskAssessment(
        risk_level=risk_result.get("risk_level", "none"),
        risk_score=risk_result.get("risk_score", 0),
        key_concerns=risk_result.get("key_concerns", []),
        recommendation=risk_result.get("recommendation", ""),
    )

    if verbose:
        print(f"    Risk Level: {state.risk_assessment.risk_level.upper()}")
        print(f"    Risk Score: {state.risk_assessment.risk_score}/100")

    # Report phase
    if verbose:
        print("\n[*] Report generation phase...")

    report_prompt = f"""Generate an adverse media screening report.

Entity: {entity_name}
Type: {entity_type}
Screening Date: {datetime.now().strftime('%Y-%m-%d')}

Findings:
{findings_text}

Risk Assessment:
- Level: {state.risk_assessment.risk_level}
- Score: {state.risk_assessment.risk_score}/100
- Key Concerns: {', '.join(state.risk_assessment.key_concerns) or 'None'}
- Recommendation: {state.risk_assessment.recommendation}

Articles Screened: {len(state.articles)}
Research Iterations: {state.research_iterations}"""

    response = await run_agent(report_agent, report_prompt)
    report_result = parse_json_response(response)

    state.executive_summary = report_result.get("executive_summary", "")
    state.detailed_report = report_result.get("detailed_report", "")

    if verbose:
        print("    Report generated")

    return state


# =============================================================================
# CLI
# =============================================================================

def cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Adverse Media Screening using AutoGen multi-agent framework"
    )
    parser.add_argument(
        "entity",
        help="Name of entity to screen",
    )
    parser.add_argument(
        "--type", "-t",
        choices=["person", "company"],
        default="person",
        help="Type of entity (default: person)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for detailed report (markdown)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--max-iterations", "-m",
        type=int,
        default=3,
        help="Maximum research iterations (default: 3)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data (for testing without API keys)",
    )

    args = parser.parse_args()

    # Check required environment variables
    required_vars = ["OPENROUTER_API_KEY"]
    if not args.mock:
        required_vars.append("TAVILY_API_KEY")
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("\nSet them in your environment or create a .env file:")
        print("  export OPENROUTER_API_KEY=your-key")
        if not args.mock:
            print("  export TAVILY_API_KEY=your-key")
        print("\nOr use --mock for testing with mock data")
        sys.exit(1)

    # Run screening
    try:
        result = asyncio.run(run_screening(
            entity_name=args.entity,
            entity_type=args.type,
            max_iterations=args.max_iterations,
            verbose=not args.quiet,
            mock=args.mock,
        ))
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Output results
    if args.json:
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        print("\n" + "=" * 60)
        print(f"SCREENING COMPLETE: {args.entity}")
        print("=" * 60)

        if result.risk_assessment:
            print(f"\nRisk Level: {result.risk_assessment.risk_level.upper()}")
            print(f"Risk Score: {result.risk_assessment.risk_score}/100")

            if result.risk_assessment.key_concerns:
                print("\nKey Concerns:")
                for concern in result.risk_assessment.key_concerns:
                    print(f"  - {concern}")

            print(f"\nRecommendation: {result.risk_assessment.recommendation}")

        print(f"\nExecutive Summary:")
        print(result.executive_summary)

    # Save report if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write(result.detailed_report)
        print(f"\nDetailed report saved to: {args.output}")


if __name__ == "__main__":
    cli()
