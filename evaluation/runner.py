"""Command-line runner for batch quality evaluation of the platform."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Dict

from app.api.v1.schemas import ResearchRequest
from app.application.research.service import ResearchService
from app.core.config import get_settings
from app.services.llm_service import LLMService
from app.services.memory_service import ResearchMemoryService
from evaluation.judges import LLMAccuracyJudge
from evaluation.metrics import (
    calculate_citation_coverage,
    calculate_composite_score,
    calculate_source_diversity,
)
from evaluation.test_cases import TEST_CASES

# Configure logging for evaluation runs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluation-runner")


async def run_evaluation_suite() -> None:
    """Runs all test cases through the platform and evaluates quality outputs."""
    settings = get_settings()

    # Ensure Tavily key is configured
    if not settings.tavily_api_key:
        print("❌ Error: TAVILY_API_KEY environment variable is required to run evaluation.")
        sys.exit(1)

    print("=" * 70)
    print("🧠 AI INDUSTRY INTELLIGENCE PLATFORM — QUALITY EVALUATION SUITE")
    print(f"Running {len(TEST_CASES)} golden test cases on local agent swarm...")
    print(f"Base URL: {settings.ollama_base_url} | Primary Model: {settings.primary_model}")
    print("=" * 70)

    # Initialize services
    llm_service = LLMService(settings)
    memory_service = ResearchMemoryService(settings)
    research_service = ResearchService(
        settings=settings,
        llm_service=llm_service,
        memory_service=memory_service,
    )
    judge = LLMAccuracyJudge(llm_service)

    # Clean FAISS index before evaluation to prevent cross-contamination
    print("\n🧹 Initializing clean memory store index...")
    memory_service.clear()

    summary_results = []

    for case in TEST_CASES:
        print(f"\n🚀 Running {case.id}: \"{case.query}\"")
        request = ResearchRequest(
            query=case.query,
            business_context=case.business_context,
            max_iterations=2,
        )

        try:
            # Create job record
            record = await research_service.create_research(request)
            print(f"  - Created research job {record.id}")

            # Execute LangGraph agents
            print("  - Launching agent swarm. This may take 1-3 minutes...")
            await research_service.run_research(record.id)

            # Get final state
            final_record = await research_service.get_research(record.id)
            if final_record.error:
                print(f"  ❌ Job failed with error: {final_record.error}")
                summary_results.append({"id": case.id, "query": case.query, "status": "FAILED", "error": final_record.error})
                continue

            # Load report pydantic values as dict
            report_dict = final_record.report.model_dump() if final_record.report else {}

            # Retrieve evidence and sources from raw state
            # Convert evidence elements to list of dict
            sources = [s.model_dump() for s in final_record.raw_state.get("sources", [])] if final_record.raw_state else []

            # 1. Compute metrics
            diversity = calculate_source_diversity(sources)
            coverage = calculate_citation_coverage(
                report_dict.get("citations", []),
                report_dict.get("sections", {}),
            )
            confidence = report_dict.get("confidence_score", 0.0)

            # 2. Run LLM judge
            print("  - Calling LLM Accuracy Judge to audit report grounding...")
            judge_res = await judge.evaluate_report(report_dict)
            accuracy = judge_res.accuracy_score

            # 3. Calculate composite
            composite = calculate_composite_score(diversity, coverage, confidence, accuracy)

            print("=" * 50)
            print(f"📊 Quality Metrics for {case.id}:")
            print(f"  - Source Diversity:   {diversity * 100:.1f}%")
            print(f"  - Citation Coverage:  {coverage * 100:.1f}%")
            print(f"  - Swarm Confidence:   {confidence * 100:.1f}%")
            print(f"  - Judge Accuracy:     {accuracy * 100:.1f}%")
            print(f"  - COMPOSITE SCORE:    {composite * 100:.1f}%")
            print("-" * 50)
            print(f"💬 Judge Rationale:\n{judge_res.rationale}")
            if judge_res.unsupported_claims:
                print("⚠️  Unsupported Claims Found:")
                for claim in judge_res.unsupported_claims:
                    print(f"    * {claim}")
            print("=" * 50)

            summary_results.append(
                {
                    "id": case.id,
                    "query": case.query,
                    "status": "SUCCESS",
                    "metrics": {
                        "diversity": diversity,
                        "coverage": coverage,
                        "confidence": confidence,
                        "accuracy": accuracy,
                        "composite": composite,
                    },
                }
            )

        except Exception as exc:
            logger.exception("Failed to run evaluation case %s", case.id)
            print(f"  ❌ System Error: {exc}")
            summary_results.append({"id": case.id, "query": case.query, "status": "ERROR", "error": str(exc)})

    # Final summary output
    print("\n" + "=" * 70)
    print("🏁 EVALUATION SUITE SUMMARY REPORT")
    print("=" * 70)
    for res in summary_results:
        status_str = res["status"]
        if status_str == "SUCCESS":
            m = res["metrics"]
            print(f"✅ {res['id']}: Composite={m['composite'] * 100:.1f}% | Accuracy={m['accuracy'] * 100:.1f}% | Diversity={m['diversity'] * 100:.1f}%")
        else:
            print(f"❌ {res['id']}: Status={status_str} | Error={res.get('error')}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())
