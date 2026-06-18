# AI Business Research Analyst Workflow

## ASCII Graph

```text
START
  |
  v
initialize
  |
  v
mark_planning -> planner
  |
  v
mark_researching -> researcher <--------------------+
  |                                                 |
  v                                                 |
mark_analyzing -> analyst                           |
  |                                                 |
  v                                                 |
mark_critiquing -> critic                           |
  |                                                 |
  v                                                 |
route_after_critic                                  |
  |                                                 |
  +-- pass ------------------> mark_writing         |
  |                           |                     |
  |                           v                     |
  |                         writer                  |
  |                           |                     |
  |                           v                     |
  |                         mark_completed -> END   |
  |                                                 |
  +-- retry_research --------> prepare_research_retry+
  |      while retry_count < max_iterations
  |
  +-- fail ------------------> fail -> END
```

## Retry Contract

The Critic controls research retries by setting `critic_review.decision`:

- `pass`: continue to Writer.
- `retry_research`: route back to Researcher while `retry_count < max_iterations`.
- `write_with_limitations`: continue to Writer after the retry budget is exhausted or when the Critic decides a caveated report is acceptable.
- `fail`: terminate without writing.

The default `max_iterations` is `3`, meaning three total research-analysis-critic passes: the initial pass plus up to two retries. The retry node increments both `iteration_count` and `retry_count` before returning to the Researcher.

## Agent Boundary

The workflow defines orchestration only. Planner, Researcher, Analyst, Critic, and Writer are injected as callables via `BusinessResearchAgents`.

Each agent should return a partial `ResearchState` update. Agent implementations belong outside this module.
