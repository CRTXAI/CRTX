# Async Data Processing Pipeline

Build an async data processing pipeline with retry logic, dead letter queue, and monitoring.

## Features
- Multi-stage async pipeline (ingest → validate → transform → load)
- Configurable stage processors (plug in any async callable)
- Automatic retry with exponential backoff (configurable max retries)
- Dead letter queue for permanently failed items (with failure reason)
- Backpressure handling (bounded queues between stages)
- Real-time metrics (items processed, failed, in-flight, throughput)
- Graceful shutdown (drain in-flight items, flush DLQ)

## Technical Requirements
- Python 3.12+ with asyncio
- asyncio.Queue for inter-stage communication
- Pydantic models for pipeline items and configuration
- Structured logging with stage context
- No external dependencies beyond stdlib + Pydantic

## Pipeline Configuration
```python
pipeline = Pipeline(
    stages=[
        Stage("ingest", ingest_fn, concurrency=1),
        Stage("validate", validate_fn, concurrency=4),
        Stage("transform", transform_fn, concurrency=8, max_retries=3),
        Stage("load", load_fn, concurrency=2, max_retries=5),
    ],
    queue_size=1000,
    dead_letter_queue=True,
)

await pipeline.run(source=data_iterator)
metrics = pipeline.get_metrics()
```

## Metrics
```python
@dataclass
class PipelineMetrics:
    total_ingested: int
    total_completed: int
    total_failed: int
    items_in_flight: int
    throughput_per_second: float
    stage_metrics: dict[str, StageMetrics]
    dead_letter_count: int
    duration_seconds: float
```

## Tests
- Happy path (all items flow through all stages)
- Retry logic (transient failure → retry → success)
- Dead letter queue (permanent failure → DLQ with reason)
- Backpressure (slow consumer, queue fills, producer waits)
- Concurrency (multiple workers per stage process in parallel)
- Graceful shutdown (SIGINT → drain → clean exit)
- Metrics accuracy (counts match actual processing)
- Empty pipeline (zero items → clean completion)
- Stage failure isolation (one stage error doesn't crash pipeline)

```bash
triad run --task "$(cat examples/data_pipeline.md)"
```
