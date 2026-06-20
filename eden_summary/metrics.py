from prometheus_client import Counter, Histogram

job_stage_seconds = Histogram(
    'eden_job_stage_seconds',
    'Processing time per job stage',
    ['stage'],
    buckets=[1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600],
)

asr_processing_ratio = Histogram(
    'eden_asr_processing_ratio',
    'ASR processing time divided by audio duration',
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

quality_guard_seconds = Histogram(
    'eden_quality_guard_seconds',
    'Tier 1 inline quality-guard execution time (sub-second budget)',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

job_terminal_total = Counter(
    'eden_job_terminal_total',
    'Jobs by terminal status',
    ['status'],
)

# §6.6 Regeneration outcome: 'applied' (repaired kept), 'reverted' (repaired worse,
# original kept), 'skipped' (trigger did not fire). Lets us observe how often regen
# fires and how often keep-if-better actually keeps the repaired version.
summq_regen_total = Counter(
    'eden_summq_regen_total',
    'SummQ regeneration outcomes',
    ['outcome'],
)