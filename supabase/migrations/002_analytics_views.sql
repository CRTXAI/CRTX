-- Triad Pro — Analytics views for model performance and cost tracking.
-- Run after 001_pro_dashboard.sql.

-- ─── Model performance summary ─────────────────────────────────────
-- Aggregates per model: stage counts, verdict distribution, latency, cost.

CREATE OR REPLACE VIEW public.model_performance AS
SELECT
  s.model_id,
  p.id AS user_id,
  COUNT(*) AS total_stages,
  COUNT(*) FILTER (WHERE v.verdict = 'APPROVE') AS approved_count,
  COUNT(*) FILTER (WHERE v.verdict = 'FLAG') AS flagged_count,
  COUNT(*) FILTER (WHERE v.verdict = 'REJECT') AS rejected_count,
  COALESCE(AVG(s.latency_ms), 0) AS avg_latency_ms,
  COALESCE(AVG(v.confidence), 0) AS avg_confidence,
  COALESCE(SUM(s.cost), 0) AS total_cost
FROM public.stages s
JOIN public.sessions sess ON sess.id = s.session_id
JOIN public.profiles p ON p.id = sess.user_id
LEFT JOIN public.verdicts v ON v.session_id = s.session_id AND v.stage_name = s.stage_name
WHERE s.status = 'completed' AND s.model_id IS NOT NULL
GROUP BY s.model_id, p.id;

-- ─── Daily cost rollup ─────────────────────────────────────────────
-- Per-day, per-model, per-stage cost and token totals.

CREATE OR REPLACE VIEW public.cost_rollup AS
SELECT
  DATE(s.created_at) AS day,
  s.model_id,
  s.stage_name,
  sess.user_id,
  COALESCE(SUM(s.cost), 0) AS total_cost,
  COALESCE(SUM(s.tokens_in + s.tokens_out), 0) AS total_tokens,
  COUNT(*) AS run_count
FROM public.stages s
JOIN public.sessions sess ON sess.id = s.session_id
WHERE s.status = 'completed' AND s.model_id IS NOT NULL
GROUP BY DATE(s.created_at), s.model_id, s.stage_name, sess.user_id;

-- ─── Indexes to support analytics queries ──────────────────────────

CREATE INDEX IF NOT EXISTS idx_stages_model ON public.stages(model_id) WHERE status = 'completed';
CREATE INDEX IF NOT EXISTS idx_stages_created ON public.stages(created_at);
CREATE INDEX IF NOT EXISTS idx_verdicts_verdict ON public.verdicts(verdict);
