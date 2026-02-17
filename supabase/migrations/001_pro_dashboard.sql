-- Triad Pro Dashboard — Supabase schema migration
-- Run this in the Supabase SQL editor or via `supabase db push`

-- ─── Users extension (Supabase Auth handles core users) ──────────

CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT,
  plan TEXT DEFAULT 'free' CHECK (plan IN ('free', 'pro', 'cloud')),
  view_preference TEXT DEFAULT 'dashboard',
  stripe_customer_id TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, email) VALUES (NEW.id, NEW.email);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ─── CLI API keys ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
  key_hash TEXT NOT NULL,
  key_prefix TEXT NOT NULL,
  label TEXT DEFAULT '',
  last_used_at TIMESTAMPTZ,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- ─── Pipeline sessions ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.sessions (
  id TEXT PRIMARY KEY,
  user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
  task TEXT NOT NULL,
  mode TEXT NOT NULL,
  route TEXT NOT NULL,
  arbiter_mode TEXT NOT NULL,
  status TEXT DEFAULT 'running',
  cost_total NUMERIC DEFAULT 0,
  tokens_total INTEGER DEFAULT 0,
  duration_ms INTEGER,
  confidence NUMERIC,
  model_assignments JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT now(),
  completed_at TIMESTAMPTZ
);

-- ─── Per-stage records ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.stages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT REFERENCES public.sessions(id) ON DELETE CASCADE,
  stage_name TEXT NOT NULL,
  model_id TEXT,
  status TEXT DEFAULT 'pending',
  tokens_in INTEGER DEFAULT 0,
  tokens_out INTEGER DEFAULT 0,
  cost NUMERIC DEFAULT 0,
  latency_ms INTEGER,
  output_text TEXT,
  verdict TEXT,
  confidence NUMERIC,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- ─── Raw pipeline events (source of truth) ──────────────────────

CREATE TABLE IF NOT EXISTS public.events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT REFERENCES public.sessions(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  payload JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT now()
);

-- ─── Arbiter verdicts ───────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.verdicts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT REFERENCES public.sessions(id) ON DELETE CASCADE,
  stage_name TEXT NOT NULL,
  verdict TEXT NOT NULL,
  confidence NUMERIC,
  issues JSONB DEFAULT '[]',
  reasoning TEXT,
  model_id TEXT,
  token_cost NUMERIC DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- ─── Row Level Security ─────────────────────────────────────────

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.stages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.verdicts ENABLE ROW LEVEL SECURITY;

-- User policies (authenticated users see own data)
CREATE POLICY "Users see own data" ON public.profiles FOR ALL USING (id = auth.uid());
CREATE POLICY "Users see own keys" ON public.api_keys FOR ALL USING (user_id = auth.uid());
CREATE POLICY "Users see own sessions" ON public.sessions FOR ALL USING (user_id = auth.uid());
CREATE POLICY "Users see own stages" ON public.stages FOR ALL USING (
  session_id IN (SELECT id FROM public.sessions WHERE user_id = auth.uid())
);
CREATE POLICY "Users see own events" ON public.events FOR ALL USING (
  session_id IN (SELECT id FROM public.sessions WHERE user_id = auth.uid())
);
CREATE POLICY "Users see own verdicts" ON public.verdicts FOR ALL USING (
  session_id IN (SELECT id FROM public.sessions WHERE user_id = auth.uid())
);

-- Service role bypass for API ingestion (service role key bypasses RLS by default,
-- but these explicit INSERT policies document the intent)
CREATE POLICY "Service can insert sessions" ON public.sessions FOR INSERT WITH CHECK (true);
CREATE POLICY "Service can update sessions" ON public.sessions FOR UPDATE USING (true);
CREATE POLICY "Service can insert stages" ON public.stages FOR INSERT WITH CHECK (true);
CREATE POLICY "Service can update stages" ON public.stages FOR UPDATE USING (true);
CREATE POLICY "Service can insert events" ON public.events FOR INSERT WITH CHECK (true);
CREATE POLICY "Service can insert verdicts" ON public.verdicts FOR INSERT WITH CHECK (true);

-- ─── Indexes ────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_sessions_user ON public.sessions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_stages_session ON public.stages(session_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON public.events(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_verdicts_session ON public.verdicts(session_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON public.api_keys(key_hash);

-- ─── Enable Realtime ────────────────────────────────────────────

ALTER PUBLICATION supabase_realtime ADD TABLE public.events;
ALTER PUBLICATION supabase_realtime ADD TABLE public.stages;
ALTER PUBLICATION supabase_realtime ADD TABLE public.sessions;
