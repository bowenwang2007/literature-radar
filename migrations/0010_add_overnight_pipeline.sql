PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

-- Overnight campaign output-first pipeline.
-- Tables are additive and idempotent. ALTER TABLE column additions for
-- sync_runs and paper_translations are applied by the guarded Python helper
-- in services/migrations.py (SQLite lacks conditional ADD COLUMN).

CREATE TABLE IF NOT EXISTS paper_screenings (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    task_id INTEGER REFERENCES workflow_batches(id) ON DELETE CASCADE,
    input_hash TEXT NOT NULL,
    model_decision TEXT NOT NULL,
    education_centrality TEXT,
    economic_relevance TEXT,
    evidence_sufficiency TEXT,
    confidence TEXT,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    reason TEXT,
    topic_tags_json TEXT NOT NULL DEFAULT '[]',
    review_priority TEXT NOT NULL DEFAULT 'medium',
    model_name TEXT,
    model_identity TEXT,
    prompt_version TEXT,
    prompt_hash TEXT,
    ontology_version TEXT,
    screened_at TEXT NOT NULL,
    raw_response_json TEXT NOT NULL DEFAULT '{}',
    parser_version TEXT NOT NULL,
    evidence_validation_version TEXT NOT NULL,
    validation_status TEXT NOT NULL DEFAULT 'valid'
        CHECK (validation_status IN ('valid','invalid','needs_recheck','recheck_failed','parse_failed')),
    validation_errors_json TEXT NOT NULL DEFAULT '[]',
    stale INTEGER NOT NULL DEFAULT 0 CHECK (stale IN (0,1)),
    review_decision TEXT,
    final_model_decision TEXT,
    disagreement_reason TEXT,
    validated_evidence_json TEXT NOT NULL DEFAULT '[]',
    review_reason TEXT,
    review_model_name TEXT,
    review_prompt_version TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_paper_screenings_paper_latest
ON paper_screenings(paper_id, stale, id DESC);

CREATE INDEX IF NOT EXISTS idx_paper_screenings_decision
ON paper_screenings(model_decision, stale);

CREATE TABLE IF NOT EXISTS provider_call_log (
    id INTEGER PRIMARY KEY,
    run_key TEXT NOT NULL,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    query_mode TEXT NOT NULL,
    query_identifier TEXT,
    status TEXT NOT NULL,
    abstract_text TEXT,
    abstract_language TEXT,
    matched_title TEXT,
    matched_doi TEXT,
    matched_authors_json TEXT NOT NULL DEFAULT '[]',
    matched_year INTEGER,
    confidence REAL,
    http_status INTEGER,
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0,1)),
    retry_after REAL,
    next_retry_at TEXT,
    error_category TEXT,
    error_message TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    cache_hit INTEGER NOT NULL DEFAULT 0 CHECK (cache_hit IN (0,1)),
    fused_skip INTEGER NOT NULL DEFAULT 0 CHECK (fused_skip IN (0,1)),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_provider_call_log_run
ON provider_call_log(run_key, provider, paper_id);

-- Phase 7B5: per-paper processing status keyed by sync_run (product path no
-- longer depends on workflow_batches / personal task state).
CREATE TABLE IF NOT EXISTS paper_processing_status (
    id INTEGER PRIMARY KEY,
    sync_run_id INTEGER NOT NULL REFERENCES sync_runs(id) ON DELETE CASCADE,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN (
            'pending', 'processing', 'abstract_ready', 'needs_retry',
            'needs_manual_abstract', 'classification_ready', 'classified', 'failed'
        )),
    stage TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    classification_json TEXT NOT NULL DEFAULT '{}',
    model_name TEXT,
    prompt_version TEXT,
    classified_at TEXT,
    title_translation_json TEXT NOT NULL DEFAULT '{}',
    abstract_retrieval_json TEXT NOT NULL DEFAULT '{}',
    human_decision TEXT NOT NULL DEFAULT 'undecided'
        CHECK (human_decision IN ('undecided', 'related', 'irrelevant', 'uncertain')),
    human_note TEXT,
    human_decided_at TEXT,
    error_type TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(sync_run_id, paper_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_processing_status_run
ON paper_processing_status(sync_run_id, status, paper_id);

CREATE INDEX IF NOT EXISTS idx_paper_processing_status_paper
ON paper_processing_status(paper_id, sync_run_id);

COMMIT;
