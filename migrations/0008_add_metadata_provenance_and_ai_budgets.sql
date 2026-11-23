PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS paper_metadata_provenance (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    field_name TEXT NOT NULL,
    active_value_hash TEXT,
    source_name TEXT NOT NULL,
    source_record_id TEXT,
    source_priority INTEGER NOT NULL,
    source_observed_at TEXT,
    fetched_at TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0 CHECK (is_current IN (0,1)),
    quality_flags_json TEXT NOT NULL DEFAULT '[]',
    decision TEXT NOT NULL CHECK (
        decision IN ('accepted','ignored','supplemented','conflict','degraded_prevention')
    ),
    replacement_reason TEXT,
    previous_provenance_id INTEGER REFERENCES paper_metadata_provenance(id),
    created_at TEXT NOT NULL,
    UNIQUE(paper_id,field_name,source_name,active_value_hash,decision)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_metadata_current_field
ON paper_metadata_provenance(paper_id,field_name) WHERE is_current=1;

CREATE INDEX IF NOT EXISTS idx_paper_metadata_provenance_source
ON paper_metadata_provenance(source_name,field_name);

CREATE TABLE IF NOT EXISTS ai_task_budgets (
    task_id INTEGER PRIMARY KEY REFERENCES workflow_batches(id) ON DELETE CASCADE,
    currency TEXT NOT NULL DEFAULT 'USD',
    price_version TEXT NOT NULL,
    max_calls INTEGER NOT NULL CHECK (max_calls > 0),
    max_input_tokens INTEGER NOT NULL CHECK (max_input_tokens > 0),
    max_output_tokens INTEGER NOT NULL CHECK (max_output_tokens > 0),
    max_estimated_cost_microunits INTEGER NOT NULL
        CHECK (max_estimated_cost_microunits > 0),
    max_strong_calls INTEGER NOT NULL CHECK (max_strong_calls > 0),
    max_translation_calls INTEGER NOT NULL CHECK (max_translation_calls > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_stage_budgets (
    task_id INTEGER NOT NULL REFERENCES workflow_batches(id) ON DELETE CASCADE,
    stage TEXT NOT NULL CHECK (stage IN (
        'title_translation','fast_semantic','strong_semantic','abstract_translation'
    )),
    max_calls INTEGER NOT NULL CHECK (max_calls > 0),
    max_input_tokens INTEGER NOT NULL CHECK (max_input_tokens > 0),
    max_output_tokens INTEGER NOT NULL CHECK (max_output_tokens > 0),
    max_estimated_cost_microunits INTEGER NOT NULL
        CHECK (max_estimated_cost_microunits > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(task_id,stage)
);

CREATE TABLE IF NOT EXISTS ai_usage_reservations (
    id INTEGER PRIMARY KEY,
    reservation_uid TEXT NOT NULL UNIQUE,
    task_id INTEGER NOT NULL REFERENCES workflow_batches(id) ON DELETE CASCADE,
    stage TEXT NOT NULL CHECK (stage IN (
        'title_translation','fast_semantic','strong_semantic','abstract_translation'
    )),
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    fencing_token INTEGER NOT NULL CHECK (fencing_token > 0),
    reserved_calls INTEGER NOT NULL DEFAULT 1 CHECK (reserved_calls > 0),
    reserved_input_tokens INTEGER NOT NULL CHECK (reserved_input_tokens > 0),
    reserved_output_tokens INTEGER NOT NULL CHECK (reserved_output_tokens > 0),
    reserved_cost_microunits INTEGER NOT NULL CHECK (reserved_cost_microunits > 0),
    actual_input_tokens INTEGER,
    actual_output_tokens INTEGER,
    actual_cost_microunits INTEGER,
    usage_missing INTEGER NOT NULL DEFAULT 0 CHECK (usage_missing IN (0,1)),
    status TEXT NOT NULL CHECK (
        status IN ('reserved','settled','failed','released')
    ),
    error_type TEXT,
    reserved_at TEXT NOT NULL,
    settled_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(task_id,stage,idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_ai_usage_task_status
ON ai_usage_reservations(task_id,status,stage);

CREATE TABLE IF NOT EXISTS ai_budget_changes (
    id INTEGER PRIMARY KEY,
    task_id INTEGER NOT NULL REFERENCES workflow_batches(id) ON DELETE CASCADE,
    changed_by TEXT NOT NULL,
    previous_budget_json TEXT NOT NULL,
    new_budget_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

COMMIT;
