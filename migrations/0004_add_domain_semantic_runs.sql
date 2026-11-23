PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS domain_semantic_runs (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    workflow_batch_id INTEGER,
    domain_evidence_run_id INTEGER NOT NULL REFERENCES domain_evidence_runs(id) ON DELETE CASCADE,
    profile_id TEXT NOT NULL,
    run_role TEXT NOT NULL CHECK (run_role IN ('fast','strong_review')),
    input_fingerprint TEXT NOT NULL,
    input_json TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    system_prompt_sha256 TEXT NOT NULL,
    user_prompt_sha256 TEXT NOT NULL,
    model_provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    escalation_policy_version TEXT NOT NULL,
    parent_fast_run_id INTEGER REFERENCES domain_semantic_runs(id),
    result_json TEXT,
    semantic_score REAL,
    decision TEXT CHECK (decision IN ('include','borderline','exclude')),
    confidence REAL,
    education_is_core INTEGER CHECK (education_is_core IN (0,1)),
    economics_mechanism_present INTEGER CHECK (economics_mechanism_present IN (0,1)),
    information_sufficiency TEXT,
    title_evidence_sufficiency TEXT,
    suspected_false_friend INTEGER CHECK (suspected_false_friend IN (0,1)),
    adjacent_domain TEXT,
    succeeded INTEGER NOT NULL CHECK (succeeded IN (0,1)),
    failure_type TEXT,
    failure_message TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost REAL,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_domain_semantic_success
ON domain_semantic_runs(input_fingerprint,run_role) WHERE succeeded=1;
CREATE INDEX IF NOT EXISTS idx_domain_semantic_paper
ON domain_semantic_runs(paper_id,run_role,created_at,id);

CREATE TABLE IF NOT EXISTS domain_composite_decisions (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    workflow_batch_id INTEGER,
    domain_evidence_run_id INTEGER NOT NULL REFERENCES domain_evidence_runs(id) ON DELETE CASCADE,
    fast_semantic_run_id INTEGER NOT NULL REFERENCES domain_semantic_runs(id),
    strong_semantic_run_id INTEGER REFERENCES domain_semantic_runs(id),
    input_fingerprint TEXT NOT NULL UNIQUE,
    escalation_json TEXT NOT NULL,
    composite_json TEXT NOT NULL,
    missed_priority_json TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    provisional_decision TEXT NOT NULL CHECK (provisional_decision IN (
        'include','borderline','exclude'
    )),
    uncertainty_level TEXT NOT NULL,
    conflict_types_json TEXT NOT NULL,
    calibration_status TEXT NOT NULL CHECK (calibration_status='uncalibrated'),
    publishable INTEGER NOT NULL DEFAULT 0 CHECK (publishable=0),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_domain_composite_paper
ON domain_composite_decisions(paper_id,created_at,id);

COMMIT;
