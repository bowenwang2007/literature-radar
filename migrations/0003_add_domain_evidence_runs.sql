PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS domain_evidence_runs (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    workflow_batch_id INTEGER,
    profile_id TEXT NOT NULL,
    ontology_version TEXT NOT NULL,
    subject_code_version TEXT NOT NULL,
    journal_prior_version TEXT NOT NULL,
    scoring_version TEXT NOT NULL,
    config_bundle_sha256 TEXT NOT NULL,
    application_version TEXT NOT NULL,
    ontology_evidence_score REAL,
    subject_code_evidence_score REAL,
    journal_prior_evidence_score REAL,
    exclusion_evidence_penalty REAL,
    concept_diversity_score REAL,
    metadata_confidence REAL NOT NULL,
    provisional_rule_score REAL,
    rule_evidence_tier TEXT NOT NULL CHECK (rule_evidence_tier IN (
        'strong_positive','moderate_positive','weak_or_mixed','little_evidence'
    )),
    opaque_title INTEGER NOT NULL CHECK (opaque_title IN (0,1)),
    opacity_level TEXT NOT NULL CHECK (opacity_level IN ('low','medium','high')),
    review_priority_tier TEXT NOT NULL CHECK (review_priority_tier IN (
        'urgent','high','normal','low'
    )),
    review_priority_score REAL,
    evidence_json TEXT NOT NULL,
    publishable INTEGER NOT NULL DEFAULT 0 CHECK (publishable IN (0,1)),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_domain_evidence_identity
ON domain_evidence_runs(
    paper_id,workflow_batch_id,config_bundle_sha256,scoring_version,created_at
);

CREATE INDEX IF NOT EXISTS idx_domain_evidence_priority
ON domain_evidence_runs(review_priority_tier,review_priority_score,created_at);

COMMIT;
