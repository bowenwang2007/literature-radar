PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

-- Gold-only additive migration.  This migration must never be applied to the
-- production or validation database families.
CREATE TABLE IF NOT EXISTS gold_corpora (
    id INTEGER PRIMARY KEY,
    corpus_version TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('building','frozen')),
    source_database_sha256 TEXT NOT NULL,
    source_manifest_json TEXT NOT NULL,
    sample_plan_json TEXT NOT NULL,
    sample_plan_sha256 TEXT NOT NULL,
    corpus_manifest_json TEXT NOT NULL,
    corpus_sha256 TEXT NOT NULL UNIQUE,
    paper_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    frozen_at TEXT
);

CREATE TABLE IF NOT EXISTS gold_corpus_papers (
    id INTEGER PRIMARY KEY,
    corpus_id INTEGER NOT NULL REFERENCES gold_corpora(id) ON DELETE RESTRICT,
    gold_paper_id TEXT NOT NULL,
    source_paper_id INTEGER NOT NULL,
    canonical_doi TEXT,
    fallback_fingerprint TEXT NOT NULL,
    original_title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    abstract TEXT,
    abstract_length INTEGER NOT NULL,
    abstract_quality_status TEXT NOT NULL,
    journal TEXT NOT NULL,
    journal_short_name TEXT NOT NULL,
    issn TEXT,
    publication_date TEXT,
    publication_year INTEGER,
    authors_json TEXT NOT NULL,
    metadata_provenance_summary_json TEXT NOT NULL,
    source_links_json TEXT NOT NULL,
    source_record_timestamp TEXT,
    sampling_stratum TEXT NOT NULL,
    stratum_population_size INTEGER NOT NULL,
    inclusion_probability REAL NOT NULL CHECK (
        inclusion_probability > 0 AND inclusion_probability <= 1
    ),
    duplicate_cluster_key TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(corpus_id,gold_paper_id),
    UNIQUE(corpus_id,source_paper_id)
);

CREATE TABLE IF NOT EXISTS gold_presented_records (
    id INTEGER PRIMARY KEY,
    corpus_id INTEGER NOT NULL REFERENCES gold_corpora(id) ON DELETE RESTRICT,
    corpus_paper_id INTEGER NOT NULL REFERENCES gold_corpus_papers(id) ON DELETE RESTRICT,
    reviewer_id TEXT NOT NULL,
    annotation_round INTEGER NOT NULL CHECK (annotation_round IN (1,2)),
    presented_record_id TEXT NOT NULL,
    presentation_order INTEGER NOT NULL,
    hidden_repeat INTEGER NOT NULL DEFAULT 0 CHECK (hidden_repeat IN (0,1)),
    repeat_of_presented_record_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(corpus_id,reviewer_id,annotation_round,presented_record_id),
    UNIQUE(corpus_id,reviewer_id,annotation_round,presentation_order)
);

CREATE TABLE IF NOT EXISTS gold_annotation_revisions (
    id INTEGER PRIMARY KEY,
    corpus_id INTEGER NOT NULL REFERENCES gold_corpora(id) ON DELETE RESTRICT,
    corpus_paper_id INTEGER NOT NULL REFERENCES gold_corpus_papers(id) ON DELETE RESTRICT,
    presented_record_id TEXT NOT NULL,
    annotation_round INTEGER NOT NULL CHECK (annotation_round IN (1,2)),
    reviewer_id TEXT NOT NULL,
    revision_number INTEGER NOT NULL,
    label TEXT CHECK (label IN ('include','borderline','exclude')),
    education_centrality TEXT CHECK (
        education_centrality IN ('core','substantial','incidental','none')
    ),
    economics_relevance TEXT CHECK (
        economics_relevance IN (
            'economic_question','economic_method_only',
            'adjacent_social_science','non_economic'
        )
    ),
    evidence_sufficiency TEXT CHECK (
        evidence_sufficiency IN ('sufficient','limited','insufficient')
    ),
    reviewer_confidence TEXT CHECK (reviewer_confidence IN ('high','medium','low')),
    reason_codes_json TEXT NOT NULL DEFAULT '[]',
    optional_notes TEXT,
    disposition TEXT NOT NULL CHECK (disposition IN ('submitted','deferred')),
    marked_for_review INTEGER NOT NULL DEFAULT 0 CHECK (marked_for_review IN (0,1)),
    source_revealed INTEGER NOT NULL DEFAULT 0 CHECK (source_revealed=0),
    reveal_timestamp TEXT CHECK (reveal_timestamp IS NULL),
    presentation_order INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    elapsed_seconds INTEGER NOT NULL CHECK (elapsed_seconds >= 0),
    protocol_version TEXT NOT NULL,
    protocol_sha256 TEXT NOT NULL,
    supersedes_annotation_id INTEGER REFERENCES gold_annotation_revisions(id),
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
    created_at TEXT NOT NULL,
    UNIQUE(presented_record_id,reviewer_id,annotation_round,revision_number)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_gold_annotation_one_active
ON gold_annotation_revisions(presented_record_id,reviewer_id,annotation_round)
WHERE active=1;

CREATE TABLE IF NOT EXISTS gold_source_reveals (
    id INTEGER PRIMARY KEY,
    annotation_id INTEGER NOT NULL REFERENCES gold_annotation_revisions(id) ON DELETE RESTRICT,
    reviewer_id TEXT NOT NULL,
    revealed_at TEXT NOT NULL,
    UNIQUE(annotation_id,reviewer_id)
);

CREATE TABLE IF NOT EXISTS gold_review_queue (
    id INTEGER PRIMARY KEY,
    corpus_id INTEGER NOT NULL REFERENCES gold_corpora(id) ON DELETE RESTRICT,
    corpus_paper_id INTEGER NOT NULL REFERENCES gold_corpus_papers(id) ON DELETE RESTRICT,
    queue_reason TEXT NOT NULL,
    source_annotation_id INTEGER REFERENCES gold_annotation_revisions(id),
    status TEXT NOT NULL CHECK (status IN ('open','resolved')),
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    UNIQUE(corpus_id,corpus_paper_id,queue_reason,status)
);

CREATE TABLE IF NOT EXISTS gold_arbitration_revisions (
    id INTEGER PRIMARY KEY,
    corpus_id INTEGER NOT NULL REFERENCES gold_corpora(id) ON DELETE RESTRICT,
    corpus_paper_id INTEGER NOT NULL REFERENCES gold_corpus_papers(id) ON DELETE RESTRICT,
    final_gold_label TEXT NOT NULL CHECK (
        final_gold_label IN ('include','borderline','exclude')
    ),
    basis TEXT NOT NULL,
    arbitrator_id TEXT NOT NULL,
    protocol_version TEXT NOT NULL,
    protocol_sha256 TEXT NOT NULL,
    supersedes_arbitration_id INTEGER REFERENCES gold_arbitration_revisions(id),
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_gold_arbitration_one_active
ON gold_arbitration_revisions(corpus_id,corpus_paper_id)
WHERE active=1;

CREATE TABLE IF NOT EXISTS gold_duplicate_clusters (
    id INTEGER PRIMARY KEY,
    corpus_id INTEGER NOT NULL REFERENCES gold_corpora(id) ON DELETE RESTRICT,
    cluster_uid TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('confirmed','candidate','not_duplicate')),
    basis_json TEXT NOT NULL,
    reviewed_by TEXT,
    reviewed_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(corpus_id,cluster_uid)
);

CREATE TABLE IF NOT EXISTS gold_duplicate_cluster_members (
    cluster_id INTEGER NOT NULL REFERENCES gold_duplicate_clusters(id) ON DELETE RESTRICT,
    corpus_paper_id INTEGER NOT NULL REFERENCES gold_corpus_papers(id) ON DELETE RESTRICT,
    PRIMARY KEY(cluster_id,corpus_paper_id)
);

CREATE TABLE IF NOT EXISTS gold_split_manifests (
    id INTEGER PRIMARY KEY,
    corpus_id INTEGER NOT NULL REFERENCES gold_corpora(id) ON DELETE RESTRICT,
    split_version TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    strategy_json TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL UNIQUE,
    frozen INTEGER NOT NULL DEFAULT 1 CHECK (frozen=1),
    created_at TEXT NOT NULL,
    UNIQUE(corpus_id,split_version)
);

CREATE TABLE IF NOT EXISTS gold_split_assignments (
    split_manifest_id INTEGER NOT NULL REFERENCES gold_split_manifests(id) ON DELETE RESTRICT,
    corpus_paper_id INTEGER NOT NULL REFERENCES gold_corpus_papers(id) ON DELETE RESTRICT,
    duplicate_cluster_uid TEXT NOT NULL,
    split_name TEXT NOT NULL CHECK (split_name IN ('development','calibration','holdout')),
    gold_label TEXT NOT NULL CHECK (gold_label IN ('include','borderline','exclude')),
    PRIMARY KEY(split_manifest_id,corpus_paper_id)
);

CREATE TABLE IF NOT EXISTS gold_governance_snapshots (
    id INTEGER PRIMARY KEY,
    corpus_id INTEGER NOT NULL REFERENCES gold_corpora(id) ON DELETE RESTRICT,
    snapshot_version TEXT NOT NULL,
    snapshot_path TEXT NOT NULL UNIQUE,
    snapshot_sha256 TEXT NOT NULL UNIQUE,
    corpus_sha256 TEXT NOT NULL,
    protocol_version TEXT NOT NULL,
    protocol_sha256 TEXT NOT NULL,
    split_manifest_sha256 TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'frozen' CHECK (status='frozen'),
    created_at TEXT NOT NULL,
    UNIQUE(corpus_id,snapshot_version)
);

CREATE TABLE IF NOT EXISTS gold_holdout_evaluations (
    id INTEGER PRIMARY KEY,
    snapshot_id INTEGER NOT NULL REFERENCES gold_governance_snapshots(id) ON DELETE RESTRICT,
    candidate_bundle_sha256 TEXT NOT NULL,
    split_manifest_sha256 TEXT NOT NULL,
    result_sha256 TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    UNIQUE(snapshot_id,candidate_bundle_sha256)
);

CREATE TABLE IF NOT EXISTS gold_pilot_runs_v2 (
    id INTEGER PRIMARY KEY,
    run_uid TEXT NOT NULL UNIQUE,
    snapshot_id INTEGER NOT NULL REFERENCES gold_governance_snapshots(id) ON DELETE RESTRICT,
    candidate_runtime_config_sha256 TEXT NOT NULL,
    split_name TEXT NOT NULL CHECK (split_name IN ('development','calibration')),
    budget_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('prepared','running','paused','completed','failed')
    ),
    label_blinded INTEGER NOT NULL DEFAULT 1 CHECK (label_blinded=1),
    randomized_seed INTEGER NOT NULL,
    stats_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gold_pilot_leases (
    id INTEGER PRIMARY KEY,
    pilot_run_id INTEGER NOT NULL REFERENCES gold_pilot_runs_v2(id) ON DELETE RESTRICT,
    owner_token TEXT NOT NULL,
    fencing_token INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active','released','stale')),
    acquired_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_gold_pilot_one_active_lease
ON gold_pilot_leases(pilot_run_id) WHERE status='active';

CREATE TABLE IF NOT EXISTS gold_pilot_results_v2 (
    id INTEGER PRIMARY KEY,
    pilot_run_id INTEGER NOT NULL REFERENCES gold_pilot_runs_v2(id) ON DELETE RESTRICT,
    corpus_paper_id INTEGER NOT NULL REFERENCES gold_corpus_papers(id) ON DELETE RESTRICT,
    paper_content_sha256 TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    model_provider TEXT,
    model_name TEXT,
    prompt_version TEXT,
    prompt_sha256 TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost_microunits INTEGER,
    latency_ms INTEGER,
    status TEXT NOT NULL CHECK (status IN ('succeeded','failed')),
    raw_response_reference TEXT,
    parsed_result_json TEXT,
    failure_type TEXT,
    no_abstract INTEGER NOT NULL CHECK (no_abstract IN (0,1)),
    limitations_json TEXT NOT NULL,
    label_blinded INTEGER NOT NULL DEFAULT 1 CHECK (label_blinded=1),
    created_at TEXT NOT NULL,
    UNIQUE(pilot_run_id,request_fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_gold_corpus_papers_stratum
ON gold_corpus_papers(corpus_id,sampling_stratum);
CREATE INDEX IF NOT EXISTS idx_gold_presented_progress
ON gold_presented_records(corpus_id,reviewer_id,annotation_round,presentation_order);
CREATE INDEX IF NOT EXISTS idx_gold_review_queue_open
ON gold_review_queue(corpus_id,status,queue_reason);
CREATE INDEX IF NOT EXISTS idx_gold_split_name
ON gold_split_assignments(split_manifest_id,split_name);

CREATE TRIGGER IF NOT EXISTS prevent_frozen_gold_corpus_update
BEFORE UPDATE ON gold_corpora WHEN OLD.status='frozen' BEGIN
    SELECT RAISE(ABORT,'frozen gold corpus is immutable');
END;
CREATE TRIGGER IF NOT EXISTS prevent_frozen_gold_corpus_delete
BEFORE DELETE ON gold_corpora WHEN OLD.status='frozen' BEGIN
    SELECT RAISE(ABORT,'frozen gold corpus is immutable');
END;
CREATE TRIGGER IF NOT EXISTS prevent_frozen_gold_corpus_paper_update
BEFORE UPDATE ON gold_corpus_papers
WHEN (SELECT status FROM gold_corpora WHERE id=OLD.corpus_id)='frozen' BEGIN
    SELECT RAISE(ABORT,'frozen gold corpus papers are immutable');
END;
CREATE TRIGGER IF NOT EXISTS prevent_frozen_gold_corpus_paper_delete
BEFORE DELETE ON gold_corpus_papers
WHEN (SELECT status FROM gold_corpora WHERE id=OLD.corpus_id)='frozen' BEGIN
    SELECT RAISE(ABORT,'frozen gold corpus papers are immutable');
END;
CREATE TRIGGER IF NOT EXISTS prevent_gold_governance_snapshot_update
BEFORE UPDATE ON gold_governance_snapshots BEGIN
    SELECT RAISE(ABORT,'gold governance snapshots are immutable');
END;
CREATE TRIGGER IF NOT EXISTS prevent_gold_governance_snapshot_delete
BEFORE DELETE ON gold_governance_snapshots BEGIN
    SELECT RAISE(ABORT,'gold governance snapshots are immutable');
END;
CREATE TRIGGER IF NOT EXISTS prevent_gold_split_update
BEFORE UPDATE ON gold_split_manifests BEGIN
    SELECT RAISE(ABORT,'gold split manifests are immutable');
END;
CREATE TRIGGER IF NOT EXISTS prevent_gold_split_delete
BEFORE DELETE ON gold_split_manifests BEGIN
    SELECT RAISE(ABORT,'gold split manifests are immutable');
END;

COMMIT;
