PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS gold_sets (
    id INTEGER PRIMARY KEY,
    gold_set_uid TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    source_db_path TEXT NOT NULL,
    source_db_sha256 TEXT NOT NULL,
    sampling_manifest_json TEXT NOT NULL,
    annotation_version TEXT NOT NULL,
    review_mode TEXT NOT NULL CHECK (review_mode IN ('dual_reviewer','single_reviewer_two_rounds')),
    minimum_round_interval_days INTEGER,
    status TEXT NOT NULL CHECK (status IN (
        'sampling','annotating_round_1','annotating_round_2','adjudicating','frozen'
    )),
    created_at TEXT NOT NULL,
    frozen_at TEXT,
    snapshot_sha256 TEXT
);

CREATE TABLE IF NOT EXISTS gold_set_items (
    id INTEGER PRIMARY KEY,
    gold_set_id INTEGER NOT NULL REFERENCES gold_sets(id) ON DELETE CASCADE,
    source_paper_id INTEGER,
    doi TEXT,
    original_title TEXT NOT NULL,
    formal_chinese_title TEXT,
    original_abstract TEXT,
    authors_json TEXT NOT NULL DEFAULT '[]',
    journal TEXT NOT NULL,
    journal_group TEXT NOT NULL,
    publication_date TEXT,
    keywords_json TEXT NOT NULL DEFAULT '[]',
    subject_codes_json TEXT NOT NULL DEFAULT '[]',
    source_metadata_json TEXT NOT NULL DEFAULT '{}',
    stratum TEXT NOT NULL,
    difficulty_tags_json TEXT NOT NULL DEFAULT '[]',
    display_order_seed INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(gold_set_id,source_paper_id),
    UNIQUE(gold_set_id,doi)
);

CREATE TABLE IF NOT EXISTS gold_set_annotations (
    id INTEGER PRIMARY KEY,
    gold_set_item_id INTEGER NOT NULL REFERENCES gold_set_items(id) ON DELETE CASCADE,
    reviewer_id TEXT NOT NULL,
    review_round INTEGER NOT NULL CHECK (review_round IN (1,2)),
    label TEXT NOT NULL CHECK (label IN ('include','borderline','exclude','uncertain')),
    reviewer_note TEXT,
    chinese_title_shown INTEGER NOT NULL DEFAULT 0 CHECK (chinese_title_shown IN (0,1)),
    ai_info_shown INTEGER NOT NULL DEFAULT 0 CHECK (ai_info_shown=0),
    marked_for_later INTEGER NOT NULL DEFAULT 0 CHECK (marked_for_later IN (0,1)),
    started_at TEXT,
    completed_at TEXT NOT NULL,
    UNIQUE(gold_set_item_id,reviewer_id,review_round)
);

CREATE TABLE IF NOT EXISTS gold_set_adjudications (
    id INTEGER PRIMARY KEY,
    gold_set_item_id INTEGER NOT NULL UNIQUE REFERENCES gold_set_items(id) ON DELETE CASCADE,
    adjudicated_label TEXT NOT NULL CHECK (
        adjudicated_label IN ('include','borderline','exclude','uncertain')
    ),
    adjudication_reason TEXT NOT NULL,
    adjudicator_id TEXT NOT NULL,
    adjudicated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gold_set_deferrals (
    id INTEGER PRIMARY KEY,
    gold_set_item_id INTEGER NOT NULL REFERENCES gold_set_items(id) ON DELETE CASCADE,
    reviewer_id TEXT NOT NULL,
    review_round INTEGER NOT NULL CHECK (review_round IN (1,2)),
    deferred_at TEXT NOT NULL,
    UNIQUE(gold_set_item_id,reviewer_id,review_round)
);

CREATE TABLE IF NOT EXISTS gold_set_snapshots (
    id INTEGER PRIMARY KEY,
    gold_set_id INTEGER NOT NULL REFERENCES gold_sets(id) ON DELETE RESTRICT,
    snapshot_version TEXT NOT NULL,
    annotation_version TEXT NOT NULL,
    snapshot_path TEXT NOT NULL UNIQUE,
    snapshot_sha256 TEXT NOT NULL UNIQUE,
    item_count INTEGER NOT NULL,
    label_counts_json TEXT NOT NULL,
    source_db_sha256 TEXT NOT NULL,
    immutable INTEGER NOT NULL DEFAULT 1 CHECK (immutable=1),
    created_at TEXT NOT NULL,
    UNIQUE(gold_set_id,snapshot_version)
);

CREATE TABLE IF NOT EXISTS source_verifications (
    id INTEGER PRIMARY KEY,
    profile_id TEXT NOT NULL,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('ontology_concept','subject_code','journal')),
    entity_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_title TEXT NOT NULL,
    publisher_or_organization TEXT,
    source_url TEXT,
    accessed_at TEXT NOT NULL,
    verification_status TEXT NOT NULL CHECK (
        verification_status IN ('pending','verified','rejected','needs_revision')
    ),
    reviewer_id TEXT NOT NULL,
    reviewer_note TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(profile_id,entity_type,entity_id,source_type,source_title)
);

CREATE TABLE IF NOT EXISTS pilot_runs (
    id INTEGER PRIMARY KEY,
    run_uid TEXT NOT NULL UNIQUE,
    gold_set_snapshot_id INTEGER NOT NULL REFERENCES gold_set_snapshots(id),
    model_bundle_version TEXT NOT NULL,
    config_bundle_sha256 TEXT NOT NULL,
    prompt_bundle_sha256 TEXT NOT NULL,
    budget_json TEXT NOT NULL,
    stats_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending','running','partial','completed','failed')),
    failure_summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pilot_results (
    id INTEGER PRIMARY KEY,
    pilot_run_id INTEGER NOT NULL REFERENCES pilot_runs(id) ON DELETE CASCADE,
    snapshot_item_id TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    result_json TEXT,
    succeeded INTEGER NOT NULL CHECK (succeeded IN (0,1)),
    failure_type TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost REAL,
    created_at TEXT NOT NULL,
    UNIQUE(pilot_run_id,request_fingerprint)
);

CREATE TABLE IF NOT EXISTS calibration_runs (
    id INTEGER PRIMARY KEY,
    gold_set_snapshot_id INTEGER NOT NULL REFERENCES gold_set_snapshots(id),
    config_bundle_sha256 TEXT NOT NULL,
    model_bundle_version TEXT NOT NULL,
    prompt_bundle_sha256 TEXT NOT NULL,
    run_type TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    split_manifest_json TEXT NOT NULL,
    attempted_policies_json TEXT NOT NULL,
    selected_policy_json TEXT,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_bundle_candidates (
    id INTEGER PRIMARY KEY,
    candidate_version TEXT NOT NULL UNIQUE,
    gold_set_snapshot_id INTEGER NOT NULL REFERENCES gold_set_snapshots(id),
    bundle_json TEXT NOT NULL,
    bundle_sha256 TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'candidate' CHECK (status='candidate'),
    created_at TEXT NOT NULL,
    approved_at TEXT CHECK (approved_at IS NULL)
);

CREATE INDEX IF NOT EXISTS idx_gold_annotations_progress
ON gold_set_annotations(gold_set_item_id,review_round,reviewer_id);
CREATE INDEX IF NOT EXISTS idx_gold_deferrals_progress
ON gold_set_deferrals(gold_set_item_id,review_round,reviewer_id);
CREATE INDEX IF NOT EXISTS idx_gold_items_stratum
ON gold_set_items(gold_set_id,stratum);
CREATE INDEX IF NOT EXISTS idx_pilot_results_run
ON pilot_results(pilot_run_id,succeeded);

CREATE TRIGGER IF NOT EXISTS prevent_gold_snapshot_update
BEFORE UPDATE ON gold_set_snapshots BEGIN
    SELECT RAISE(ABORT,'gold set snapshots are immutable');
END;
CREATE TRIGGER IF NOT EXISTS prevent_gold_snapshot_delete
BEFORE DELETE ON gold_set_snapshots BEGIN
    SELECT RAISE(ABORT,'gold set snapshots are immutable');
END;

COMMIT;
