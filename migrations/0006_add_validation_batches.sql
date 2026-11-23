PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS validation_batches (
    id INTEGER PRIMARY KEY,
    source_database_path TEXT NOT NULL,
    source_database_sha256 TEXT NOT NULL,
    source_journals_json TEXT NOT NULL,
    date_from TEXT,
    date_to TEXT,
    total_papers INTEGER NOT NULL,
    ontology_version TEXT NOT NULL,
    semantic_prompt_version TEXT NOT NULL,
    composite_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('annotation','completed')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS validation_items (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL REFERENCES validation_batches(id) ON DELETE CASCADE,
    source_paper_id INTEGER NOT NULL,
    doi TEXT,
    original_title TEXT NOT NULL,
    chinese_title TEXT,
    authors_json TEXT NOT NULL DEFAULT '[]',
    original_abstract TEXT,
    journal TEXT NOT NULL,
    publication_date TEXT,
    ai_decision TEXT NOT NULL CHECK (ai_decision IN ('include','borderline','exclude')),
    ai_reason TEXT,
    semantic_score REAL,
    confidence REAL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    semantic_json TEXT NOT NULL DEFAULT '{}',
    sampled_stratum TEXT NOT NULL CHECK (
        sampled_stratum IN ('include','borderline','exclude')
    ),
    created_at TEXT NOT NULL,
    UNIQUE(batch_id, source_paper_id)
);

CREATE TABLE IF NOT EXISTS validation_annotations (
    id INTEGER PRIMARY KEY,
    validation_item_id INTEGER NOT NULL UNIQUE
        REFERENCES validation_items(id) ON DELETE CASCADE,
    label TEXT NOT NULL CHECK (label IN ('include','borderline','exclude')),
    reviewer TEXT NOT NULL,
    notes TEXT,
    difficulty TEXT NOT NULL CHECK (difficulty IN ('easy','medium','hard')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_validation_batches_status
ON validation_batches(status, created_at);

CREATE INDEX IF NOT EXISTS idx_validation_items_batch_stratum
ON validation_items(batch_id, sampled_stratum, id);

CREATE INDEX IF NOT EXISTS idx_validation_annotations_item
ON validation_annotations(validation_item_id);

COMMIT;
