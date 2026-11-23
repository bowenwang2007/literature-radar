PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS workflow_batches (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL
        CHECK (source_type IN ('selected_papers', 'sync')),
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN (
            'draft', 'ready', 'running', 'paused',
            'ready_for_review', 'completed', 'failed'
        )),
    source_config_json TEXT NOT NULL DEFAULT '{}',
    screening_policy TEXT NOT NULL DEFAULT '{}',
    translation_policy TEXT NOT NULL DEFAULT '{}',
    sync_completed_at TEXT,
    screening_completed_at TEXT,
    translation_completed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    last_error TEXT,
    result_summary_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS workflow_batch_items (
    batch_id INTEGER NOT NULL REFERENCES workflow_batches(id) ON DELETE CASCADE,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (batch_id, paper_id)
);

CREATE INDEX IF NOT EXISTS idx_workflow_batches_status
    ON workflow_batches(status, updated_at);

CREATE INDEX IF NOT EXISTS idx_workflow_batch_items_paper
    ON workflow_batch_items(paper_id, batch_id);

COMMIT;
