PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

-- A tombstone is both a resync suppression rule and a compact recovery record.
-- The full paper and every directly related row are archived as JSON before the
-- transactional delete is issued.
CREATE TABLE IF NOT EXISTS paper_tombstones (
    id INTEGER PRIMARY KEY,
    doi TEXT UNIQUE,
    fingerprint TEXT NOT NULL UNIQUE,
    journal_id INTEGER,
    title TEXT NOT NULL,
    has_human_data INTEGER NOT NULL DEFAULT 0 CHECK (has_human_data IN (0,1)),
    archived_json TEXT NOT NULL,
    removed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_paper_tombstones_journal
ON paper_tombstones(journal_id, removed_at);

-- Verification is deliberately detached from the primary retrieval path. One
-- cached result per sync run is enough for this local, single-user product.
CREATE TABLE IF NOT EXISTS retrieval_verifications (
    id INTEGER PRIMARY KEY,
    sync_run_id INTEGER NOT NULL UNIQUE REFERENCES sync_runs(id) ON DELETE CASCADE,
    source_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'consistent', 'broadly_consistent', 'possible_gaps', 'unavailable'
    )),
    local_count INTEGER NOT NULL,
    independent_count INTEGER,
    comparable_independent_count INTEGER,
    exact_doi_matches INTEGER NOT NULL DEFAULT 0,
    possible_missing_count INTEGER NOT NULL DEFAULT 0,
    source_only_count INTEGER NOT NULL DEFAULT 0,
    local_only_count INTEGER NOT NULL DEFAULT 0,
    query_count INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    detail_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    verified_at TEXT NOT NULL
);

COMMIT;
