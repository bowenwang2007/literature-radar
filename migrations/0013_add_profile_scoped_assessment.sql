PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

-- Profile-aware assessment model.
--
-- Paper facts (DOI, title, authors, journal, dates, abstract, translation,
-- provider provenance) are global and stored once in `papers` /
-- `paper_translations` / `paper_metadata_provenance`.
--
-- An assessment, however, is only meaningful relative to a research profile:
-- the same paper can be relevant to Education Economics and irrelevant to
-- Labor Economics.  These two tables make the profile the owner of the
-- judgement, while `paper_processing_status` keeps recording *what happened in
-- one retrieval run* (status, stage, attempts, errors) as an audit snapshot.
--
-- The legacy run-scoped columns (classification_json, human_decision, ...) are
-- retained for historical rows; the product path reads through the profile
-- tables and no longer treats a retrieval run as the owner of a long-term
-- judgement.
CREATE TABLE IF NOT EXISTS paper_profile_assessments (
    id INTEGER PRIMARY KEY,
    profile_id TEXT NOT NULL,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    decision TEXT,
    screening_basis TEXT,
    classification_json TEXT NOT NULL DEFAULT '{}',
    model_name TEXT,
    prompt_version TEXT,
    input_fingerprint TEXT,
    source_run_id INTEGER REFERENCES sync_runs(id) ON DELETE SET NULL,
    assessed_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(profile_id, paper_id)
);

CREATE INDEX IF NOT EXISTS idx_profile_assessments_profile
ON paper_profile_assessments(profile_id, decision);

CREATE INDEX IF NOT EXISTS idx_profile_assessments_paper
ON paper_profile_assessments(paper_id);

CREATE TABLE IF NOT EXISTS paper_profile_reviews (
    id INTEGER PRIMARY KEY,
    profile_id TEXT NOT NULL,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    human_decision TEXT NOT NULL DEFAULT 'undecided'
        CHECK (human_decision IN ('undecided', 'related', 'irrelevant', 'uncertain')),
    human_note TEXT,
    decided_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(profile_id, paper_id)
);

CREATE INDEX IF NOT EXISTS idx_profile_reviews_profile
ON paper_profile_reviews(profile_id, human_decision);

-- `sync_runs.profile_id` records which profile asked for a retrieval run.  A
-- retrieval run is a discovery snapshot, not a judgement container.  SQLite
-- cannot add a column conditionally, so the guarded Python helper in
-- services/migrations.py adds it and backfills the default profile.

COMMIT;
