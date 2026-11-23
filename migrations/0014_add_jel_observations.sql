PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

-- JEL classification codes are a *paper fact*, not a profile opinion.
--
-- "Does this paper carry JEL code D91?" has one answer for the whole product,
-- exactly like its DOI or its abstract.  It is therefore stored on `papers`
-- (the pre-existing `jel_codes_json` column) and never duplicated per profile.
-- Every profile reads the same value; what differs per profile is only whether
-- that code matters for *its* research question.
--
-- This table is the observation ledger behind that single value.  One row per
-- (paper, provider, observed code set), so an auditor can answer three
-- questions that a bare JSON column cannot:
--
--   1. where did this code come from?      -> source_name / source_record_url
--   2. was it read or was it inferred?     -> extraction_method
--   3. did we ever look, and fail?         -> resolution ('resolved',
--                                              'absent_at_source', ...)
--
-- Row 3 matters most for honesty: "no JEL code is stored for this paper" and
-- "we asked the source and the source has no JEL code for this paper" are
-- different statements, and only the ledger can tell them apart.
CREATE TABLE IF NOT EXISTS paper_jel_observations (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    source_name TEXT NOT NULL,
    source_record_url TEXT,
    -- How the codes were obtained.  'structured_metadata' means the source
    -- exposed them as a machine-readable field; 'structured_page' means they
    -- were read verbatim from a source-rendered classification line.  Neither
    -- involves any model, inference, or title-based guesswork: a code that was
    -- not published by the source is never written here.
    extraction_method TEXT NOT NULL,
    -- 'resolved'    -> source published codes for this paper
    -- 'absent_at_source' -> source was reached and has no codes for this paper
    -- 'unresolved'  -> source could not be reached or the paper was not found
    resolution TEXT NOT NULL CHECK (resolution IN (
        'resolved', 'absent_at_source', 'unresolved'
    )),
    codes_json TEXT NOT NULL DEFAULT '[]',
    codes_normalized TEXT NOT NULL DEFAULT '',
    raw_evidence TEXT,
    http_status INTEGER,
    failure_reason TEXT,
    observed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jel_observations_paper
ON paper_jel_observations(paper_id, source_name);

CREATE INDEX IF NOT EXISTS idx_jel_observations_resolution
ON paper_jel_observations(resolution);

-- One row per (paper, source): the most recent answer from that source.
-- Re-running enrichment updates the answer instead of accumulating duplicates.
CREATE UNIQUE INDEX IF NOT EXISTS idx_jel_observations_paper_source
ON paper_jel_observations(paper_id, source_name);

COMMIT;
