PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

-- Phase 7C4: a minimal provenance field on the existing tombstone table.
-- When a previously removed paper is deliberately restored through the
-- retrieval-verification intake flow, the tombstone keeps its removal archive
-- and history while `restored_at` stops it from acting as a resync
-- suppression rule. SQLite cannot conditionally add a column, so the guarded
-- Python helper in services/migrations.py applies this change only when the
-- column is absent. This file is the versioned record of that schema change.
ALTER TABLE paper_tombstones ADD COLUMN restored_at TEXT;

COMMIT;
