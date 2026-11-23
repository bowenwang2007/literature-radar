PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS workflow_task_leases (
    id INTEGER PRIMARY KEY,
    task_id INTEGER NOT NULL REFERENCES workflow_batches(id) ON DELETE CASCADE,
    owner_id TEXT NOT NULL CHECK (length(trim(owner_id)) > 0),
    acquired_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    released_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('active','released','stale')),
    current_stage TEXT NOT NULL,
    fencing_token INTEGER NOT NULL CHECK (fencing_token > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(task_id, fencing_token)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_workflow_task_active_lease
ON workflow_task_leases(task_id) WHERE status='active';

CREATE INDEX IF NOT EXISTS idx_workflow_task_leases_task_history
ON workflow_task_leases(task_id, fencing_token DESC);

CREATE INDEX IF NOT EXISTS idx_workflow_task_leases_expiry
ON workflow_task_leases(status, expires_at);

CREATE TABLE IF NOT EXISTS workflow_task_reviews (
    id INTEGER PRIMARY KEY,
    task_id INTEGER NOT NULL REFERENCES workflow_batches(id) ON DELETE CASCADE,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    review_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending','approved','rejected','needs_revision')),
    reincluded INTEGER NOT NULL DEFAULT 0 CHECK (reincluded IN (0,1)),
    recovery_status TEXT NOT NULL DEFAULT 'not_applicable'
        CHECK (recovery_status IN ('not_applicable','recovered','not_recovered')),
    final_decision TEXT CHECK (final_decision IN ('include','exclude')),
    reviewer TEXT,
    reviewer_notes TEXT,
    title_override TEXT,
    abstract_override TEXT,
    needs_edit INTEGER NOT NULL DEFAULT 0 CHECK (needs_edit IN (0,1)),
    source TEXT NOT NULL DEFAULT 'v05_human'
        CHECK (source IN ('v05_human','legacy_backfill','human_recovery')),
    provenance_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    reviewed_at TEXT,
    UNIQUE(task_id, paper_id)
);

CREATE INDEX IF NOT EXISTS idx_workflow_task_reviews_task_status
ON workflow_task_reviews(task_id, review_status, paper_id);

CREATE INDEX IF NOT EXISTS idx_workflow_task_reviews_paper
ON workflow_task_reviews(paper_id, task_id);

CREATE TABLE IF NOT EXISTS workflow_task_review_migration_conflicts (
    id INTEGER PRIMARY KEY,
    legacy_review_id INTEGER NOT NULL REFERENCES reviews(id) ON DELETE CASCADE,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    conflict_type TEXT NOT NULL CHECK (conflict_type IN ('ambiguous_multiple_tasks')),
    task_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(legacy_review_id, conflict_type)
);

-- A legacy review is copied only when the paper belongs to exactly one workflow task.
-- Ambiguous records remain exclusively in the legacy table and are reported below.
INSERT OR IGNORE INTO workflow_task_reviews (
    task_id,paper_id,review_status,reincluded,recovery_status,final_decision,
    reviewer,reviewer_notes,title_override,abstract_override,needs_edit,
    source,provenance_json,created_at,updated_at,reviewed_at
)
SELECT
    MIN(wbi.batch_id),r.paper_id,r.status,0,'not_applicable',
    CASE WHEN r.status='approved' THEN 'include'
         WHEN r.status='rejected' THEN 'exclude'
         ELSE NULL END,
    NULL,r.reviewer_note,r.edited_chinese_title,r.edited_chinese_summary,
    CASE WHEN r.status='needs_revision' THEN 1 ELSE 0 END,
    'legacy_backfill',
    json_object('legacy_review_id',r.id,'backfill_rule','single_task_only'),
    COALESCE(r.reviewed_at,wb.created_at),
    COALESCE(r.reviewed_at,wb.updated_at),
    r.reviewed_at
FROM reviews r
JOIN workflow_batch_items wbi ON wbi.paper_id=r.paper_id
JOIN workflow_batches wb ON wb.id=wbi.batch_id
GROUP BY r.id,r.paper_id
HAVING COUNT(DISTINCT wbi.batch_id)=1;

INSERT OR IGNORE INTO workflow_task_review_migration_conflicts (
    legacy_review_id,paper_id,conflict_type,task_ids_json,created_at
)
SELECT
    r.id,r.paper_id,'ambiguous_multiple_tasks',
    json_group_array(wbi.batch_id),MIN(wb.created_at)
FROM reviews r
JOIN workflow_batch_items wbi ON wbi.paper_id=r.paper_id
JOIN workflow_batches wb ON wb.id=wbi.batch_id
GROUP BY r.id,r.paper_id
HAVING COUNT(DISTINCT wbi.batch_id)>1;

COMMIT;
