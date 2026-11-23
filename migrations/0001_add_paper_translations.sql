PRAGMA foreign_keys = ON;
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS paper_translations (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL UNIQUE REFERENCES papers(id) ON DELETE CASCADE,
    chinese_title TEXT,
    chinese_abstract TEXT,
    title_translation_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (title_translation_status IN (
            'pending', 'ai_generated', 'human_edited', 'unavailable', 'failed'
        )),
    abstract_translation_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (abstract_translation_status IN (
            'pending', 'ai_generated', 'human_edited', 'unavailable', 'failed'
        )),
    title_translation_source TEXT
        CHECK (title_translation_source IN ('ai_generated', 'human_edited')),
    abstract_translation_source TEXT
        CHECK (abstract_translation_source IN ('ai_generated', 'human_edited')),
    model_name TEXT,
    prompt_version TEXT,
    translated_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_paper_translations_status
    ON paper_translations(title_translation_status, abstract_translation_status);

COMMIT;
