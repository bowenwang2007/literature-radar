from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from eduecon_radar.config import Settings
from eduecon_radar.db import Database, utcnow
from eduecon_radar.domain.evaluation import DomainEvaluator
from eduecon_radar.domain.gold_set import GoldSetLoader, create_gold_set_template
from eduecon_radar.domain.goldset_archive_gate import validate_archive_manifest
from eduecon_radar.domain.goldset_bundle import (
    build_model_bundle_candidate,
    inspect_model_bundle_candidate,
)
from eduecon_radar.domain.goldset_calibration import (
    SplitManifest,
    create_split_manifest,
    load_human_gold_snapshot,
    save_calibration_report,
    save_split_manifest,
    search_decision_policies,
)
from eduecon_radar.domain.goldset_pilot import (
    ControlledGoldPilot,
    PilotBudget,
    PilotRequest,
    SemanticGoldPilotProcessor,
    semantic_prompt_bundle_sha256,
    validate_pilot_preflight,
)
from eduecon_radar.domain.goldset_sampling import create_real_gold_set_sample
from eduecon_radar.domain.goldset_storage import (
    GoldSetRepository,
)
from eduecon_radar.domain.loader import DomainConfigLoader
from eduecon_radar.domain.semantic_evaluation import SemanticEvaluationService
from eduecon_radar.domain.semantic_judgment import DeepSeekSemanticAdapter
from eduecon_radar.domain.semantic_prompting import (
    SemanticPromptBuilder,
    load_semantic_engine_config,
)
from eduecon_radar.domain.semantic_storage import (
    apply_domain_semantic_migration,
    inspect_semantic_run_read_only,
)
from eduecon_radar.domain.storage import (
    apply_domain_evidence_migration,
    inspect_domain_evidence_read_only,
)
from eduecon_radar.journals import load_journals
from eduecon_radar.logging_config import configure_logging
from eduecon_radar.resources import verify_runtime_resources
from eduecon_radar.services.ai_budget import AIBudgetManager, BudgetLimits
from eduecon_radar.services.ai_screening import ScreeningService
from eduecon_radar.services.archive import (
    archive_database,
    compare_archived_protected_data,
    inspect_database_read_only,
)
from eduecon_radar.services.crossref import CrossrefClient
from eduecon_radar.services.deepseek import DeepSeekClient
from eduecon_radar.services.doi_audit import (
    audit_doi_normalization,
    generate_doi_repair_plan,
)
from eduecon_radar.services.exporting import (
    export_mentor_csv_text,
    export_mentor_markdown,
    export_personal_csv_text,
    export_personal_markdown,
)
from eduecon_radar.services.gold_governance import (
    GoldGovernanceRepository,
    prepare_gold_corpus,
)
from eduecon_radar.services.gold_metrics import evaluate_gold_predictions
from eduecon_radar.services.gold_pilot_governance import (
    GoldPilotBudget,
    GoldPilotRunner,
    evaluate_holdout_once,
    validate_gold_pilot_preflight,
)
from eduecon_radar.services.ingestion import SyncService
from eduecon_radar.services.jel_enrichment import (
    JelEnrichmentService,
    audit_jel_enrichment,
    build_providers,
    format_jel_report,
    load_jel_source_settings,
    normalize_jel_codes,
)
from eduecon_radar.services.metadata_provenance import (
    audit_metadata_provenance,
    inspect_paper_provenance,
    metadata_quality_report,
)
from eduecon_radar.services.migration_integrity import (
    baseline_migration_ledger,
    migrate_database,
    migration_status,
    status_payload,
    verify_migrations,
)
from eduecon_radar.services.migrations import (
    apply_p0_hardening_migration,
    apply_translation_migration,
    apply_workflow_migration,
    require_p0_hardening_schema,
)
from eduecon_radar.services.profile_quality import (
    compare_sheet,
    design_sample,
    format_comparison,
    format_intake,
    format_screening_basis,
    intake_sheet,
    load_assessments,
    measure_screening_basis,
    sample_assessments,
    write_sheet,
)
from eduecon_radar.services.prototype_workflow import mark_export_completed
from eduecon_radar.services.review import dashboard_stats
from eduecon_radar.services.runtime_bundle import RuntimeBundleResolver
from eduecon_radar.services.runtime_doctor import run_doctor
from eduecon_radar.services.schema_manifest import (
    generate_schema_manifest,
    verify_schema_manifest,
)
from eduecon_radar.services.semantic_smoke import (
    diagnose_semantic_runs,
    run_fast_semantic_smoke,
    semantic_settings_diagnostic,
)
from eduecon_radar.services.task_lock import TaskLeaseManager
from eduecon_radar.services.translation import (
    TranslationParameters,
    TranslationService,
    preview_translations,
)
from eduecon_radar.services.unpaywall import UnpaywallClient
from eduecon_radar.services.validation_batch import ValidationRepository
from eduecon_radar.services.validation_report import (
    build_validation_report,
    write_validation_report,
)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期必须为 YYYY-MM-DD") from exc


def _project_path(project_root: Path, value: Path) -> Path:
    return (value if value.is_absolute() else project_root / value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eduecon-radar",
        description=(
            "Literature Radar 命令行：多研究方向文献检索、初筛、人工审核、"
            "论文库与导出（产品路径），以及质量评估与治理工具（研究路径）"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="初始化 SQLite 数据库和期刊配置")

    sync = subparsers.add_parser("sync", help="从 Crossref 增量同步元数据并执行筛选")
    sync.add_argument("--since", type=parse_date, help="同步起始日期 YYYY-MM-DD")
    sync.add_argument("--journal", help="仅同步期刊短名，例如 AER")
    sync.add_argument("--limit", type=int, help="每本期刊最多抓取数量")
    sync.add_argument("--dry-run", action="store_true", help="只查询和计数，不写入论文")

    screen = subparsers.add_parser("screen", help="筛选尚未处理的论文")
    screen.add_argument("--limit", type=int, help="最多处理数量")
    screen.add_argument("--no-ai", action="store_true", help="只运行规则筛选")
    screen.add_argument(
        "--no-strong-model", action="store_true", help="即使边界或冲突也不调用强模型"
    )

    export = subparsers.add_parser("export", help="导出导师版或个人研究记录")
    export.add_argument("--format", choices=["markdown", "csv"], required=True)
    export.add_argument(
        "--mode",
        choices=["mentor", "personal"],
        default="mentor",
        help="mentor仅含原始元数据及正式翻译；personal包含AI和人工个人记录",
    )
    export.add_argument("--output", type=Path, help="输出路径；省略时写入 exports 目录")
    export.add_argument("--journal", help="仅导出期刊短名")
    export.add_argument("--topic", help="仅导出包含该主题的记录")
    export.add_argument("--from-date", type=parse_date)
    export.add_argument("--to-date", type=parse_date)
    export.add_argument(
        "--status",
        choices=["pending", "approved", "rejected", "needs_revision"],
        default="approved",
    )
    export.add_argument("--has-abstract", action="store_true", help="仅导出有原始摘要的论文")
    export.add_argument(
        "--include-human-note", action="store_true", help="导师版额外包含独立人工备注"
    )
    export.add_argument(
        "--task-id",
        type=int,
        help="v0.5 task id；提供后严格使用任务级人工审核状态",
    )

    translate = subparsers.add_parser("translate", help="生成独立的标题和原始摘要忠实翻译")
    translate.add_argument(
        "--status",
        choices=["pending", "approved", "rejected", "needs_revision"],
        default="approved",
    )
    translate.add_argument("--journal")
    translate.add_argument("--limit", type=int, default=10)
    translate.add_argument("--missing-only", dest="missing_only", action="store_true")
    translate.add_argument("--include-existing", dest="missing_only", action="store_false")
    translate.set_defaults(missing_only=True)
    translate.add_argument("--dry-run", action="store_true")
    translate.add_argument("--confirm-ai-cost", action="store_true")
    translate.add_argument("--overwrite-human", action="store_true")
    translate.add_argument("--confirm-overwrite-human", action="store_true")

    migration = subparsers.add_parser(
        "migrate-translations", help="显式安装非破坏性的Translation Layer"
    )
    migration.add_argument(
        "--backup-confirmed", action="store_true", help="确认目标数据库已完成备份"
    )
    workflow_migration = subparsers.add_parser(
        "migrate-workflows", help="显式安装非破坏性的Workflow Layer"
    )
    workflow_migration.add_argument(
        "--backup-confirmed", action="store_true", help="确认目标数据库已完成备份"
    )

    archive = subparsers.add_parser(
        "archive-production-database",
        help="只读核验并归档现有生产数据库，不修改源文件",
    )
    archive.add_argument("--source-path", type=Path)
    archive.add_argument("--archive-dir", type=Path, required=True)
    archive.add_argument("--label", default="v0.4.1-frozen")
    archive.add_argument("--confirm-read-only-source", action="store_true")

    gold_template = subparsers.add_parser(
        "create-gold-set-template", help="从只读数据库生成待人工标注的Gold Set模板"
    )
    gold_template.add_argument("--source-db", type=Path, required=True)
    gold_template.add_argument("--output", type=Path, required=True)
    gold_template.add_argument("--sample-size", type=int, default=20)
    gold_template.add_argument("--random-seed", type=int, default=20260719)
    gold_template.add_argument("--stratify", action="store_true")
    gold_template.add_argument("--include-existing-screening-metadata", action="store_true")
    gold_template.add_argument("--confirm-read-only-source", action="store_true")
    gold_template.add_argument("--overwrite", action="store_true")

    validate_gold = subparsers.add_parser(
        "validate-gold-set", help="验证Gold Set CSV格式和人工标注约束"
    )
    validate_gold.add_argument("--input", type=Path, required=True)
    validate_gold.add_argument("--require-labels", action="store_true")

    evaluate = subparsers.add_parser("evaluate-domain-evidence", help="生成未校准规则证据模拟评估")
    evaluate.add_argument("--gold-set", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--overwrite", action="store_true")

    inspect_evidence = subparsers.add_parser(
        "inspect-domain-evidence", help="只读查看论文的最新领域证据run"
    )
    inspect_evidence.add_argument("--database", type=Path, required=True)
    inspect_evidence.add_argument("--paper-id", type=int, required=True)

    evidence_migration = subparsers.add_parser(
        "migrate-domain-evidence", help="显式安装独立的领域证据run表"
    )
    evidence_migration.add_argument("--database", type=Path, required=True)
    evidence_migration.add_argument("--backup-confirmed", action="store_true")

    subparsers.add_parser("validate-semantic-prompts", help="离线验证Phase 4提示词清单和SHA-256")
    inspect_semantic = subparsers.add_parser(
        "inspect-semantic-run", help="只读查看论文最新的Phase 4 shadow decision"
    )
    inspect_semantic.add_argument("--database", type=Path, required=True)
    inspect_semantic.add_argument("--paper-id", type=int, required=True)
    diagnose_semantic = subparsers.add_parser(
        "diagnose-semantic", help="Read-only sanitized semantic failure diagnostics"
    )
    diagnose_semantic.add_argument("--database", type=Path, required=True)
    diagnose_semantic.add_argument("--paper-limit", type=int, default=1)
    semantic_smoke = subparsers.add_parser(
        "semantic-smoke", help="Explicitly run Fast semantic on one paper"
    )
    semantic_smoke.add_argument("--database", type=Path, required=True)
    semantic_smoke.add_argument("--paper-id", type=int, required=True)
    semantic_smoke.add_argument("--confirm-live-call", action="store_true")
    semantic_smoke.add_argument("--allow-unreleased-bundle", action="store_true")
    semantic_smoke.add_argument("--debug", action="store_true")
    inspect_bundle = subparsers.add_parser(
        "inspect-runtime-bundle",
        help="Read-only verification of the released production runtime bundle",
    )
    inspect_bundle.add_argument("--manifest", type=Path)
    profile_choices = ("production", "validation", "gold")
    for command in ("migration-status", "verify-migrations"):
        item = subparsers.add_parser(command)
        item.add_argument("--database", type=Path, required=True)
        item.add_argument("--profile", choices=profile_choices, required=True)
    migrate_integrity = subparsers.add_parser("migrate-database")
    migrate_integrity.add_argument("--database", type=Path, required=True)
    migrate_integrity.add_argument("--profile", choices=profile_choices, required=True)
    migrate_integrity.add_argument("--dry-run", action="store_true")
    migrate_integrity.add_argument("--apply", action="store_true")
    migrate_integrity.add_argument("--baseline-confirmed", action="store_true")
    baseline = subparsers.add_parser("baseline-migration-ledger")
    baseline.add_argument("--database", type=Path, required=True)
    baseline.add_argument("--profile", choices=profile_choices, required=True)
    baseline.add_argument("--confirm-baseline", action="store_true")
    reconcile = subparsers.add_parser(
        "reconcile-migration-ledger",
        help="按实际 schema 证据补记已落地但未登记的 migration（默认只读审计）",
    )
    reconcile.add_argument("--database", type=Path, required=True)
    reconcile.add_argument("--profile", choices=profile_choices, required=True)
    reconcile.add_argument(
        "--confirm-record",
        action="store_true",
        help="确认写入账本；不加该参数只做只读审计，不修改数据库",
    )
    reconcile.add_argument("--output", type=Path, help="可选：把完整报告写入 JSON 文件")
    generate_manifest = subparsers.add_parser(
        "generate-schema-manifest",
        help=(
            "生成 schema manifest；默认从临时新库，指定 --database 时"
            "从该库的真实 schema 与 migration ledger 生成"
        ),
    )
    generate_manifest.add_argument("--profile", choices=profile_choices, required=True)
    generate_manifest.add_argument("--output", type=Path)
    generate_manifest.add_argument(
        "--database",
        type=Path,
        help=(
            "从该数据库的真实 schema 与 migration ledger 生成（只读打开），"
            "可保留生产库真实的 inferred provenance；"
            "省略时沿用临时新库的旧行为"
        ),
    )
    verify_manifest = subparsers.add_parser("verify-schema-manifest")
    verify_manifest.add_argument("--database", type=Path, required=True)
    verify_manifest.add_argument("--profile", choices=profile_choices, required=True)
    verify_manifest.add_argument("--manifest", type=Path)
    verify_schema = subparsers.add_parser("verify-schema")
    verify_schema.add_argument("--database", type=Path, required=True)
    verify_schema.add_argument("--profile", choices=profile_choices, required=True)
    verify_schema.add_argument("--manifest", type=Path)
    subparsers.add_parser("verify-runtime-resources")
    audit_doi = subparsers.add_parser("audit-doi-normalization")
    audit_doi.add_argument("--database", type=Path, required=True)
    repair_doi = subparsers.add_parser("generate-doi-repair-plan")
    repair_doi.add_argument("--database", type=Path, required=True)
    quality = subparsers.add_parser("metadata-quality-report")
    quality.add_argument("--database", type=Path, required=True)
    audit_provenance = subparsers.add_parser("audit-metadata-provenance")
    audit_provenance.add_argument("--database", type=Path, required=True)
    inspect_provenance = subparsers.add_parser("inspect-paper-provenance")
    inspect_provenance.add_argument("--database", type=Path, required=True)
    inspect_provenance.add_argument("--paper-id", type=int, required=True)
    jel_report = subparsers.add_parser(
        "jel-report", help="只读：JEL 覆盖面、来源与未获取原因"
    )
    jel_report.add_argument("--database", type=Path, required=True)
    jel_report.add_argument("--json", action="store_true")
    jel_enrich = subparsers.add_parser(
        "enrich-jel", help="从已发布来源补全论文 JEL 分类码（默认不写库）"
    )
    jel_enrich.add_argument("--database", type=Path, required=True)
    jel_enrich.add_argument(
        "--provider",
        action="append",
        choices=["manual", "repec_ideas"],
        help="限定使用的来源；可重复。默认使用配置中已启用的来源",
    )
    jel_enrich.add_argument(
        "--enable-repec",
        action="store_true",
        help=(
            "本次运行启用 RePEc/IDEAS 来源（配置中默认关闭）。"
            "启用后按配置的限速与页数上限抓取其 JEL 分类列表页，并写入本地缓存"
        ),
    )
    jel_enrich.add_argument("--paper-id", type=int, action="append")
    jel_enrich.add_argument(
        "--all-papers",
        action="store_true",
        help="包含已有 JEL 码的论文（默认只处理尚无 JEL 码的论文）",
    )
    jel_enrich.add_argument("--limit", type=int)
    jel_enrich.add_argument("--manual-file", type=Path)
    jel_enrich.add_argument("--cache-dir", type=Path)
    jel_enrich.add_argument("--apply", action="store_true", help="真正写入数据库")
    set_jel = subparsers.add_parser(
        "set-jel-codes", help="人工录入已核对来源的 JEL 码（写入并记录 provenance）"
    )
    set_jel.add_argument("--database", type=Path, required=True)
    set_jel.add_argument("--paper-id", type=int, required=True)
    set_jel.add_argument("--codes", required=True, help="空格或逗号分隔，例如 'D91 D03'")
    set_jel.add_argument("--source-url", default=None, help="核对该码的出处链接")
    set_jel.add_argument("--note", default=None)
    quality_sample = subparsers.add_parser(
        "profile-quality-sample",
        help="抽取人工标注样本并导出标注表（不调用 AI）",
    )
    quality_sample.add_argument("--database", type=Path, required=True)
    quality_sample.add_argument(
        "--profile",
        action="append",
        help="限定 profile；可重复。省略时覆盖全部 profile（推荐：跨 profile 抽样才能分层）",
    )
    quality_sample.add_argument("--output", type=Path, required=True)
    quality_sample.add_argument(
        "--total",
        type=int,
        default=0,
        help="目标总行数（分层设计，建议 130）。设置后忽略 --per-decision/--per-basis",
    )
    quality_sample.add_argument(
        "--boundary-share",
        type=float,
        default=0.35,
        help="边界层占比（默认 0.35）；其余按各 profile 判断数比例抽取代表性样本",
    )
    quality_sample.add_argument(
        "--min-per-profile",
        type=int,
        default=15,
        help=(
            "每个 profile 在代表性层至少保留的行数（默认 15）。"
            "严格按比例会让小队列只分到 2-3 行，无法对该队列下任何结论"
        ),
    )
    quality_sample.add_argument("--per-decision", type=int, default=0)
    quality_sample.add_argument("--per-basis", type=int, default=0)
    quality_sample.add_argument("--paper-id", type=int, action="append")
    quality_sample.add_argument("--only-paper-id", type=int, action="append")
    quality_sample.add_argument("--seed", type=int, default=20260925)
    quality_sample.add_argument(
        "--blind",
        action="store_true",
        help="导出时不写出自动判断，避免标注被锚定",
    )
    quality_sample.add_argument(
        "--no-abstract", action="store_true", help="不导出摘要（仅标题筛选评估）"
    )
    quality_sample.add_argument("--json", action="store_true")
    quality_compare = subparsers.add_parser(
        "profile-quality-compare",
        help="比较人工标注与自动判断，报告 FP/FN/边界案例",
    )
    quality_compare.add_argument("--database", type=Path, required=True)
    quality_compare.add_argument("--sheet", type=Path, required=True)
    quality_compare.add_argument("--profile", default=None)
    quality_compare.add_argument(
        "--sampling-mode",
        default="balanced_per_decision",
        help="写入报告的抽样方式说明；平衡抽样不是语料库的随机样本",
    )
    quality_compare.add_argument("--json", action="store_true")
    quality_basis = subparsers.add_parser(
        "profile-quality-basis",
        help="只读：统计某 profile 的判断有多少来自 title-only、多少来自完整筛选",
    )
    quality_basis.add_argument("--database", type=Path, required=True)
    quality_basis.add_argument("--profile", required=True)
    quality_basis.add_argument("--json", action="store_true")
    quality_intake = subparsers.add_parser(
        "profile-quality-intake",
        help="只读：检查返回的标注表（盲性、完成率、skip、必填列）",
    )
    quality_intake.add_argument("--sheet", type=Path, required=True)
    quality_intake.add_argument("--json", action="store_true")
    quality_intake.add_argument(
        "--allow-unblinded",
        action="store_true",
        help="允许 auto_decision 非空（默认视为不可用于比较，因为标注者可能看到过答案）",
    )
    inspect_budget = subparsers.add_parser("inspect-task-budget")
    inspect_budget.add_argument("--database", type=Path, required=True)
    inspect_budget.add_argument("--task-id", type=int, required=True)
    estimate_budget = subparsers.add_parser("estimate-task-budget")
    estimate_budget.add_argument("--database", type=Path, required=True)
    estimate_budget.add_argument("--task-id", type=int, required=True)
    set_budget = subparsers.add_parser("set-task-budget")
    set_budget.add_argument("--database", type=Path, required=True)
    set_budget.add_argument("--task-id", type=int, required=True)
    set_budget.add_argument("--max-calls", type=int, required=True)
    set_budget.add_argument("--max-input-tokens", type=int, required=True)
    set_budget.add_argument("--max-output-tokens", type=int, required=True)
    set_budget.add_argument("--max-estimated-cost-microunits", type=int, required=True)
    set_budget.add_argument("--max-strong-calls", type=int, required=True)
    set_budget.add_argument("--max-translation-calls", type=int, required=True)
    set_budget.add_argument("--changed-by", required=True)
    set_budget.add_argument("--reason", required=True)
    unsettled = subparsers.add_parser("audit-unsettled-reservations")
    unsettled.add_argument("--database", type=Path, required=True)
    unsettled.add_argument("--task-id", type=int)
    reconcile = subparsers.add_parser("reconcile-ai-usage")
    reconcile.add_argument("--database", type=Path, required=True)
    reconcile.add_argument("--task-id", type=int)
    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--json", action="store_true")
    inspect_lock = subparsers.add_parser(
        "inspect-task-lock", help="Read-only inspection of a workflow task lease"
    )
    inspect_lock.add_argument("--database", type=Path, required=True)
    inspect_lock.add_argument("--task-id", type=int, required=True)
    recover_lock = subparsers.add_parser(
        "recover-stale-task-lock", help="Recover only an already expired workflow task lease"
    )
    recover_lock.add_argument("--database", type=Path, required=True)
    recover_lock.add_argument("--task-id", type=int, required=True)
    p0_migration = subparsers.add_parser(
        "migrate-p0-safety",
        help="Install additive Phase 6.6B task lease and task-review schema",
    )
    p0_migration.add_argument("--database", type=Path, required=True)
    p0_migration.add_argument("--backup-confirmed", action="store_true")
    p0_migration.add_argument("--archive-manifest", type=Path)
    semantic_migration = subparsers.add_parser(
        "migrate-domain-semantic", help="在已验证备份保护下安装独立Phase 4语义run表"
    )
    semantic_migration.add_argument("--database", type=Path, required=True)
    semantic_migration.add_argument("--backup-confirmed", action="store_true")
    semantic_migration.add_argument("--confirm-non-production-db", action="store_true")
    semantic_migration.add_argument(
        "--archive-manifest",
        type=Path,
        help="生产数据库归档manifest；省略时仅检查数据库同级archives目录",
    )
    for command, help_text in (
        ("evaluate-semantic-judgments", "离线评估合成Phase 4判断fixture"),
        ("run-semantic-fixtures", "运行固定的离线Phase 4模拟验收"),
        ("compare-shadow-decisions", "比较规则、fast与provisional shadow结果"),
    ):
        semantic_eval = subparsers.add_parser(command, help=help_text)
        semantic_eval.add_argument("--fixtures", type=Path)
        semantic_eval.add_argument("--output", type=Path)
        semantic_eval.add_argument("--overwrite", action="store_true")

    init_gold = subparsers.add_parser(
        "init-gold-set-database", help="Initialize an independent Phase 5 Gold Set database"
    )
    init_gold.add_argument("--database", type=Path, required=True)
    init_gold.add_argument("--confirm-non-production-db", action="store_true")

    real_gold = subparsers.add_parser(
        "create-real-gold-set", help="Sample real papers from a read-only source database"
    )
    real_gold.add_argument("--source-db", type=Path, required=True)
    real_gold.add_argument("--output-dir", type=Path, required=True)
    real_gold.add_argument("--gold-set-id", required=True)
    real_gold.add_argument("--sample-size", type=int, default=300)
    real_gold.add_argument("--random-seed", type=int, default=20260719)
    real_gold.add_argument("--confirm-read-only-source", action="store_true")

    import_gold = subparsers.add_parser(
        "import-gold-set-items", help="Import a blank sample into an independent Gold database"
    )
    import_gold.add_argument("--database", type=Path, required=True)
    import_gold.add_argument("--items", type=Path, required=True)
    import_gold.add_argument("--sampling-manifest", type=Path, required=True)
    import_gold.add_argument("--name", required=True)
    import_gold.add_argument("--annotation-version", required=True)
    import_gold.add_argument(
        "--review-mode",
        choices=["dual_reviewer", "single_reviewer_two_rounds"],
        default="single_reviewer_two_rounds",
    )
    import_gold.add_argument("--minimum-round-interval-days", type=int, default=7)

    for command in ("validate-gold-set-annotations", "inspect-gold-set-consistency"):
        inspect_gold = subparsers.add_parser(command)
        inspect_gold.add_argument("--database", type=Path, required=True)
        inspect_gold.add_argument("--gold-set-id", type=int, required=True)

    snapshot_gold = subparsers.add_parser("export-gold-set-snapshot")
    snapshot_gold.add_argument("--database", type=Path, required=True)
    snapshot_gold.add_argument("--gold-set-id", type=int, required=True)
    snapshot_gold.add_argument("--snapshot-version", required=True)
    snapshot_gold.add_argument("--output", type=Path, required=True)

    pilot = subparsers.add_parser("run-semantic-gold-pilot")
    pilot.add_argument("--gold-set-snapshot", type=Path, required=True)
    pilot.add_argument("--gold-set-sha256", required=True)
    pilot.add_argument("--output-db", type=Path, required=True)
    pilot.add_argument("--archive-manifest", type=Path, required=True)
    pilot.add_argument("--model-bundle-version", required=True)
    pilot.add_argument("--config-bundle-sha256", required=True)
    pilot.add_argument("--run-uid", required=True)
    pilot.add_argument("--max-papers", type=int, required=True)
    pilot.add_argument("--max-fast-calls", type=int, required=True)
    pilot.add_argument("--max-strong-calls", type=int, required=True)
    pilot.add_argument("--max-input-tokens", type=int, required=True)
    pilot.add_argument("--max-output-tokens", type=int, required=True)
    pilot.add_argument("--budget-limit", type=float, required=True)
    pilot.add_argument("--max-retries", type=int, default=1)
    pilot.add_argument("--failure-rate-threshold", type=float, default=0.25)
    pilot.add_argument("--allow-live-ai", action="store_true")
    pilot.add_argument("--confirm", required=True)
    pilot.add_argument("--resume-pilot-run", action="store_true")

    for command in ("evaluate-calibration-run", "search-decision-policies"):
        calibrate = subparsers.add_parser(command)
        calibrate.add_argument("--gold-set-snapshot", type=Path, required=True)
        calibrate.add_argument("--gold-set-sha256", required=True)
        calibrate.add_argument("--predictions", type=Path, required=True)
        calibrate.add_argument("--split-manifest", type=Path, required=True)
        calibrate.add_argument("--output", type=Path, required=True)
        calibrate.add_argument("--database", type=Path)
        calibrate.add_argument("--random-seed", type=int, default=20260719)
        calibrate.add_argument("--minimum-validation-recall", type=float, default=0.9)

    compare = subparsers.add_parser("compare-strategy-candidates")
    compare.add_argument("--calibration-report", type=Path, required=True)

    build_candidate = subparsers.add_parser("build-model-bundle-candidate")
    build_candidate.add_argument("--database", type=Path, required=True)
    build_candidate.add_argument("--archive-manifest", type=Path, required=True)
    build_candidate.add_argument("--gold-set-snapshot", type=Path, required=True)
    build_candidate.add_argument("--gold-set-sha256", required=True)
    build_candidate.add_argument("--gold-set-id", type=int, required=True)
    build_candidate.add_argument("--candidate-version", required=True)
    build_candidate.add_argument("--calibration-report", type=Path, required=True)
    build_candidate.add_argument("--red-team-report", type=Path, required=True)
    build_candidate.add_argument("--budget-report", type=Path, required=True)
    build_candidate.add_argument("--known-limitations", type=Path, required=True)
    build_candidate.add_argument("--output", type=Path, required=True)

    inspect_candidate = subparsers.add_parser("inspect-model-bundle-candidate")
    inspect_candidate.add_argument("--input", type=Path, required=True)

    plan_corpus = subparsers.add_parser(
        "plan-gold-sample",
        help="Plan a deterministic Gold sample from an explicitly read-only source DB",
    )
    plan_corpus.add_argument("--source-database", type=Path, required=True)
    plan_corpus.add_argument("--sample-plan", type=Path, required=True)
    plan_corpus.add_argument("--output-manifest", type=Path, required=True)
    plan_corpus.add_argument("--corpus-version")

    prepare_corpus = subparsers.add_parser(
        "prepare-gold-corpus",
        help="Freeze sampled source metadata into an independent Gold database; never calls AI",
    )
    prepare_corpus.add_argument("--source-database", type=Path, required=True)
    prepare_corpus.add_argument("--gold-database", type=Path, required=True)
    prepare_corpus.add_argument("--sample-plan", type=Path, required=True)
    prepare_corpus.add_argument("--output-manifest", type=Path, required=True)
    prepare_corpus.add_argument("--corpus-version")
    prepare_corpus.add_argument("--dry-run", action="store_true")

    for command in ("inspect-gold-corpus", "verify-gold-corpus"):
        item = subparsers.add_parser(command, help="Read-only Gold corpus inspection")
        item.add_argument("--gold-database", type=Path, required=True)
        item.add_argument("--corpus-version", required=True)

    for command in (
        "gold-annotation-status",
        "inspect-gold-review-queue",
        "calculate-annotation-agreement",
        "build-duplicate-clusters",
        "review-duplicate-candidates",
    ):
        item = subparsers.add_parser(command, help="Offline Gold annotation governance")
        item.add_argument("--gold-database", type=Path, required=True)
        item.add_argument("--corpus-version", required=True)

    create_split = subparsers.add_parser(
        "create-gold-split", help="Create an immutable group-aware leakage-safe split"
    )
    create_split.add_argument("--gold-database", type=Path, required=True)
    create_split.add_argument("--corpus-version", required=True)
    create_split.add_argument("--split-version", required=True)
    create_split.add_argument("--random-seed", type=int, default=20260726)
    create_split.add_argument("--development", type=float, default=0.20)
    create_split.add_argument("--calibration", type=float, default=0.50)
    create_split.add_argument("--holdout", type=float, default=0.30)

    verify_split = subparsers.add_parser(
        "verify-gold-split", help="Verify split SHA256 and duplicate-cluster isolation"
    )
    verify_split.add_argument("--gold-database", type=Path, required=True)
    verify_split.add_argument("--split-manifest-sha256", required=True)

    freeze_snapshot = subparsers.add_parser(
        "freeze-gold-snapshot", help="Freeze a governed immutable Gold snapshot"
    )
    freeze_snapshot.add_argument("--gold-database", type=Path, required=True)
    freeze_snapshot.add_argument("--corpus-version", required=True)
    freeze_snapshot.add_argument("--snapshot-version", required=True)
    freeze_snapshot.add_argument("--split-manifest-sha256", required=True)
    freeze_snapshot.add_argument("--protocol", type=Path, required=True)
    freeze_snapshot.add_argument("--known-limitations", type=Path, required=True)
    freeze_snapshot.add_argument("--output", type=Path, required=True)

    for command in ("inspect-gold-snapshot", "verify-gold-snapshot"):
        item = subparsers.add_parser(command, help="Read-only immutable Gold snapshot inspection")
        item.add_argument("--gold-database", type=Path, required=True)
        item.add_argument("--snapshot-sha256", required=True)

    compare_snapshots = subparsers.add_parser(
        "compare-gold-snapshots", help="Compare two immutable Gold snapshots"
    )
    compare_snapshots.add_argument("--gold-database", type=Path, required=True)
    compare_snapshots.add_argument("--left-sha256", required=True)
    compare_snapshots.add_argument("--right-sha256", required=True)

    gold_pilot = subparsers.add_parser(
        "run-gold-ai-pilot",
        help="Explicitly confirmed, budgeted Gold AI Pilot; never permits Holdout",
    )
    gold_pilot.add_argument("--gold-database", type=Path, required=True)
    gold_pilot.add_argument("--production-database", type=Path, required=True)
    gold_pilot.add_argument("--gold-snapshot", type=Path, required=True)
    gold_pilot.add_argument("--gold-snapshot-sha256", required=True)
    gold_pilot.add_argument("--candidate-runtime-config", type=Path, required=True)
    gold_pilot.add_argument("--candidate-runtime-config-sha256", required=True)
    gold_pilot.add_argument("--split", choices=("development", "calibration"), required=True)
    gold_pilot.add_argument("--run-uid", required=True)
    gold_pilot.add_argument("--max-calls", type=int, required=True)
    gold_pilot.add_argument("--max-input-tokens", type=int, required=True)
    gold_pilot.add_argument("--max-output-tokens", type=int, required=True)
    gold_pilot.add_argument("--max-cost", type=int, required=True)
    gold_pilot.add_argument("--confirm-live-ai", action="store_true")
    gold_pilot.add_argument("--confirm-labels-not-in-prompt", action="store_true")
    gold_pilot.add_argument("--confirm-isolated-gold-database", action="store_true")
    gold_pilot.add_argument("--confirm-production-database-readonly", action="store_true")

    for command in ("inspect-gold-pilot", "reconcile-gold-pilot-usage"):
        item = subparsers.add_parser(command, help="Inspect/resume/reconcile a governed Gold pilot")
        item.add_argument("--gold-database", type=Path, required=True)
        item.add_argument("--run-uid", required=True)

    resume_pilot = subparsers.add_parser(
        "resume-gold-pilot",
        help="Resume a governed Gold Pilot with all confirmations and budgets repeated",
    )
    for action in gold_pilot._actions[1:]:  # argparse has no public clone API
        if action.dest in {"help"}:
            continue
        kwargs: dict[str, object] = {"dest": action.dest}
        if action.option_strings:
            kwargs.pop("dest")
            if action.type is not None:
                kwargs["type"] = action.type
            if action.choices is not None:
                kwargs["choices"] = action.choices
            if isinstance(action, argparse._StoreTrueAction):
                kwargs["action"] = "store_true"
            else:
                kwargs["required"] = action.required
            resume_pilot.add_argument(*action.option_strings, **kwargs)

    calibration = subparsers.add_parser(
        "evaluate-calibration", help="Evaluate offline predictions without reading Holdout labels"
    )
    calibration.add_argument("--predictions", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)

    holdout_eval = subparsers.add_parser(
        "evaluate-holdout", help="One-time, explicitly confirmed final Holdout evaluation"
    )
    holdout_eval.add_argument("--gold-database", type=Path, required=True)
    holdout_eval.add_argument("--snapshot-sha256", required=True)
    holdout_eval.add_argument("--candidate-bundle-sha256", required=True)
    holdout_eval.add_argument("--split-manifest-sha256", required=True)
    holdout_eval.add_argument("--predictions", type=Path, required=True)
    holdout_eval.add_argument("--confirm-final-candidate", action="store_true")
    holdout_eval.add_argument("--confirm-holdout-one-time-evaluation", action="store_true")

    validation_sample = subparsers.add_parser(
        "create-validation-sample",
        help="Read-only stratified sampling from the production paper database",
    )
    validation_sample.add_argument("--journal", action="append", default=[])
    validation_sample.add_argument("--year-from", type=int)
    validation_sample.add_argument("--year-to", type=int)
    validation_sample.add_argument("--size", type=int, default=100)
    validation_sample.add_argument("--random-seed", type=int, default=20260723)
    validation_sample.add_argument(
        "--validation-db", type=Path, default=Path("data/validation.db")
    )
    validation_sample.add_argument("--output", type=Path)

    validate_screening = subparsers.add_parser(
        "validate-screening",
        help="Evaluate AI decisions against human labels in a validation batch",
    )
    validate_screening.add_argument("--validation-batch-id", type=int, required=True)
    validate_screening.add_argument(
        "--validation-db", type=Path, default=Path("data/validation.db")
    )
    validate_screening.add_argument(
        "--output-dir", type=Path, default=Path("exports/validation")
    )
    validate_screening.add_argument("--overwrite", action="store_true")

    subparsers.add_parser("stats", help="显示 AI 筛选、人工审核与最近同步统计")

    process_papers = subparsers.add_parser(
        "process-papers",
        help="续跑工作台处理链（摘要补全、标题翻译、初筛），处理未完成论文",
    )
    process_papers.add_argument(
        "--sync-run-id",
        type=int,
        help="目标检索批次；省略时使用最近一次检索",
    )
    process_papers.add_argument(
        "--limit",
        type=int,
        help="本次最多处理多少篇，按论文 id 升序",
    )
    process_papers.add_argument(
        "--yes",
        action="store_true",
        help="确认执行；省略时只列出将要处理的论文，不发起任何网络或 AI 请求",
    )

    lifecycle_check = subparsers.add_parser(
        "lifecycle-check",
        help="只读检查论文生命周期不变量（有摘要但无判断、状态搁浅等）",
    )
    lifecycle_check.add_argument(
        "--sync-run-id",
        type=int,
        help="仅检查某个检索批次；省略时检查全部有处理记录的论文",
    )
    return parser


def _screening_service(settings: Settings, database: Database, *, no_ai: bool) -> ScreeningService:
    client = None
    if not no_ai:
        if not settings.deepseek_api_key:
            raise ValueError("未配置 DEEPSEEK_API_KEY。请写入本地 .env，或显式使用 --no-ai。")
        client = DeepSeekClient(
            api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url
        )
    return ScreeningService(
        database,
        settings.prompts_dir,
        client,
        settings.deepseek_model_fast,
        settings.deepseek_model_strong,
    )


def _current_git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


SEMANTIC_ARCHIVE_MAX_AGE = timedelta(days=7)


def _require_recent_semantic_archive(production_database: Path, manifest_path: Path | None) -> Path:
    production = production_database.resolve()
    if manifest_path is not None:
        candidates = [manifest_path.resolve()]
    else:
        archive_dir = production.parent / "archives"
        candidates = sorted(
            archive_dir.glob("*.manifest.json") if archive_dir.is_dir() else (),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
    failures: list[str] = []
    now = datetime.now(UTC)
    for candidate in candidates:
        gate = validate_archive_manifest(
            candidate,
            production_database=production,
            active_database=production,
            require_git_commit=False,
        )
        if not gate.valid and set(gate.reasons) == {"current_source_sha256_mismatch"}:
            relaxed = validate_archive_manifest(
                candidate,
                production_database=production,
                active_database=production,
                source_change_explanation="verify additive schema change below",
                require_git_commit=False,
            )
            protected_differences: tuple[str, ...] = ()
            if relaxed.valid and relaxed.archive_path:
                integrity, _ = inspect_database_read_only(production)
                if integrity == "ok":
                    protected_differences = compare_archived_protected_data(
                        production, Path(relaxed.archive_path)
                    )
            if relaxed.valid and not protected_differences:
                gate = relaxed
            elif protected_differences:
                failures.append(f"{candidate.name}: {','.join(protected_differences)}")
                continue
        if not gate.valid:
            failures.append(f"{candidate.name}: {','.join(gate.reasons)}")
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(str(payload["created_at"]))
            if created_at.tzinfo is None:
                raise ValueError("created_at has no timezone")
            age = now - created_at.astimezone(UTC)
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            failures.append(f"{candidate.name}: invalid_created_at:{exc}")
            continue
        if age < timedelta(minutes=-5):
            failures.append(f"{candidate.name}: archive_created_at_in_future")
            continue
        if age > SEMANTIC_ARCHIVE_MAX_AGE:
            failures.append(f"{candidate.name}: archive_manifest_too_old")
            continue
        return candidate
    detail = "; ".join(failures[:5]) or "no manifest found"
    raise ValueError(
        "Production semantic migration requires a valid archive manifest created within "
        f"the last {SEMANTIC_ARCHIVE_MAX_AGE.days} days: {detail}"
    )


def run(args: argparse.Namespace, settings: Settings) -> int:
    project_root = Path(getattr(settings, "project_root", Path.cwd()))
    semantic_config_dir = getattr(
        settings,
        "domain_config_dir",
        project_root / "config" / "domains" / "education_economics",
    )
    semantic_fixture = (
        semantic_config_dir / "gold_sets" / "fixtures" / "phase4_semantic_judgments.json"
    )
    if args.command in {"migration-status", "verify-migrations"}:
        target = _project_path(project_root, args.database)
        status = (
            migration_status(target, args.profile)
            if args.command == "migration-status"
            else verify_migrations(target, args.profile)
        )
        print(json.dumps(status_payload(status), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if not status.errors else 1
    if args.command == "migrate-database":
        if args.apply and args.dry_run:
            raise ValueError("Choose either --dry-run or --apply")
        target = _project_path(project_root, args.database)
        status = migrate_database(
            target,
            args.profile,
            dry_run=not args.apply,
            baseline_confirmed=args.baseline_confirmed,
        )
        print(json.dumps(status_payload(status), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "baseline-migration-ledger":
        target = _project_path(project_root, args.database)
        status = baseline_migration_ledger(
            target, args.profile, confirmed=args.confirm_baseline
        )
        print(json.dumps(status_payload(status), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "reconcile-migration-ledger":
        from eduecon_radar.services.migration_integrity import PROFILE_MIGRATIONS
        from eduecon_radar.services.migration_ledger import (
            reconcile_migration_ledger,
        )

        target = _project_path(project_root, args.database)
        report = reconcile_migration_ledger(
            target,
            args.profile,
            PROFILE_MIGRATIONS[args.profile],
            confirmed=args.confirm_record,
        )
        if args.output:
            output = _project_path(project_root, args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            report = {**report, "report_path": str(output.resolve())}
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if report.get("status") == "blocked" else 0
    if args.command == "generate-schema-manifest":
        output = (
            _project_path(project_root, args.output)
            if args.output
            else project_root / "manifests" / f"{args.profile}_schema_manifest.json"
        )
        # Two sources, deliberately distinct.  Without --database the manifest
        # describes a fresh database built by running the migrations; with it the
        # manifest describes the given database's real schema *and* its real
        # migration ledger, which is the only way to keep a production ledger's
        # `inferred` provenance.  The two are not interchangeable:
        # `verify-schema` refuses a fresh-built manifest against a reconciled
        # production ledger (and vice versa), so passing the wrong source cannot
        # silently launder provenance.
        target_database = (
            _project_path(project_root, args.database) if args.database else None
        )
        result = generate_schema_manifest(
            args.profile, output=output, database=target_database
        )
        print(
            json.dumps(
                {
                    "database_profile": args.profile,
                    "source": (
                        str(target_database.resolve())
                        if target_database is not None
                        else "fresh_database_from_migrations"
                    ),
                    "output": str(output.resolve()),
                    "manifest_sha256": result["manifest_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command in {"verify-schema-manifest", "verify-schema"}:
        target = _project_path(project_root, args.database)
        manifest = (
            _project_path(project_root, args.manifest)
            if args.manifest
            else project_root / "manifests" / f"{args.profile}_schema_manifest.json"
        )
        result = verify_schema_manifest(target, args.profile, manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "verify-runtime-resources":
        result = verify_runtime_resources()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] == "ok" else 1
    if args.command in {"audit-doi-normalization", "generate-doi-repair-plan"}:
        target = _project_path(project_root, args.database)
        result = (
            audit_doi_normalization(target)
            if args.command == "audit-doi-normalization"
            else generate_doi_repair_plan(target)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command in {
        "metadata-quality-report",
        "audit-metadata-provenance",
        "inspect-paper-provenance",
    }:
        target = _project_path(project_root, args.database)
        if args.command == "metadata-quality-report":
            result = metadata_quality_report(target)
        elif args.command == "audit-metadata-provenance":
            result = audit_metadata_provenance(target)
        else:
            result = inspect_paper_provenance(target, args.paper_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.get("status") != "blocked" else 1
    if args.command == "jel-report":
        target = _project_path(project_root, args.database)
        report = audit_jel_enrichment(target)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(format_jel_report(report))
            if report.get("status") == "partial":
                return 1
        return 0
    if args.command in {"enrich-jel", "set-jel-codes"}:
        return _run_jel_command(args, project_root)
    if args.command in {
        "profile-quality-sample",
        "profile-quality-compare",
        "profile-quality-basis",
        "profile-quality-intake",
    }:
        return _run_profile_quality_command(args, project_root)
    if args.command in {
        "inspect-task-budget",
        "estimate-task-budget",
        "set-task-budget",
        "audit-unsettled-reservations",
        "reconcile-ai-usage",
    }:
        target = _project_path(project_root, args.database)
        manager = AIBudgetManager(Database(target))
        if args.command in {"inspect-task-budget", "estimate-task-budget"}:
            result = manager.inspect(args.task_id)
        elif args.command == "set-task-budget":
            result = manager.set_budget(
                args.task_id,
                BudgetLimits(
                    args.max_calls,
                    args.max_input_tokens,
                    args.max_output_tokens,
                    args.max_estimated_cost_microunits,
                    args.max_strong_calls,
                    args.max_translation_calls,
                ),
                changed_by=args.changed_by,
                reason=args.reason,
            )
        else:
            unsettled_rows = manager.unsettled(args.task_id)
            result = {
                "status": "warning" if unsettled_rows else "ok",
                "mode": "read_only_reconciliation"
                if args.command == "reconcile-ai-usage"
                else "audit",
                "reservations": unsettled_rows,
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if result.get("status") == "warning" else 0
    if args.command == "doctor":
        result = run_doctor(settings)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"Literature Radar doctor: {result['status']}")
            for check in result["checks"]:
                print(f"[{check['status']}] {check['name']}: {check['detail']}")
        return 0 if result["status"] in {"ok", "warning"} else 1
    if args.command == "inspect-runtime-bundle":
        result = RuntimeBundleResolver(project_root).inspect(
            _project_path(project_root, args.manifest) if args.manifest else None
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["trusted"] else 1
    if args.command in {"inspect-task-lock", "recover-stale-task-lock"}:
        target = _project_path(project_root, args.database)
        if not target.is_file():
            raise FileNotFoundError(f"Database does not exist: {target}")
        require_p0_hardening_schema(Database(target))
        manager = TaskLeaseManager(Database(target))
        result = (
            manager.inspect(args.task_id)
            if args.command == "inspect-task-lock"
            else manager.recover_stale(args.task_id)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "migrate-p0-safety":
        if not args.backup_confirmed:
            raise ValueError("migrate-p0-safety requires --backup-confirmed")
        target = _project_path(project_root, args.database)
        if target == settings.database_path.resolve():
            _require_recent_semantic_archive(
                target,
                _project_path(project_root, args.archive_manifest)
                if args.archive_manifest
                else None,
            )
        apply_p0_hardening_migration(
            Database(target),
            project_root / "migrations" / "0007_add_runtime_leases_and_task_reviews.sql",
        )
        print(json.dumps({"database": str(target), "phase_6_6b_schema": True}))
        return 0
    if args.command == "create-validation-sample":
        source_database = Path(settings.database_path).resolve()
        validation_path = _project_path(project_root, args.validation_db)
        if validation_path == source_database:
            raise ValueError(
                "Validation database must be separate from the source database"
            )
        repository = ValidationRepository(Database(validation_path))
        repository.initialize(
            project_root / "migrations" / "0006_add_validation_batches.sql"
        )
        result = repository.create_sample(
            source_database=source_database,
            journals=tuple(args.journal),
            year_from=args.year_from,
            year_to=args.year_to,
            size=args.size,
            random_seed=args.random_seed,
        )
        output = (
            _project_path(project_root, args.output)
            if args.output
            else project_root
            / "exports"
            / f"validation_batch_{result.batch.batch_id}_annotations.csv"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Validation annotation template exists: {output}")
        output.write_text(result.annotation_csv, encoding="utf-8")
        print(
            json.dumps(
                {
                    "validation_database": str(validation_path),
                    "batch": asdict(result.batch),
                    "stratum_counts": result.stratum_counts,
                    "annotation_template": str(output.resolve()),
                    "source_database_mode": "read-only",
                    "ai_calls": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "validate-screening":
        validation_path = _project_path(project_root, args.validation_db)
        if not validation_path.is_file():
            raise FileNotFoundError(f"Validation database does not exist: {validation_path}")
        repository = ValidationRepository(Database(validation_path))
        report = build_validation_report(repository, args.validation_batch_id)
        outputs = write_validation_report(
            report,
            output_dir=_project_path(project_root, args.output_dir),
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "validation_batch_id": args.validation_batch_id,
                    "performance": report["performance"],
                    "outputs": outputs,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "diagnose-semantic":
        if args.paper_limit < 1 or args.paper_limit > 10:
            raise ValueError("--paper-limit must be between 1 and 10")
        result = {
            "configuration": semantic_settings_diagnostic(settings),
            "request_schema": "FastSemanticResult",
            **diagnose_semantic_runs(args.database.resolve(), paper_limit=args.paper_limit),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "semantic-smoke":
        if not args.confirm_live_call:
            raise ValueError("semantic-smoke requires --confirm-live-call")
        result = run_fast_semantic_smoke(
            database=Database(args.database.resolve()),
            settings=settings,
            paper_id=args.paper_id,
            debug=getattr(args, "debug", False),
            allow_unreleased_bundle=getattr(args, "allow_unreleased_bundle", False),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] in {"succeeded", "already_succeeded"} else 1
    if args.command in {"plan-gold-sample", "prepare-gold-corpus"}:
        source = args.source_database.resolve()
        target = (
            args.gold_database.resolve()
            if args.command == "prepare-gold-corpus"
            else project_root / "data" / "development" / "_gold_plan_only.db"
        )
        result = prepare_gold_corpus(
            source_database=source,
            gold_database=target,
            sample_plan_path=args.sample_plan.resolve(),
            output_manifest=args.output_manifest.resolve(),
            corpus_version=args.corpus_version,
            dry_run=args.command == "plan-gold-sample" or args.dry_run,
            ontology_path=semantic_config_dir / "ontology.yaml",
        )
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command in {"inspect-gold-corpus", "verify-gold-corpus"}:
        repository = GoldGovernanceRepository(Database(args.gold_database.resolve()))
        result = (
            repository.inspect_corpus(args.corpus_version)
            if args.command == "inspect-gold-corpus"
            else repository.verify_corpus(args.corpus_version)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result.get("valid", True) else 1
    if args.command in {
        "gold-annotation-status",
        "inspect-gold-review-queue",
        "calculate-annotation-agreement",
        "build-duplicate-clusters",
        "review-duplicate-candidates",
    }:
        repository = GoldGovernanceRepository(Database(args.gold_database.resolve()))
        if args.command == "gold-annotation-status":
            result = repository.annotation_status(args.corpus_version)
        elif args.command == "inspect-gold-review-queue":
            result = repository.review_queue(args.corpus_version)
        elif args.command == "calculate-annotation-agreement":
            result = asdict(repository.calculate_agreement(args.corpus_version))
        elif args.command == "build-duplicate-clusters":
            result = repository.build_duplicate_clusters(args.corpus_version)
        else:
            result = repository.duplicate_candidates(args.corpus_version)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "create-gold-split":
        result = GoldGovernanceRepository(Database(args.gold_database.resolve())).create_split(
            corpus_version=args.corpus_version,
            split_version=args.split_version,
            random_seed=args.random_seed,
            development=args.development,
            calibration=args.calibration,
            holdout=args.holdout,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "verify-gold-split":
        result = GoldGovernanceRepository(Database(args.gold_database.resolve())).verify_split(
            args.split_manifest_sha256
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["valid"] else 1
    if args.command == "freeze-gold-snapshot":
        result = GoldGovernanceRepository(Database(args.gold_database.resolve())).freeze_snapshot(
            corpus_version=args.corpus_version,
            snapshot_version=args.snapshot_version,
            split_manifest_sha256=args.split_manifest_sha256,
            protocol_path=args.protocol.resolve(),
            output=args.output.resolve(),
            git_commit=_current_git_commit(project_root),
            known_limitations=_load_string_list(args.known_limitations),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command in {"inspect-gold-snapshot", "verify-gold-snapshot"}:
        result = GoldGovernanceRepository(Database(args.gold_database.resolve())).inspect_snapshot(
            args.snapshot_sha256
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["hash_valid"] else 1
    if args.command == "compare-gold-snapshots":
        result = GoldGovernanceRepository(
            Database(args.gold_database.resolve())
        ).compare_snapshots(args.left_sha256, args.right_sha256)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command in {"run-gold-ai-pilot", "resume-gold-pilot"}:
        budget = GoldPilotBudget(
            max_calls=args.max_calls,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            max_cost_microunits=args.max_cost,
        )
        preflight = validate_gold_pilot_preflight(
            gold_database=args.gold_database,
            production_database=args.production_database,
            snapshot_path=args.gold_snapshot,
            snapshot_sha256=args.gold_snapshot_sha256,
            candidate_runtime_config=args.candidate_runtime_config,
            candidate_runtime_config_sha256=args.candidate_runtime_config_sha256,
            split_name=args.split,
            budget=budget,
            confirm_live_ai=args.confirm_live_ai,
            confirm_labels_not_in_prompt=args.confirm_labels_not_in_prompt,
            confirm_isolated_gold_database=args.confirm_isolated_gold_database,
            confirm_production_database_readonly=args.confirm_production_database_readonly,
        )
        bundle = DomainConfigLoader(semantic_config_dir).load()
        engine = load_semantic_engine_config(semantic_config_dir)
        prompts = SemanticPromptBuilder(semantic_config_dir / "prompts")
        candidate_runtime = json.loads(
            args.candidate_runtime_config.read_text(encoding="utf-8")
        )
        if (
            candidate_runtime["provider"] != engine.fast.provider
            or candidate_runtime["fast_model"] != engine.fast.model
            or candidate_runtime["strong_model"] != engine.strong_review.model
            or candidate_runtime["prompt_bundle_sha256"]
            != semantic_prompt_bundle_sha256(prompts)
        ):
            raise ValueError(
                "Candidate runtime config identities do not match the inspected runtime"
            )
        if not (settings.deepseek_api_key or "").strip():
            raise ValueError("Gold AI Pilot requires an explicitly configured API key")
        adapter = DeepSeekSemanticAdapter(
            DeepSeekClient(
                api_key=settings.deepseek_api_key or "",
                base_url=settings.deepseek_base_url,
            )
        )
        semantic_processor = SemanticGoldPilotProcessor(
            bundle=bundle, engine=engine, prompts=prompts, client=adapter
        )

        def processor(item: dict) -> dict:
            return semantic_processor(
                {
                    "item_id": item["gold_paper_id"],
                    "doi": item.get("canonical_doi"),
                    "original_title": item["original_title"],
                    "original_abstract": item.get("abstract"),
                    "authors": item.get("authors") or [],
                    "journal": item.get("journal"),
                    "journal_group": "gold_pilot",
                    "publication_date": item.get("publication_date"),
                    "keywords": [],
                    "subject_codes": [],
                    "difficulty_tags": [],
                },
                allow_strong=True,
            )

        result = GoldPilotRunner(Database(args.gold_database.resolve())).run(
            run_uid=args.run_uid,
            preflight=preflight,
            budget=budget,
            processor=processor,
            resume=args.command == "resume-gold-pilot",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result["status"] == "completed" else 1
    if args.command in {"inspect-gold-pilot", "reconcile-gold-pilot-usage"}:
        runner = GoldPilotRunner(Database(args.gold_database.resolve()))
        result = (
            runner.inspect(args.run_uid)
            if args.command == "inspect-gold-pilot"
            else runner.reconcile_usage(args.run_uid)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "evaluate-calibration":
        payload = json.loads(args.predictions.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        if any(str(item.get("split_name")) == "holdout" for item in rows):
            raise ValueError("Calibration evaluation must not read Holdout rows or labels")
        result = evaluate_gold_predictions(rows)
        _write_safe_json(
            args.output,
            args.predictions,
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            False,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evaluate-holdout":
        payload = json.loads(args.predictions.read_text(encoding="utf-8"))
        predictions = payload.get("predictions", payload) if isinstance(payload, dict) else payload
        result = evaluate_holdout_once(
            database=Database(args.gold_database.resolve()),
            snapshot_sha256=args.snapshot_sha256,
            candidate_bundle_sha256=args.candidate_bundle_sha256,
            split_manifest_sha256=args.split_manifest_sha256,
            predictions=predictions,
            confirm_final_candidate=args.confirm_final_candidate,
            confirm_holdout_one_time_evaluation=args.confirm_holdout_one_time_evaluation,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "init-gold-set-database":
        target = args.database.resolve()
        if not args.confirm_non_production_db:
            raise ValueError("Use --confirm-non-production-db for Gold Set initialization")
        if target == settings.database_path.resolve():
            raise ValueError("Gold Set schema cannot be initialized in the production database")
        if target.exists():
            raise FileExistsError(f"Gold Set database already exists: {target}")
        status = migrate_database(
            target,
            "gold",
            dry_run=False,
            baseline_confirmed=False,
        )
        print(
            json.dumps(
                {
                    "database": str(target),
                    "profile": "gold",
                    "applied": list(status.applied),
                    "phase6_6d1_schema": True,
                }
            )
        )
        return 0
    if args.command == "create-real-gold-set":
        bundle = DomainConfigLoader(semantic_config_dir).load()
        manifest = create_real_gold_set_sample(
            source_db=args.source_db,
            output_dir=args.output_dir,
            gold_set_id=args.gold_set_id,
            bundle=bundle,
            sample_size=args.sample_size,
            random_seed=args.random_seed,
            confirm_read_only_source=args.confirm_read_only_source,
        )
        print(json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "import-gold-set-items":
        target = args.database.resolve()
        if target == settings.database_path.resolve():
            raise ValueError("Gold Set items cannot be imported into the production database")
        repository = GoldSetRepository(Database(target))
        gold_set_id = repository.import_sample(
            items_csv=args.items,
            sampling_manifest=args.sampling_manifest,
            name=args.name,
            annotation_version=args.annotation_version,
            review_mode=args.review_mode,
            minimum_round_interval_days=args.minimum_round_interval_days,
        )
        print(json.dumps({"gold_set_id": gold_set_id, "database": str(target)}))
        return 0
    if args.command in {"validate-gold-set-annotations", "inspect-gold-set-consistency"}:
        repository = GoldSetRepository(Database(args.database.resolve()))
        overview = repository.overview(args.gold_set_id)
        agreement = repository.agreement(args.gold_set_id)
        print(
            json.dumps(
                {"overview": overview, "agreement": asdict(agreement)},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0 if agreement.ready_for_calibration else 1
    if args.command == "export-gold-set-snapshot":
        target = args.database.resolve()
        if target == settings.database_path.resolve():
            raise ValueError("Gold snapshots cannot be exported from the production database")
        result = GoldSetRepository(Database(target)).export_snapshot(
            gold_set_id=args.gold_set_id,
            snapshot_version=args.snapshot_version,
            output=args.output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-semantic-gold-pilot":
        bundle = DomainConfigLoader(semantic_config_dir).load()
        engine = load_semantic_engine_config(semantic_config_dir)
        prompts = SemanticPromptBuilder(semantic_config_dir / "prompts")
        budget = PilotBudget(
            max_papers=args.max_papers,
            max_fast_calls=args.max_fast_calls,
            max_strong_calls=args.max_strong_calls,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            budget_limit=args.budget_limit,
            max_retries=args.max_retries,
            failure_rate_threshold=args.failure_rate_threshold,
        )
        request = PilotRequest(
            gold_set_snapshot=args.gold_set_snapshot,
            gold_set_sha256=args.gold_set_sha256,
            output_database=args.output_db,
            archive_manifest=args.archive_manifest,
            production_database=settings.database_path,
            model_bundle_version=args.model_bundle_version,
            expected_config_sha256=args.config_bundle_sha256,
            run_uid=args.run_uid,
            allow_live_ai=args.allow_live_ai,
            confirmation=args.confirm,
            budget=budget,
            allow_resume=args.resume_pilot_run,
        )
        preflight = validate_pilot_preflight(
            request,
            bundle=bundle,
            engine=engine,
            prompts=prompts,
            api_key=settings.deepseek_api_key,
        )
        adapter = DeepSeekSemanticAdapter(
            DeepSeekClient(
                api_key=settings.deepseek_api_key or "",
                base_url=settings.deepseek_base_url,
            )
        )
        processor = SemanticGoldPilotProcessor(
            bundle=bundle,
            engine=engine,
            prompts=prompts,
            client=adapter,
        )
        stats = ControlledGoldPilot(Database(args.output_db.resolve())).run(
            request, preflight, processor
        )
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
        return 0 if stats.status == "completed" else 1
    if args.command in {"evaluate-calibration-run", "search-decision-policies"}:
        snapshot = load_human_gold_snapshot(args.gold_set_snapshot, args.gold_set_sha256)
        if args.split_manifest.exists():
            split = _load_split_manifest(args.split_manifest)
            if split.snapshot_sha256 != snapshot["snapshot_sha256"]:
                raise ValueError("Split manifest belongs to a different Gold snapshot")
        else:
            split = create_split_manifest(snapshot, random_seed=args.random_seed)
            save_split_manifest(split, args.split_manifest)
        predictions_payload = json.loads(args.predictions.read_text(encoding="utf-8"))
        predictions = (
            predictions_payload["predictions"]
            if isinstance(predictions_payload, dict)
            else predictions_payload
        )
        calibration_database = None
        snapshot_database_id = None
        if args.command == "evaluate-calibration-run":
            if args.database is None:
                raise ValueError("Final holdout evaluation requires --database")
            target = args.database.resolve()
            if target == settings.database_path.resolve():
                raise ValueError("Calibration runs cannot use the production database")
            calibration_database = Database(target)
            GoldSetRepository(calibration_database)  # validates the independent schema
            with calibration_database.connect() as connection:
                row = connection.execute(
                    "SELECT id FROM gold_set_snapshots WHERE snapshot_sha256=?",
                    (snapshot["snapshot_sha256"],),
                ).fetchone()
                if row is None:
                    raise ValueError("Gold snapshot is not registered in this Gold database")
                snapshot_database_id = int(row["id"])
                prior = connection.execute(
                    """SELECT split_manifest_json FROM calibration_runs
                    WHERE gold_set_snapshot_id=? AND run_type='holdout_final'""",
                    (snapshot_database_id,),
                ).fetchall()
            if any(
                json.loads(item["split_manifest_json"]).get("manifest_sha256")
                == split.manifest_sha256
                for item in prior
            ):
                raise ValueError("This Gold snapshot/split holdout has already been evaluated once")
        report = search_decision_policies(
            snapshot,
            predictions,
            split,
            minimum_validation_recall=args.minimum_validation_recall,
            random_seed=args.random_seed,
            evaluate_holdout=args.command == "evaluate-calibration-run",
        )
        save_calibration_report(report, args.output)
        if calibration_database is not None and snapshot_database_id is not None:
            try:
                bundle = DomainConfigLoader(semantic_config_dir).load()
                engine = load_semantic_engine_config(semantic_config_dir)
                prompts = SemanticPromptBuilder(semantic_config_dir / "prompts")
                with calibration_database.connect() as connection:
                    connection.execute(
                        """INSERT INTO calibration_runs
                        (gold_set_snapshot_id,config_bundle_sha256,model_bundle_version,
                         prompt_bundle_sha256,run_type,random_seed,split_manifest_json,
                         attempted_policies_json,selected_policy_json,result_json,created_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            snapshot_database_id,
                            bundle.config_sha256,
                            engine.version,
                            semantic_prompt_bundle_sha256(prompts),
                            "holdout_final",
                            args.random_seed,
                            json.dumps(asdict(split), sort_keys=True),
                            json.dumps(report["attempted_policies"], sort_keys=True),
                            json.dumps(report["selected_policy"], sort_keys=True),
                            json.dumps(report, ensure_ascii=False, sort_keys=True),
                            utcnow(),
                        ),
                    )
            except Exception:
                args.output.unlink(missing_ok=True)
                raise
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "compare-strategy-candidates":
        report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
        print(json.dumps(report["strategy_candidates"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-model-bundle-candidate":
        target = args.database.resolve()
        if target == settings.database_path.resolve():
            raise ValueError("Model bundle candidates require an independent Gold database")
        bundle = DomainConfigLoader(semantic_config_dir).load()
        engine = load_semantic_engine_config(semantic_config_dir)
        prompts = SemanticPromptBuilder(semantic_config_dir / "prompts")
        repository = GoldSetRepository(Database(target))
        candidate = build_model_bundle_candidate(
            database=Database(target),
            production_database=settings.database_path,
            archive_manifest=args.archive_manifest,
            snapshot_path=args.gold_set_snapshot,
            snapshot_sha256=args.gold_set_sha256,
            candidate_version=args.candidate_version,
            bundle=bundle,
            engine=engine,
            prompt_bundle_sha256=semantic_prompt_bundle_sha256(prompts),
            calibration_report=json.loads(args.calibration_report.read_text(encoding="utf-8")),
            agreement_report=repository.agreement(args.gold_set_id),
            agreement_gold_set_id=args.gold_set_id,
            red_team_report=json.loads(args.red_team_report.read_text(encoding="utf-8")),
            budget=json.loads(args.budget_report.read_text(encoding="utf-8")),
            known_limitations=_load_string_list(args.known_limitations),
            git_commit=_current_git_commit(project_root),
            output=args.output,
        )
        print(json.dumps(candidate, ensure_ascii=False, indent=2))
        return 0
    if args.command == "inspect-model-bundle-candidate":
        print(
            json.dumps(
                inspect_model_bundle_candidate(args.input),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "validate-semantic-prompts":
        result = SemanticPromptBuilder(semantic_config_dir / "prompts").validate()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "inspect-semantic-run":
        result = inspect_semantic_run_read_only(args.database.resolve(), args.paper_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "migrate-domain-semantic":
        if getattr(args, "database", None) is None:
            raise ValueError("Semantic migration requires an explicit --database path")
        target = args.database.resolve()
        if not args.backup_confirmed:
            raise ValueError("Semantic migration requires --backup-confirmed")
        if not args.confirm_non_production_db:
            raise ValueError("Semantic migration requires --confirm-non-production-db")
        production = settings.database_path.resolve()
        if target == production:
            _require_recent_semantic_archive(
                target,
                getattr(args, "archive_manifest", None),
            )
        apply_domain_semantic_migration(
            Database(target),
            project_root / "migrations" / "0004_add_domain_semantic_runs.sql",
        )
        print(
            "Phase 4 semantic schema migration applied to the explicitly selected database; "
            "no semantic analysis was run."
        )
        return 0
    if args.command in {
        "evaluate-semantic-judgments",
        "run-semantic-fixtures",
        "compare-shadow-decisions",
    }:
        fixture_path = (args.fixtures or semantic_fixture).resolve()
        evaluator = SemanticEvaluationService()
        fixtures = evaluator.load_fixtures(fixture_path)
        report = evaluator.evaluate(fixtures)
        content = report.model_dump_json(indent=2)
        if args.output:
            _write_safe_json(args.output, fixture_path, content, args.overwrite)
        print(content)
        return 0
    if args.command == "archive-production-database":
        manifest = archive_database(
            source_path=args.source_path or settings.database_path,
            archive_dir=args.archive_dir,
            label=args.label,
            confirm_read_only_source=args.confirm_read_only_source,
            current_git_commit=_current_git_commit(settings.project_root),
            logical_application_version="v0.4.1",
        )
        print(json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "create-gold-set-template":
        manifest = create_gold_set_template(
            source_db=args.source_db,
            output=args.output,
            sample_size=args.sample_size,
            random_seed=args.random_seed,
            stratify=args.stratify,
            include_existing_screening_metadata=args.include_existing_screening_metadata,
            confirm_read_only_source=args.confirm_read_only_source,
            overwrite=args.overwrite,
        )
        print(json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-gold-set":
        gold_set = GoldSetLoader().load(args.input, require_labels=args.require_labels)
        print(
            json.dumps(
                {
                    "valid": True,
                    "rows": len(gold_set.items),
                    "labeled": sum(item.gold_label is not None for item in gold_set.items),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "evaluate-domain-evidence":
        gold_set = GoldSetLoader().load(args.gold_set, require_labels=True)
        report = DomainEvaluator().evaluate(gold_set)
        content = report.model_dump_json(indent=2)
        if args.output:
            unsafe_output = (
                args.output.resolve() == args.gold_set.resolve()
                or args.output.name.casefold() == ".env"
            )
            if unsafe_output:
                raise ValueError("Evaluation output must be a separate non-production file")
            if args.output.suffix.casefold() != ".json":
                raise ValueError("Evaluation output must use a .json extension")
            if args.output.exists() and not args.overwrite:
                raise FileExistsError(f"Evaluation output already exists: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if args.overwrite else "x"
            with args.output.open(mode, encoding="utf-8") as output:
                output.write(content + "\n")
        print(content)
        return 0
    if args.command == "inspect-domain-evidence":
        result = inspect_domain_evidence_read_only(args.database, args.paper_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "migrate-domain-evidence":
        if not args.backup_confirmed:
            raise ValueError("迁移前必须备份目标数据库并使用 --backup-confirmed")
        database = Database(args.database.resolve())
        apply_domain_evidence_migration(
            database,
            settings.project_root / "migrations" / "0003_add_domain_evidence_runs.sql",
        )
        print("Domain Evidence migration已应用到显式指定数据库；未执行任何证据分析。")
        return 0
    database = Database(settings.database_path)
    database.init_schema()
    if args.command == "init-db":
        journal_file = load_journals(settings.journals_path)
        for journal in journal_file.journals:
            database.upsert_journal(journal)
        print(f"数据库已初始化：{settings.database_path}")
        print(f"已载入 {len(journal_file.journals)} 本期刊配置。")
        return 0
    if args.command == "sync":
        unpaywall = UnpaywallClient(settings.unpaywall_email) if settings.unpaywall_email else None
        service = SyncService(database, CrossrefClient(), unpaywall)
        stats = service.sync(
            load_journals(settings.journals_path),
            since=args.since,
            journal_short_name=args.journal,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        if args.journal and not stats["journals_checked"]:
            raise ValueError(f"未找到已启用期刊：{args.journal}")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 1 if stats["records_failed"] else 0
    if args.command == "screen":
        service = _screening_service(settings, database, no_ai=args.no_ai)
        result = service.run(
            limit=args.limit, no_ai=args.no_ai, allow_strong=not args.no_strong_model
        )
        print(json.dumps(vars(result), ensure_ascii=False, indent=2))
        return 0
    if args.command == "export":
        filters = {
            "journal": args.journal,
            "topic": args.topic,
            "date_from": args.from_date,
            "date_to": args.to_date,
            "review_status": args.status,
            "has_abstract": True if args.has_abstract else None,
            "task_id": args.task_id,
        }
        if args.mode == "mentor":
            content = (
                export_mentor_markdown(
                    database, include_human_note=args.include_human_note, **filters
                )
                if args.format == "markdown"
                else export_mentor_csv_text(
                    database, include_human_note=args.include_human_note, **filters
                )
            )
        else:
            content = (
                export_personal_markdown(database, **filters)
                if args.format == "markdown"
                else export_personal_csv_text(database, **filters)
            )
        suffix = ".md" if args.format == "markdown" else ".csv"
        stem = (
            "eduecon_mentor_original_abstracts"
            if args.mode == "mentor"
            else "eduecon_personal_research_notes"
        )
        output = args.output or settings.project_root / "exports" / f"{stem}{suffix}"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        if args.task_id is not None:
            mark_export_completed(database, args.task_id)
        print(f"已导出：{output.resolve()}")
        return 0
    if args.command == "translate":
        parameters = TranslationParameters(
            review_status=args.status,
            journal=args.journal,
            limit=args.limit,
            missing_only=args.missing_only,
            overwrite_human=args.overwrite_human,
        )
        if args.dry_run:
            preview = preview_translations(database, parameters)
            print(json.dumps(vars(preview), ensure_ascii=False, indent=2, default=list))
            return 0
        if not args.confirm_ai_cost:
            raise ValueError("正式翻译前必须显式使用 --confirm-ai-cost")
        if args.overwrite_human and not args.confirm_overwrite_human:
            raise ValueError("覆盖人工校订前必须显式使用 --confirm-overwrite-human")
        if not settings.deepseek_api_key:
            raise ValueError("未配置 DEEPSEEK_API_KEY，不能执行AI翻译")
        service = TranslationService(
            database,
            settings.prompts_dir,
            DeepSeekClient(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url),
            settings.deepseek_model_fast,
        )
        print(json.dumps(vars(service.run(parameters)), ensure_ascii=False, indent=2))
        return 0
    if args.command == "migrate-translations":
        if not args.backup_confirmed:
            raise ValueError("迁移前必须备份数据库并使用 --backup-confirmed 明确确认")
        apply_translation_migration(
            database, settings.project_root / "migrations" / "0001_add_paper_translations.sql"
        )
        print("Translation Layer迁移已完成；旧screening和review数据未迁移或改写。")
        return 0
    if args.command == "migrate-workflows":
        if not args.backup_confirmed:
            raise ValueError("迁移前必须备份数据库并使用 --backup-confirmed 明确确认")
        apply_workflow_migration(
            database, settings.project_root / "migrations" / "0002_add_workflow_batches.sql"
        )
        print("Workflow Layer迁移已完成；论文、筛选、翻译和审核数据未复制或改写。")
        return 0
    if args.command == "process-papers":
        return _run_process_papers(args, settings, database)
    if args.command == "lifecycle-check":
        from eduecon_radar.services.deterministic_screening import (
            FRONT_MATTER_PROMPT_VERSION,
        )
        from eduecon_radar.services.local_pipeline import lifecycle_invariant_report
        from eduecon_radar.services.profile_registry import load_profile_registry
        from eduecon_radar.services.profile_store import assessment_drift

        report = lifecycle_invariant_report(database, args.sync_run_id)
        # Judgement drift is a different contract from the lifecycle one: it
        # compares the run snapshot with the canonical profile assessment.
        drift = list(assessment_drift(database))
        report["judgement_drift"] = drift
        report["judgement_drift_count"] = len(drift)
        # Every judgement must be traceable to a prompt version some profile
        # declares. A paper judged by another queue's prompts is a real defect:
        # it happened once, when a CLI continuation had no profile.
        declared = {FRONT_MATTER_PROMPT_VERSION}
        try:
            registry = load_profile_registry(
                settings.profiles_root,
                catalog_path=settings.journal_catalog_path,
                default_profile_id=settings.default_profile,
            )
            declared |= {
                registry.profile(item[0]).screening_prompt_version
                for item in registry.profile_choices()
            }
        except Exception:
            logging.getLogger(__name__).exception(
                "Unable to load profiles for the prompt-version check"
            )
        with database.connect() as connection:
            versions = {
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT prompt_version FROM paper_profile_assessments "
                    "WHERE prompt_version IS NOT NULL"
                ).fetchall()
            }
        unknown = sorted(version for version in versions if version not in declared)
        report["declared_prompt_versions"] = sorted(declared)
        report["unknown_prompt_versions"] = unknown
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report["violation_count"] or drift or unknown else 0
    if args.command == "stats":
        print(json.dumps(dashboard_stats(database), ensure_ascii=False, indent=2))
        return 0
    return 2


def _write_safe_json(output: Path, input_path: Path, content: str, overwrite: bool) -> None:
    output = output.resolve()
    if output == input_path or output.name.casefold() == ".env":
        raise ValueError("Semantic evaluation output must be a separate non-secret file")
    if output.suffix.casefold() != ".json":
        raise ValueError("Semantic evaluation output must use a .json extension")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Semantic evaluation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output.open(mode, encoding="utf-8") as stream:
        stream.write(content + "\n")


def _load_split_manifest(path: Path) -> SplitManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["development"] = tuple(payload["development"])
    payload["validation"] = tuple(payload["validation"])
    payload["holdout"] = tuple(payload["holdout"])
    return SplitManifest(**payload)


def _load_string_list(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [line.strip() for line in raw.splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Known limitations must be a JSON string list or one line per item")
    return value


def _selectable_paper_ids(database: Database, run_id: int) -> tuple[int, ...]:
    """Papers that can still make progress in one retrieval run.

    Delegates to ``local_pipeline.resumable_paper_ids`` so the CLI and the
    workbench button share one definition of "unfinished" instead of keeping
    two selection queries that can drift apart.
    """
    from eduecon_radar.services.local_pipeline import resumable_paper_ids

    return resumable_paper_ids(database, run_id)


def _run_process_papers(
    args: argparse.Namespace, settings: Settings, database: Database
) -> int:
    """Continue the product processing chain for unfinished papers.

    The workbench reaches this chain only through a button that acts on the
    currently selected retrieval run, so an interrupted batch had no
    scriptable, auditable way to resume. This command provides that path and
    reuses the exact same pipeline builder as the UI.
    """
    from eduecon_radar.services.local_pipeline import (
        latest_sync_run,
        lifecycle_invariant_report,
        process_batch,
    )
    from eduecon_radar.services.migrations import overnight_schema_ready

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit 必须为正整数")
    if not overnight_schema_ready(database):
        raise RuntimeError(
            "该数据库尚未应用 0010 overnight pipeline 迁移，无法续跑处理链。"
            "请先按 docs/MIGRATION_AND_SCHEMA_RUNBOOK.md 迁移。"
        )

    if args.sync_run_id is None:
        run = latest_sync_run(database)
        if run is None:
            print(json.dumps({"status": "no_runs"}, ensure_ascii=False, indent=2))
            return 0
        run_id = int(run["id"])
    else:
        run_id = int(args.sync_run_id)
        with database.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sync_runs WHERE id=?", (run_id,)
            ).fetchone()
        if exists is None:
            raise ValueError(f"检索批次不存在：{run_id}")

    paper_ids = _selectable_paper_ids(database, run_id)
    if args.limit is not None:
        paper_ids = paper_ids[: args.limit]

    if not paper_ids:
        print(
            json.dumps(
                {"status": "nothing_to_do", "sync_run_id": run_id, "papers": 0},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not args.yes:
        # Dry run: no provider call, no AI call, no write.
        print(
            json.dumps(
                {
                    "status": "preview",
                    "sync_run_id": run_id,
                    "papers": len(paper_ids),
                    "paper_ids": list(paper_ids),
                    "note": (
                        "未执行处理。加 --yes 才会发起网络与 AI 请求并写入数据库。"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not settings.deepseek_api_key:
        raise ValueError(
            "未配置 DEEPSEEK_API_KEY：摘要补全不需要 AI，但标题翻译与初筛需要。"
            "请在 .env 配置后重试。"
        )

    from eduecon_radar.services.local_pipeline import run_profile_id
    from eduecon_radar.services.pipeline_builder import build_local_pipeline_services
    from eduecon_radar.services.profile_registry import load_profile_registry

    # Continue the run under *its own* research profile: the screening prompts
    # and prompt version are profile content, so judging a behavioural run with
    # another profile's prompts would silently change what "relevant" means.
    profile_id = run_profile_id(database, run_id)
    profile = None
    try:
        registry = load_profile_registry(
            settings.profiles_root,
            catalog_path=settings.journal_catalog_path,
            default_profile_id=settings.default_profile,
        )
        profile = registry.profile(profile_id)
    except Exception:
        logging.getLogger(__name__).exception(
            "Unable to load research profile %s", profile_id
        )

    services = build_local_pipeline_services(database, settings, profile=profile)
    if services.enrichment is None:
        raise ValueError("无法初始化摘要补全服务，已中止")

    result = process_batch(database, run_id, paper_ids, services, profile_id=profile_id)

    # Report the post-run state from the database rather than only the batch
    # return value, so an operator can see what actually persisted.
    state_counts: dict[str, int] = {}
    with database.connect() as connection:
        rows = connection.execute(
            f"""SELECT status, COUNT(*) n FROM paper_processing_status
            WHERE sync_run_id=? AND paper_id IN ({",".join("?" * len(paper_ids))})
            GROUP BY status""",
            (run_id, *paper_ids),
        ).fetchall()
    for row in rows:
        state_counts[str(row["status"])] = int(row["n"])

    payload = {
        "status": "processed",
        "sync_run_id": run_id,
        "requested": len(paper_ids),
        "completed": len(result.get("completed") or ()),
        "needs_abstract": len(result.get("needs_abstract") or ()),
        "needs_retry": len(result.get("needs_retry") or ()),
        "ai_calls": int(result.get("ai_calls") or 0),
        "input_tokens": int(result.get("input_tokens") or 0),
        "output_tokens": int(result.get("output_tokens") or 0),
        "status_after": state_counts,
        # Surface lifecycle violations immediately after a run instead of
        # leaving stranded papers to be discovered by a later audit.
        "lifecycle_violations": lifecycle_invariant_report(database, run_id)[
            "violation_count"
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _load_manual_jel_entries(path: Path | None) -> dict[int, list[str]]:
    """Read a manual JEL entry table: ``paper_id: "D91 D03"`` or a JSON object."""
    if path is None:
        return {}
    payload = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        raw = json.loads(payload)
        return {int(key): list(value) for key, value in raw.items()}
    entries: dict[int, list[str]] = {}
    for line in payload.splitlines():
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        paper_id, _, codes = text.partition(":")
        entries[int(paper_id.strip())] = re.split(r"[,;\s]+", codes.strip())
    return entries


def _run_jel_command(args: argparse.Namespace, project_root: Path) -> int:
    """``enrich-jel`` / ``set-jel-codes``.

    JEL codes are a paper fact, so these commands write to ``papers`` and the
    observation ledger only.  They never touch an assessment, a review, or a
    profile: a code that arrives later changes what a profile *could* use, not
    what a profile already decided.
    """
    target = _project_path(project_root, args.database)
    database = Database(target)
    settings = load_jel_source_settings()
    cache_dir = (
        _project_path(project_root, args.cache_dir)
        if args.cache_dir
        else project_root / "data" / "cache" / "jel"
    )

    if args.command == "set-jel-codes":
        codes = [item for item in re.split(r"[,;\s]+", args.codes) if item]
        manual = {int(args.paper_id): codes}
        providers = build_providers(
            settings=settings,
            manual_entries=manual,
            manual_note=args.note or "manual entry via set-jel-codes",
            manual_source_url=args.source_url,
            cache_dir=cache_dir,
            provider_names=["manual"],
        )
        service = JelEnrichmentService(database, providers)
        papers = service.load_papers(paper_ids=[int(args.paper_id)], only_missing=False)
        if not papers:
            print(f"错误：找不到论文 {args.paper_id}", file=sys.stderr)
            return 2
        stats = service.enrich(papers)
        print(
            json.dumps(
                {
                    "command": "set-jel-codes",
                    "paper_id": int(args.paper_id),
                    "codes": normalize_jel_codes(codes),
                    "stats": stats.as_dict(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        report = audit_jel_enrichment(target)
        print(format_jel_report(report))
        return 0

    # -- enrich-jel --------------------------------------------------------- #
    if args.enable_repec:
        settings = replace(
            settings,
            repec_ideas=replace(settings.repec_ideas, enabled=True),
        )
    manual_entries = _load_manual_jel_entries(
        _project_path(project_root, args.manual_file) if args.manual_file else None
    )
    providers = build_providers(
        settings=settings,
        manual_entries=manual_entries,
        manual_note="manual entry file",
        cache_dir=cache_dir,
        provider_names=args.provider,
    )
    enabled = [provider.name for provider in providers if provider.enabled]
    service = JelEnrichmentService(database, providers)
    papers = service.load_papers(
        paper_ids=args.paper_id,
        only_missing=not args.all_papers,
        limit=args.limit,
    )
    plan = {
        "command": "enrich-jel",
        "mode": "apply" if args.apply else "dry_run",
        "database": str(target),
        "providers_declared": [provider.name for provider in providers],
        "providers_enabled": enabled,
        "cache_dir": str(cache_dir),
        "papers_selected": len(papers),
        "paper_ids": [paper.paper_id for paper in papers[:40]],
        "codes_searched": list(settings.codes) if "repec_ideas" in enabled else [],
    }
    if not args.apply:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("\n（预演：未写入任何数据。确认后加 --apply）")
        return 0
    stats = service.enrich(papers)
    payload = {**plan, "stats": stats.as_dict()}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print()
    print(format_jel_report(audit_jel_enrichment(target)))
    return 0


def _run_profile_quality_command(args: argparse.Namespace, project_root: Path) -> int:
    """``profile-quality-sample`` / ``profile-quality-compare``.

    Both commands are read-only against the database.  They exist so a human can
    check a profile's boundary; they never write a label, tune a threshold, or
    edit a prompt.
    """
    target = _project_path(project_root, args.database) if getattr(
        args, "database", None
    ) else None
    if args.command == "profile-quality-intake":
        sheet_path = _project_path(project_root, args.sheet)
        if not sheet_path.is_file():
            print(f"错误：找不到标注表 {sheet_path}", file=sys.stderr)
            return 2
        intake = intake_sheet(
            sheet_path.read_text(encoding="utf-8-sig"),
            require_blind=not args.allow_unblinded,
        )
        if args.json:
            print(json.dumps(intake.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(format_intake(intake))
        return 0 if intake.usable else 1
    if args.command == "profile-quality-basis":
        basis = measure_screening_basis(target, args.profile)
        if args.json:
            print(json.dumps(basis.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(format_screening_basis(basis))
        return 0
    if args.command == "profile-quality-sample":
        profiles = args.profile or None
        assessments = load_assessments(target)
        if profiles:
            wanted = set(profiles)
            assessments = [item for item in assessments if item.profile_id in wanted]
        if not assessments:
            print("错误：所选范围内没有已存判断，无法抽样", file=sys.stderr)
            return 2
        output = _project_path(project_root, args.output)
        if args.total:
            plan = design_sample(
                assessments,
                total=args.total,
                boundary_share=args.boundary_share,
                min_per_profile=args.min_per_profile,
                include_ids=args.paper_id,
                seed=args.seed,
            )
            rows = plan.rows
            stratum_of = plan.stratum_of
            design = {
                "kind": "stratified_representative_plus_boundary",
                "total_requested": args.total,
                "boundary_share": args.boundary_share,
                "min_per_profile": args.min_per_profile,
                "per_profile": plan.per_profile,
                "per_profile_stratum": plan.per_profile_stratum,
                "seats": plan.seats,
                "representative_capacity": plan.capacity,
                "metric_slice": (
                    "precision/recall are computed on the representative stratum only"
                ),
                "deviation": (
                    "a per-profile floor is reserved before the proportional draw, so "
                    "the largest queue is slightly under-weighted in exchange for every "
                    "queue having enough rows to be measured"
                ),
            }
        else:
            rows = sample_assessments(
                assessments,
                per_decision=args.per_decision or 8,
                per_basis=args.per_basis,
                include_ids=args.paper_id,
                only_ids=args.only_paper_id,
                seed=args.seed,
            )
            stratum_of = {key: "representative" for key in (
                (item.profile_id, item.paper_id) for item in rows
            )}
            design = {
                "kind": "balanced_per_decision",
                "per_decision": args.per_decision or 8,
                "per_basis": args.per_basis,
                "metric_slice": (
                    "balanced draws are NOT a probability sample of the library; "
                    "precision/recall from them do not generalise"
                ),
            }
        manifest = write_sheet(
            output,
            rows,
            blind=args.blind,
            include_abstract=not args.no_abstract,
            stratum_of=stratum_of,
            design=design,
        )
        if args.json:
            print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"标注表已导出：{output}")
            print(f"  行数            : {manifest['rows']}")
            print(f"  抽样设计        : {design['kind']}")
            print(f"  profile 分布    : {manifest['profiles']}")
            print(f"  每 profile 行数 : {design.get('per_profile', '(n/a)')}")
            if design.get("per_profile_stratum"):
                print(f"  分层结构        : {design['per_profile_stratum']}")
            print(f"  自动判断分布    : {manifest['decisions']}")
            print(f"  筛选依据分布    : {manifest['bases']}")
            print(f"  盲标注          : {manifest['blind']}")
            print(f"  允许的人工判断  : {', '.join(manifest['human_decisions_allowed'])}")
            print(f"  同名 sidecar    : {output.name}.manifest.json")
            print(f"  指标口径        : {design['metric_slice']}")
            print("\n请在 human_decision 列填写判断（relevant / irrelevant / uncertain / skip），")
            print("可在 human_note 列写下理由，然后运行 profile-quality-compare。")
        return 0

    sheet_path = _project_path(project_root, args.sheet)
    if not sheet_path.is_file():
        print(f"错误：找不到标注表 {sheet_path}", file=sys.stderr)
        return 2
    # A cross-profile sheet carries profile_id on every row, so the comparison
    # resolves each row against its own queue rather than forcing one profile.
    assessments = load_assessments(target)
    report = compare_sheet(
        sheet_path.read_text(encoding="utf-8-sig"),
        assessments,
        profile_id="",
        sampling_mode=args.sampling_mode,
    )
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_comparison(report))
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.load(Path.cwd())
    configure_logging(settings.log_level, [settings.deepseek_api_key or ""])
    try:
        raise SystemExit(run(args, settings))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
