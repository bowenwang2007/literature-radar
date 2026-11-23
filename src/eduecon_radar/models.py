from enum import StrEnum


class AIDecision(StrEnum):
    INCLUDE = "include"
    BORDERLINE = "borderline"
    EXCLUDE = "exclude"
    UNPROCESSED = "unprocessed"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
