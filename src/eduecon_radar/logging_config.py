from __future__ import annotations

import logging


class SecretRedactionFilter(logging.Filter):
    def __init__(self, secrets: list[str] | None = None) -> None:
        super().__init__()
        self.secrets = [value for value in (secrets or []) if value]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for secret in self.secrets:
            message = message.replace(secret, "***")
        record.msg = message
        record.args = ()
        return True


def configure_logging(level: str = "INFO", secrets: list[str] | None = None) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(SecretRedactionFilter(secrets))
    logging.basicConfig(level=level, handlers=[handler], force=True)
