"""Product access guard.

The local app is single-user and must only answer loopback requests.  This
lives in its own module so the shipped Streamlit entry point does not have to
import the historical personal-mode service module (which pulls in the retired
task pipeline, the pilot runtime and the domain research stack) just to ask
whether a request came from localhost.
"""

from __future__ import annotations


def local_request_allowed(host: str | None) -> bool:
    """True when the request host is the loopback interface."""
    hostname = (host or "").strip().casefold()
    if hostname.startswith("localhost:"):
        hostname = "localhost"
    elif hostname.startswith("127.0.0.1:"):
        hostname = "127.0.0.1"
    return hostname in {"localhost", "127.0.0.1"}
