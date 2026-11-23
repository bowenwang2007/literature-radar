import httpx

from eduecon_radar.services.resilient_http import ResilientHttpClient


def test_429_retry_after_has_injected_sleep():
    calls = 0
    sleeps = []

    def handler(request):
        nonlocal calls
        calls += 1
        return (
            httpx.Response(429, headers={"Retry-After": "3"}, request=request)
            if calls == 1
            else httpx.Response(200, json={"ok": True}, request=request)
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resilient = ResilientHttpClient(client, sleeper=sleeps.append, rate_limit_per_second=0)
    assert resilient.get_json("https://example.org/data")["ok"]
    assert sleeps == [3.0]


def test_5xx_backoff_and_no_secret_in_error():
    sleeps = []
    transport = httpx.MockTransport(lambda request: httpx.Response(503, request=request))
    client = ResilientHttpClient(
        httpx.Client(transport=transport),
        max_retries=1,
        sleeper=sleeps.append,
        jitter=lambda: 0,
        rate_limit_per_second=0,
    )
    try:
        client.get("https://example.org", headers={"Authorization": "Bearer secret"})
    except httpx.HTTPStatusError as exc:
        assert "secret" not in str(exc)
    assert sleeps == [1]
