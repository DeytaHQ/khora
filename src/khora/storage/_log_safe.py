from urllib.parse import urlparse


def _safe_url_for_log(url: str) -> str:
    """Render URL with credentials redacted, suitable for log messages."""
    parsed = urlparse(url)
    if not (parsed.username or parsed.password):
        return url
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return parsed._replace(netloc=f"<redacted>@{host}").geturl()
