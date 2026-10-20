"""Bounded text search for the agent; search snippets are untrusted external input."""
import logging

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class WebSearchError(Exception):
    pass


MAX_RESULT_BYTES = 6000


def truncate_utf8(value: str, max_bytes: int) -> str:
    return value.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def search_web(query: str, max_results: int = 5, timelimit: str | None = None) -> list[dict[str, str]]:
    settings = get_settings()
    # Retry via DDGS' auto backend selection, dropping the date filter on the second attempt.
    # DDGS randomizes eligible engines, so a fresh instance may use a different upstream.
    limits = [timelimit, None] if timelimit else [None, None]
    last_error: Exception | None = None
    last_attempt_had_no_results = False

    for attempt, search_limit in enumerate(limits, start=1):
        try:
            results = DDGS(timeout=settings.web_search_timeout_seconds, proxy=settings.web_search_proxy).text(
                query=query,
                region="wt-wt",
                safesearch="moderate",
                timelimit=search_limit,
                max_results=max_results,
                backend="auto",
            )
            if results:
                return _clean_results(results, max_results)
            last_attempt_had_no_results = True
            last_error = None
        except DDGSException as exc:
            if str(exc).strip().lower() == "no results found.":
                last_attempt_had_no_results = True
                last_error = None
                logger.info("Web search attempt %d returned no results", attempt)
            else:
                last_attempt_had_no_results = False
                last_error = exc
                logger.warning("Web search attempt %d failed (%s)", attempt, type(exc).__name__)
        except Exception as exc:
            last_attempt_had_no_results = False
            last_error = exc
            logger.warning("Web search attempt %d failed (%s)", attempt, type(exc).__name__)

    if last_attempt_had_no_results:
        return []
    logger.error("Web search failed after retry (%s)", type(last_error).__name__ if last_error else "unknown error")
    raise WebSearchError("搜索上游连续请求失败，请检查 FastAPI 服务器的出网或 WEB_SEARCH_PROXY 代理配置，或稍后重试。") from last_error


def _clean_results(results, max_results: int) -> list[dict[str, str]]:
    cleaned = []
    for item in results or []:
        title = truncate_utf8(str(item.get("title") or "").strip(), 600)
        url = truncate_utf8(str(item.get("href") or item.get("url") or "").strip(), 1200)
        snippet = str(item.get("body") or item.get("snippet") or "").strip()
        # Do not pass malformed or non-web URLs back as cited sources.
        if not title or not url.startswith(("https://", "http://")):
            continue
        remaining = MAX_RESULT_BYTES - len(str(cleaned).encode("utf-8")) - len(title.encode()) - len(url.encode()) - 80
        if remaining < 0:
            continue
        candidate = {"title": title, "url": url, "snippet": truncate_utf8(snippet, min(900, remaining))}
        if len(str([*cleaned, candidate]).encode("utf-8")) <= MAX_RESULT_BYTES:
            cleaned.append(candidate)
        if len(cleaned) >= max_results:
            break
    return cleaned
