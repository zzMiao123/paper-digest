"""Crossref client for fetching newly indexed works."""

from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode
from urllib.request import Request

from .arxiv_client import Paper
from .config import FeedConfig
from .network import fetch_bytes_with_retry

CROSSREF_API_URL = "https://api.crossref.org/works"

TOP_JOURNAL_PRIORITY = {
    # Tier 0
    "nature": 0,
    "science": 0,

    # Tier 1: Nature big journals / Science family / PNAS
    "nature materials": 1,
    "nature electronics": 1,
    "nature nanotechnology": 1,
    "nature machine intelligence": 1,
    "nature biomedical engineering": 1,
    "nature communications": 1,
    "science robotics": 1,
    "science advances": 1,
    "proceedings of the national academy of sciences": 1,
    "pnas": 1,

    # Tier 2: Advanced family
    "advanced materials": 2,
    "advanced functional materials": 2,
    "advanced science": 2,
    "advanced intelligent systems": 2,
    "advanced materials technologies": 2,
    "advanced healthcare materials": 2,
    "advanced sensor research": 2,

    # Tier 3: robotics / haptics venues
    "ieee robotics and automation letters": 3,
    "robotics and automation letters": 3,
    "ieee transactions on robotics": 3,
    "transactions on robotics": 3,
    "ieee transactions on haptics": 3,
    "transactions on haptics": 3,
}

BLOCKED_JOURNALS = {
    "scientific reports",
}

# 手动维护。先用字符串，避免 IF 每年更新导致误读。
# 你可以之后按 UIUC Web of Science / JCR 更新成最新值。
JOURNAL_IMPACT_FACTOR = {
    "nature": "high",
    "science": "high",
    "nature materials": "high",
    "nature electronics": "high",
    "nature nanotechnology": "high",
    "nature machine intelligence": "high",
    "nature biomedical engineering": "high",
    "nature communications": "high",
    "science robotics": "high",
    "science advances": "high",
    "proceedings of the national academy of sciences": "high",
    "pnas": "high",
    "advanced materials": "high",
    "advanced functional materials": "high",
    "advanced science": "high",
    "advanced intelligent systems": "check JCR",
    "advanced materials technologies": "check JCR",
    "advanced healthcare materials": "check JCR",
    "advanced sensor research": "check JCR",
    "ieee robotics and automation letters": "robotics venue",
    "ieee transactions on robotics": "robotics venue",
    "ieee transactions on haptics": "haptics venue",
}

def _extract_journal_name(item: dict[str, object]) -> str:
    values = item.get("container-title", [])
    if isinstance(values, list) and values:
        value = values[0]
        if isinstance(value, str):
            return _clean_text(value)
    return ""


def _journal_key(journal: str) -> str:
    return " ".join(journal.lower().replace("&", "and").split())


def _journal_priority(journal: str) -> int | None:
    key = _journal_key(journal)

    if key in BLOCKED_JOURNALS:
        return None

    if key in TOP_JOURNAL_PRIORITY:
        return TOP_JOURNAL_PRIORITY[key]

    # 允许 Nature 小 NC / Communications 系列，但排除 Scientific Reports
    if key.startswith("nature ") and key != "scientific reports":
        return 1

    if key.startswith("npj "):
        return 2

    if key.startswith("communications "):
        return 2

    return None


def _journal_if(journal: str) -> str:
    key = _journal_key(journal)
    return JOURNAL_IMPACT_FACTOR.get(key, "check JCR")


def _corresponding_author_guess(authors: list[str]) -> list[str]:
    # Crossref 通常不给真正的 corresponding author。
    # 这里用最后一位作者作为 engineering/materials 论文的粗略 proxy。
    if not authors:
        return []
    return [authors[-1]]

class CrossrefClientError(RuntimeError):
    """Raised when Crossref fetches or parsing fail."""
class CrossrefSkipItem(RuntimeError):
    """Raised when a Crossref item should be skipped without failing the run."""

def fetch_latest_crossref_papers(
    feed: FeedConfig,
    *,
    now: datetime,
    lookback_hours: int,
    request_delay_seconds: float,
    request_timeout_seconds: int = 60,
    retry_attempts: int = 4,
    retry_backoff_seconds: float = 10.0,
    contact_email: str | None = None,
) -> list[Paper]:
    """Fetch recent Crossref works for a configured feed."""

    from_index_date = (now - timedelta(hours=lookback_hours)).date().isoformat()
    filters = [f"from-index-date:{from_index_date}"]
    for work_type in feed.types:
        filters.append(f"type:{work_type}")

    params = {
        "rows": feed.max_results,
        "sort": "indexed",
        "order": "desc",
        "filter": ",".join(filters),
        "query.bibliographic": " ".join(feed.queries),
    }
    if contact_email:
        params["mailto"] = contact_email

    request = Request(
        f"{CROSSREF_API_URL}?{urlencode(params)}",
        headers={
            "User-Agent": _build_user_agent(contact_email),
            "Accept": "application/json",
        },
    )

    payload = fetch_bytes_with_retry(
        request,
        timeout_seconds=request_timeout_seconds,
        request_delay_seconds=request_delay_seconds,
        retry_attempts=retry_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        error_factory=CrossrefClientError,
        operation_description=f"failed to fetch works for feed {feed.name!r}",
    )
    papers = parse_crossref_response(payload)
    return papers


def parse_crossref_response(payload: bytes) -> list[Paper]:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CrossrefClientError("received malformed JSON from Crossref") from exc

    message = raw.get("message")
    if not isinstance(message, dict):
        raise CrossrefClientError("Crossref response did not include a message object")

    items = message.get("items", [])
    if not isinstance(items, list):
        raise CrossrefClientError("Crossref response items payload is invalid")

    papers: list[Paper] = []

    for item in items:
        if not isinstance(item, dict):
            continue
    
        try:
            papers.append(parse_crossref_item(item))
        except CrossrefSkipItem:
            continue
    
    return papers


def parse_crossref_item(item: dict[str, object]) -> Paper:
    doi = _required_string(item.get("DOI"), "Crossref DOI is missing")
    abstract_url = _string(item.get("URL")) or f"https://doi.org/{doi}"
    indexed_at = _parse_crossref_datetime(item.get("indexed"))

    title_values = item.get("title", [])
    title = (
        _clean_text(title_values[0])
        if isinstance(title_values, list) and title_values
        else doi
    )

    subjects = item.get("subject")
    categories = (
        [subject for subject in subjects if isinstance(subject, str)]
        if isinstance(subjects, list)
        else []
    )

    journal = _extract_journal_name(item)
    priority = _journal_priority(journal)
    
    if priority is None:
        raise CrossrefSkipItem(f"journal not in whitelist: {journal}")

    
    authors = _extract_authors(item.get("author"))

    return Paper(
        title=title,
        summary=_extract_abstract(item.get("abstract")),
        authors=authors,
        categories=categories,
        paper_id=f"https://doi.org/{doi}",
        abstract_url=abstract_url,
        pdf_url=None,
        published_at=indexed_at,
        updated_at=indexed_at,
        source="crossref",
        date_label="Indexed",
        doi=doi,
        source_urls={
            "crossref": abstract_url,
            "doi": f"https://doi.org/{doi}",
        },
        tags=[
            f"Journal: {journal}",
            f"IF: {_journal_if(journal)}",
            f"JournalPriority: {priority}",
        ],
        topics=_corresponding_author_guess(authors),
    )


def _extract_authors(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    authors: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        literal = _string(item.get("literal"))
        if literal:
            authors.append(literal)
            continue

        given = _string(item.get("given"))
        family = _string(item.get("family"))
        full_name = " ".join(part for part in [given, family] if part)
        if full_name:
            authors.append(full_name)
    return authors


def _extract_abstract(value: object) -> str:
    if not isinstance(value, str):
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return _clean_text(html.unescape(without_tags))


def _parse_crossref_datetime(value: object) -> datetime:
    if not isinstance(value, dict):
        raise CrossrefClientError("Crossref item is missing indexed date information")

    date_time = value.get("date-time")
    if isinstance(date_time, str):
        return _parse_iso_datetime(date_time)

    timestamp = value.get("timestamp")
    if isinstance(timestamp, int):
        return datetime.fromtimestamp(timestamp / 1000, tz=UTC)

    raise CrossrefClientError("Crossref indexed date format is invalid")


def _parse_iso_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CrossrefClientError(f"invalid Crossref datetime: {value!r}") from exc


def _required_string(value: object, message: str) -> str:
    result = _string(value)
    if not result:
        raise CrossrefClientError(message)
    return result


def _string(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _build_user_agent(contact_email: str | None) -> str:
    base = "paper-digest/0.1 (https://github.com/X-PG13/paper-digest)"
    if contact_email:
        return f"{base}; mailto:{contact_email}"
    return base
