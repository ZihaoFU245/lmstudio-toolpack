"""
Web Search (single-tool, enriched)
----------------------------------
A FastMCP server that exposes **one** tool: `web_search`.

Problem this solves:
- Many models only call `web_search`. Plain DDG snippets are too shallow.
- This tool **fetches and summarizes** the top result pages automatically,
  returning rich, self-contained blocks (title, URL, key points, excerpt).

Features:
- DuckDuckGo HTML/Lite search (no JS)
- Concurrent page fetch for top-N results
- Heuristic extraction: <title>, meta description, first paragraphs, headings
- Lightweight extractive summarization keyed to the query (no LLM needed)
- Strict timeouts; graceful fallbacks

Dependencies:
- fastmcp, aiohttp, beautifulsoup4, yarl

Run:
  python web-tools.py
"""

from fastmcp import FastMCP

import asyncio
import threading
from dataclasses import dataclass, asdict
from typing import List, Optional, Set, Tuple, Dict
from urllib.parse import quote_plus, urljoin
from collections import deque
import re
import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup
from yarl import URL

# ------------------------------
# Constants / Headers
# ------------------------------
DUCKDUCKGO_HTML = "https://duckduckgo.com/html/"  # no-JS results page
DUCKDUCKGO_LITE = "https://lite.duckduckgo.com/lite/"  # ultra-simple HTML fallback

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0 Safari/537.36"
)

BASE_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
HTML_HEADERS = {
    **BASE_HEADERS,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Output caps (to avoid overly long strings)
MAX_SUMMARY_POINTS = 8
MAX_EXCERPT_CHARS = 1000
MAX_SNIPPET_CHARS = 320
MAX_BLOCKS = 12

# Common ad/tracker domains (suffix match)
AD_BLOCKLIST = {
    "doubleclick.net",
    "googlesyndication.com",
    "googleadservices.com",
    "adservice.google.com",
    "adnxs.com",
    "adsrvr.org",
    "adform.net",
    "advertising.com",
    "taboola.com",
    "outbrain.com",
    "criteo.com",
    "quantserve.com",
    "scorecardresearch.com",
    "moatads.com",
    "rubiconproject.com",
    "smartadserver.com",
    "pubmatic.com",
    "openx.net",
    "bluekai.com",
    "facebook.net",
    "fbcdn.net",
    "twitter.com",
}

# Obvious non-HTML/binary extensions to skip when crawling
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".ico", ".bmp", ".tif", ".tiff",
    ".pdf", ".zip", ".gz", ".tgz", ".rar", ".7z", ".tar",
    ".exe", ".msi", ".dmg", ".pkg", ".apk",
    ".mp4", ".mp3", ".mov", ".avi", ".wmv", ".mkv", ".m4a",
    ".css", ".js", ".json",
}


# ------------------------------
# Data structures
# ------------------------------
@dataclass
class SearchResult:
    rank: int
    title: str
    url: str
    snippet: str
    meta_description: Optional[str] = None


# ------------------------------
# Async searcher
# ------------------------------
class Searcher:
    def __init__(
        self,
        *,
        timeout_sec: int = 15,
        concurrency: int = 8,
        verify_ssl: bool = True,
    ):
        self.timeout = ClientTimeout(total=timeout_sec)
        self.sem = asyncio.Semaphore(concurrency)
        self.verify_ssl = verify_ssl

    async def _fetch(self, session: aiohttp.ClientSession, url: str, **kwargs) -> str:
        async with self.sem:
            async with session.get(
                url, timeout=self.timeout, ssl=self.verify_ssl, **kwargs
            ) as resp:
                resp.raise_for_status()
                return await resp.text()

    async def search_duckduckgo(
        self, query: str, max_results: int = 10, *, country: str = "us-en"
    ) -> List[SearchResult]:
        """
        Perform a search on DuckDuckGo HTML results (no-JS), with lite fallback.
        """
        params = {
            "q": query,
            "kl": country,  # region/lang
        }
        async with aiohttp.ClientSession(headers=HTML_HEADERS) as session:
            html = await self._fetch(session, DUCKDUCKGO_HTML, params=params)
            results = self._parse_duckduckgo_results(html)
            if not results:
                lite_html = await self._fetch(session, DUCKDUCKGO_LITE, params=params)
                results = self._parse_duckduckgo_lite_results(lite_html)
            return results[:max_results]

    def _parse_duckduckgo_results(self, html: str) -> List[SearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        out: List[SearchResult] = []
        for i, res in enumerate(soup.select("div.result"), start=1):
            a = res.select_one("a.result__a")
            if not a:
                continue
            raw_title = a.get_text() or ""
            title = re.sub(r"\s+", " ", raw_title).strip()
            href = a.get("href")
            url = self._clean_url(href)
            snippet_el = res.select_one("a.result__snippet") or res.select_one("div.result__snippet")
            raw_snippet = snippet_el.get_text() if snippet_el else ""
            snippet = re.sub(r"\s+", " ", raw_snippet).strip()
            out.append(SearchResult(rank=i, title=title, url=url, snippet=snippet))
        return out

    def _parse_duckduckgo_lite_results(self, html: str) -> List[SearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        out: List[SearchResult] = []
        seen: Set[str] = set()
        anchors = soup.select("a[href]")
        rank = 1
        for a in anchors:
            text = re.sub(r"\s+", " ", (a.get_text() or "").strip())
            if not text or len(text) < 2:
                continue
            href = a.get("href") or ""
            url = self._clean_url(href)
            try:
                u = URL(url) if url else None
            except Exception:
                u = None
            if not u or not u.is_absolute() or u.scheme not in ("http", "https"):
                continue
            if u.host and "duckduckgo.com" in u.host:
                continue
            if url in seen:
                continue
            seen.add(url)
            # Try to find a nearby snippet (heuristic)
            snippet = ""
            parent = a.parent
            sib = parent.find_next_sibling() if parent else None
            if sib:
                snippet = sib.get_text(" ", strip=True)[:300]
            if not snippet and parent:
                snippet = (parent.get_text(" ", strip=True) or "")[:300]
            out.append(SearchResult(rank=rank, title=text, url=url, snippet=snippet))
            rank += 1
        return out

    def _clean_url(self, href: Optional[str]) -> str:
        if not href:
            return ""
        try:
            u = URL(href)
            # Handle DDG redirect links sometimes found in HTML results
            if u.host and "duckduckgo.com" in u.host and u.path.startswith("/l/"):
                uddg = u.query.get("uddg")
                if uddg:
                    try:
                        return str(URL(uddg))
                    except Exception:
                        return uddg
            return str(u)
        except Exception:
            return href


# ------------------------------
# Page fetching & extraction
# ------------------------------
async def _visit_website_async(
    url: str,
    max_chars: int = 10_000,
    timeout_sec: int = 20,
    session: Optional[aiohttp.ClientSession] = None,
    with_links: bool = False,
) -> dict:
    timeout = ClientTimeout(total=timeout_sec)
    created_session = False
    if session is None:
        session = aiohttp.ClientSession(headers=BASE_HEADERS, timeout=timeout)
        created_session = True
    try:
        async with session.get(url, allow_redirects=True) as resp:
            status = resp.status
            final_url = str(resp.url)
            html = await resp.text()
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url}
    finally:
        if created_session:
            await session.close()

    soup = BeautifulSoup(html, "html.parser")
    _clean_soup(soup)
    title, desc = _extract_title_desc(soup)
    text, headings = _extract_text_and_headings(soup)
    payload = {
        "ok": True,
        "status": status,
        "url": url,
        "final_url": final_url,
        "title": title,
        "description": desc,
        "text": text[:max_chars],
        "headings": headings[:15],
    }
    if with_links:
        links = _extract_links(soup, base_url=final_url)
        payload["links"] = links
    return payload


async def _visit_many_async(urls: List[str], max_chars: int = 10_000, timeout_sec: int = 20, concurrency: int = 8) -> List[dict]:
    sem = asyncio.Semaphore(concurrency)
    timeout = ClientTimeout(total=timeout_sec)

    async with aiohttp.ClientSession(headers=BASE_HEADERS, timeout=timeout) as session:
        async def one(u: str):
            async with sem:
                return await _visit_website_async(u, max_chars=max_chars, timeout_sec=timeout_sec, session=session)

        tasks = [one(u) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    out: List[dict] = []
    for r in results:
        if isinstance(r, dict):
            out.append(r)
        else:
            out.append({"ok": False, "error": str(r)})
    return out


def _extract_title_desc(soup: BeautifulSoup, *, title_limit: Optional[int] = None, desc_limit: int = 500) -> Tuple[str, str]:
    raw_title = soup.title.get_text() if soup.title else ""
    title = re.sub(r"\s+", " ", raw_title).strip()
    if title_limit:
        title = title[:title_limit]

    desc = ""
    for sel in [
        'meta[name="description"]',
        'meta[name="Description"]',
        'meta[property="og:description"]',
        'meta[name="twitter:description"]',
    ]:
        el = soup.select_one(sel)
        if el and el.get("content"):
            desc = el.get("content").strip()
            break

    if not desc:
        # Fallback: first paragraph
        p = soup.find("p")
        if p:
            desc = re.sub(r"\s+", " ", p.get_text(strip=True))
    return title, desc[:desc_limit]

def _clean_soup(soup: BeautifulSoup) -> None:
    # Remove noise elements, common ad containers and offscreen elements
    for tag in soup(["script", "style", "noscript", "template", "iframe", "svg", "canvas", "form"]):
        tag.decompose()
    for tag in soup(["nav", "footer", "aside", "header"]):
        tag.decompose()
    for el in soup.select('[aria-hidden="true"], [role="banner"], [role="complementary"]'):
        el.decompose()
    # Remove common ad containers by id/class heuristics
    for el in soup.select('[id*="ad" i], [class*="ad" i], [id*="sponsor" i], [class*="sponsor" i], [class*="promo" i]'):
        try:
            el.decompose()
        except Exception:
            pass

def _extract_text_and_headings(soup: BeautifulSoup) -> Tuple[str, List[str]]:
    headings: List[str] = []
    for h in soup.find_all(["h1", "h2", "h3"]):
        txt = re.sub(r"\s+", " ", h.get_text(strip=True))
        if txt:
            headings.append(txt)
    raw = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", raw).strip()
    return text, headings

def _extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    links: List[str] = []
    seen: Set[str] = set()
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        abs_url = urljoin(base_url, href)
        try:
            u = URL(abs_url)
        except Exception:
            continue
        if not u.is_absolute() or u.scheme not in ("http", "https"):
            continue
        path = u.raw_path or ""
        for ext in SKIP_EXTENSIONS:
            if path.lower().endswith(ext):
                abs_url = ""
                break
        if not abs_url:
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        links.append(abs_url)
    return links


# ------------------------------
# Utility: Run async from sync
# ------------------------------
def _run_async(coro):
    """Run an async coroutine from sync context safely (supports nested event loops)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        container: Dict[str, object] = {}
        def runner():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                container["result"] = new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join()
        return container.get("result")


# ------------------------------
# Simple query-keyed summarizer
# ------------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def _keywords(query: str) -> Set[str]:
    return {w.lower() for w in _WORD_RE.findall(query) if len(w) > 2}

def _score_sentence(sent: str, keys: Set[str]) -> float:
    words = [w.lower() for w in _WORD_RE.findall(sent)]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in keys)
    # Coverage + density; boost shorter sentences a bit
    return hits / len(words) + 0.25 * (1.0 if hits > 0 else 0.0)

def _pick_key_points(text: str, query: str, max_points: int = 3) -> List[str]:
    keys = _keywords(query)
    if not text:
        return []
    # Prefer first 2000 chars to keep it topical
    region = text[:2000]
    sents = _SENT_SPLIT_RE.split(region)
    scored = [(s.strip(), _score_sentence(s, keys)) for s in sents if len(s.strip()) > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    points: List[str] = []
    seen: Set[str] = set()
    for s, sc in scored:
        if sc <= 0:
            continue
        # avoid near-duplicates
        sig = " ".join(_WORD_RE.findall(s.lower())[:10])
        if sig in seen:
            continue
        seen.add(sig)
        points.append(s if len(s) <= 240 else (s[:237].rstrip() + "..."))
        if len(points) >= max_points:
            break
    return points

def _first_excerpt(text: str, *, limit: int = MAX_EXCERPT_CHARS) -> str:
    if not text:
        return ""
    t = text.strip()
    return t[:limit] if len(t) > limit else t


# ------------------------------
# Formatting helpers
# ------------------------------
def _fmt_line_block(index: int, title: str, url: str, snippet: str) -> str:
    title = (title or url or "").strip()
    url = (url or "").strip()
    snippet = re.sub(r"\s+", " ", (snippet or "").strip())
    if len(snippet) > MAX_SNIPPET_CHARS:
        snippet = snippet[:MAX_SNIPPET_CHARS - 3].rstrip() + "..."
    return f"[{index}] {title}\n    {url}\n    {snippet}"

def _fmt_rich_block(index: int, title: str, url: str, points: List[str], excerpt: str) -> str:
    lines = [f"[{index}] {title}".rstrip(), f"    {url}"]
    if points:
        for p in points:
            lines.append(f"    - {p}")
    if excerpt:
        lines.append(f"    Excerpt: {excerpt}")
    return "\n".join(lines)


# ------------------------------
# Crawl helpers (ad-block + BFS)
# ------------------------------
def _host_of(u: URL) -> str:
    try:
        return (u.host or "").lower()
    except Exception:
        return ""

def _is_ad_domain(host: str) -> bool:
    host = (host or "").lower()
    return any(host == d or host.endswith("." + d) for d in AD_BLOCKLIST)

def _same_site(host: str, base_host: str) -> bool:
    host = (host or "").lower()
    base_host = (base_host or "").lower()
    return host == base_host or host.endswith("." + base_host)

async def _crawl_bfs_async(
    start_url: str,
    *,
    max_depth: int = 3,
    max_pages: int = 20,
    same_domain: bool = True,
    timeout_sec: int = 20,
    concurrency: int = 6,
) -> List[dict]:
    try:
        base = URL(start_url)
    except Exception:
        return [{"ok": False, "error": "Invalid start URL", "url": start_url}]
    base_host = _host_of(base)

    seen: Set[str] = set()
    results: List[dict] = []
    q: deque[Tuple[str, int]] = deque()
    q.append((str(base), 0))

    sem = asyncio.Semaphore(concurrency)
    timeout = ClientTimeout(total=timeout_sec)

    async with aiohttp.ClientSession(headers=BASE_HEADERS, timeout=timeout) as session:
        while q and len(results) < max_pages:
            url, depth = q.popleft()
            if url in seen or depth > max_depth:
                continue
            seen.add(url)

            try:
                async with sem:
                    page = await _visit_website_async(url, timeout_sec=timeout_sec, session=session, with_links=True)
            except Exception as e:
                page = {"ok": False, "error": str(e), "url": url}

            results.append({**page, "depth": depth})

            # Enqueue children if OK
            if page.get("ok") and page.get("links") and depth < max_depth:
                final_url = page.get("final_url") or page.get("url") or url
                try:
                    parent = URL(final_url)
                except Exception:
                    parent = URL(url)
                for link in page["links"]:
                    try:
                        u = URL(link)
                    except Exception:
                        continue
                    host = _host_of(u)
                    if _is_ad_domain(host):
                        continue
                    if same_domain and not _same_site(host, base_host):
                        continue
                    # skip if already queued/seen
                    if link in seen:
                        continue
                    # cap queue by max_pages budget
                    if len(seen) + len(q) >= max_pages:
                        break
                    q.append((link, depth + 1))

    return results


# ------------------------------
# MCP server and single tool
# ------------------------------
mcp = FastMCP("Web Search")

@mcp.tool("web_search")
def web_search(
    query: str,
    max_results: int = 10,
    country: str = "us-en",
    site: Optional[str] = None,
    enrich_pages: int = 8,
    return_mode: str = "rich",
) -> str:
    """
    Web search that returns **self-contained**, enriched results.

    Args:
        query: Search query.
        max_results: Number of search results to retrieve (default 8).
        country: DuckDuckGo 'kl' region/language code.
        site: If provided, restricts search via 'site:example.com'.
        enrich_pages: Number of top results to fetch and summarize (default 5).
        return_mode: 'rich' (default) for key points + excerpt, or 'simple' for DDG snippets.

    Output format per entry (rich):
        [n] Title
            https://example.com/...
            - Key point 1
            - Key point 2
            - Key point 3
            Excerpt: Concise extract from the page...

    If a page fetch fails, the entry falls back to the plain snippet.
    """
    # 1) Perform search
    q = query
    if site:
        q = f"{query} site:{site}"

    searcher = Searcher()
    try:
        results: List[SearchResult] = _run_async(searcher.search_duckduckgo(q, max_results=max_results)) or []
    except Exception:
        results = []

    if not results:
        ddg_url = "https://duckduckgo.com/?q=" + quote_plus(q)
        return _fmt_line_block(1, "No results found", ddg_url, f"No results for: {q}")

    # Simple mode: only DDG snippets
    if return_mode.lower() == "simple" or enrich_pages <= 0:
        blocks = []
        for i, r in enumerate(results[:MAX_BLOCKS], start=1):
            snippet = r.meta_description or r.snippet
            blocks.append(_fmt_line_block(i, r.title, r.url, snippet))
        return "\n\n".join(blocks)

    # 2) Enrich: fetch top N pages concurrently
    top_urls = [r.url for r in results[:max(1, min(enrich_pages, max_results, MAX_BLOCKS))]]
    pages = _run_async(_visit_many_async(urls=top_urls, max_chars=10_000, timeout_sec=20, concurrency=6)) or []
    # Map url -> page data
    page_by_url: Dict[str, dict] = {}
    for p in pages:
        if isinstance(p, dict) and (p.get("final_url") or p.get("url")):
            page_by_url[p.get("final_url") or p.get("url")] = p

    # 3) Build rich blocks with key points and excerpt; fallback to snippet
    blocks: List[str] = []
    for i, r in enumerate(results[:MAX_BLOCKS], start=1):
        # Prefer canonical final_url if we fetched it
        page = None
        # Attempt exact key, otherwise any dict where url matches
        for k, v in page_by_url.items():
            if k == r.url or v.get("url") == r.url or v.get("final_url") == r.url:
                page = v
                break

        if page and page.get("ok"):
            title = page.get("title") or r.title or (page.get("final_url") or r.url)
            url = page.get("final_url") or page.get("url") or r.url
            text = page.get("text") or ""
            # Heuristic: start points from headings + scored sentences
            points = []
            # include up to 1-2 headings if relevant
            heads = page.get("headings") or []
            for h in heads[:2]:
                if h and len(points) < MAX_SUMMARY_POINTS:
                    points.append(h if len(h) <= 200 else (h[:197].rstrip() + "..."))
            # add scored sentences keyed to the query
            remaining = max(0, MAX_SUMMARY_POINTS - len(points))
            if remaining > 0:
                points.extend(_pick_key_points(text, query, max_points=remaining))
            excerpt = _first_excerpt(text, limit=MAX_EXCERPT_CHARS)
            blocks.append(_fmt_rich_block(i, title, url, points, excerpt))
        else:
            # Fallback to plain snippet
            snippet = (r.meta_description or r.snippet or "").strip()
            blocks.append(_fmt_line_block(i, r.title, r.url, snippet))

    return "\n\n".join(blocks)


# ------------------------------
# visit_url tool (BFS crawl)
# ------------------------------
@mcp.tool("visit_url")
def visit_url(
    url: str,
    depth: int = 3,
    max_pages: int = 20,
    same_domain: bool = True,
    return_mode: str = "rich",
) -> str:
    """
    Visit a URL and perform a breadth-first crawl up to `depth` (default 3),
    extracting rich text while blocking common ad/tracker domains.

    Args:
        url: Starting URL to visit.
        depth: BFS depth (0 = only the page), default 3.
        max_pages: Cap total pages visited, default 20.
        same_domain: If True, restrict crawl to the same site, default True.
        return_mode: 'rich' (default) yields key points + excerpt per page; 'simple' yields just title + URL.

    Output format per page (rich):
        [n] Title
            https://example.com/path
            - Key point 1
            - Key point 2
            Excerpt: ...
    """
    pages: List[dict] = _run_async(
        _crawl_bfs_async(
            url,
            max_depth=max(0, depth),
            max_pages=max(1, max_pages),
            same_domain=bool(same_domain),
            timeout_sec=20,
            concurrency=6,
        )
    ) or []

    if not pages:
        return _fmt_line_block(1, "No content", url, "No pages visited")

    blocks: List[str] = []
    idx = 1
    for p in pages[:MAX_BLOCKS]:
        if not isinstance(p, dict) or not p.get("ok"):
            err = (p.get("error") if isinstance(p, dict) else "Fetch error") or "Fetch error"
            blocks.append(_fmt_line_block(idx, f"Failed: {url}", url, err))
            idx += 1
            continue
        title = p.get("title") or p.get("final_url") or p.get("url") or url
        final = p.get("final_url") or p.get("url") or url
        text = p.get("text") or ""
        if return_mode.lower() == "simple":
            blocks.append(_fmt_line_block(idx, title, final, p.get("description") or ""))
        else:
            points: List[str] = []
            heads = p.get("headings") or []
            for h in heads[:2]:
                if h and len(points) < MAX_SUMMARY_POINTS:
                    points.append(h if len(h) <= 200 else (h[:197].rstrip() + "..."))
            remaining = max(0, MAX_SUMMARY_POINTS - len(points))
            if remaining > 0:
                # Use title as a weak query key to pick sentences
                points.extend(_pick_key_points(text, title or "", max_points=remaining))
            excerpt = _first_excerpt(text, limit=MAX_EXCERPT_CHARS)
            # prepend depth context in title if deeper than 0
            d = p.get("depth")
            title2 = title if d in (None, 0) else f"(d={d}) {title}"
            blocks.append(_fmt_rich_block(idx, title2, final, points, excerpt))
        idx += 1

    return "\n\n".join(blocks)


# ------------------------------
# Main (HTTP server)
# ------------------------------
if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
