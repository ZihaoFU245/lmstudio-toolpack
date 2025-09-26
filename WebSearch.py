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
MAX_DETAIL_CHARS = 1200
MAX_BLOCKS = 12
MAX_LINKS_IN_DETAIL = 5
TRUNCATION_NOTICE = "The response is too long that is truncated, further reading please use visit_url tool for {{ url_target }}"
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
    referer: Optional[str] = None,
    collect_link_details: bool = True,
) -> dict:
    timeout = ClientTimeout(total=timeout_sec)
    created_session = False
    if session is None:
        session = aiohttp.ClientSession(headers=HTML_HEADERS, timeout=timeout)
        created_session = True
    try:
        request_headers = {"Referer": referer} if referer else None
        async with session.get(url, allow_redirects=True, headers=request_headers) as resp:
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
    if collect_link_details:
        payload["anchors"] = _collect_anchor_samples(soup, base_url=final_url)
    return payload

async def _visit_many_async(urls: List[str], max_chars: int = 10_000, timeout_sec: int = 20, concurrency: int = 8) -> List[dict]:
    sem = asyncio.Semaphore(concurrency)
    timeout = ClientTimeout(total=timeout_sec)
    async with aiohttp.ClientSession(headers=HTML_HEADERS, timeout=timeout) as session:
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
        host = _host_of(u)
        if _is_ad_domain(host):
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

def _collect_anchor_samples(soup: BeautifulSoup, base_url: str, limit: int = 12) -> List[Tuple[str, str]]:
    anchors: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    base_url = base_url or ""
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
        host = _host_of(u)
        if _is_ad_domain(host):
            continue
        path = (u.raw_path or "").lower()
        if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        text = a.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            text = u.host or abs_url
        text = text.replace('"', "'")
        anchors.append((text[:120], abs_url))
        if len(anchors) >= limit:
            break
    return anchors

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
# Formatting helpers
# ------------------------------
def _normalize_whitespace(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())
def _compose_detail(text: Optional[str], anchors: List[Tuple[str, str]], url: str, *, limit: int = MAX_DETAIL_CHARS) -> str:
    body = _normalize_whitespace(text)
    truncated = False
    if limit and len(body) > limit:
        body = body[:limit].rstrip()
        truncated = True
    link_fragments: List[str] = []
    for label, link in anchors[:MAX_LINKS_IN_DETAIL]:
        if not link:
            continue
        label_clean = _normalize_whitespace(label) or link
        label_clean = label_clean[:120]
        link_fragments.append(f'<a href="{link}">{label_clean}</a>')
    detail_parts: List[str] = []
    if body:
        detail_parts.append(body)
    if link_fragments:
        detail_parts.append("Links: " + ", ".join(link_fragments))
    detail = " ".join(detail_parts).strip()
    if truncated:
        detail = (detail + " " if detail else "") + TRUNCATION_NOTICE.replace("{{ url_target }}", url)
    return detail or "No readable text found."
def _fmt_detail_block(index: int, title: str, url: str, referer: Optional[str], detail: str) -> str:
    title = (title or url or "").strip()
    url = (url or "").strip()
    referer_text = _normalize_whitespace(referer) if referer else "None"
    lines = [
        f"[{index}] {title}:",
        f"    URL: {url or 'Unknown'}",
        f"    Referer: {referer_text or 'None'}",
        f"    Detail: {detail}",
    ]
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
    q: deque[Tuple[str, int, Optional[str]]] = deque()
    q.append((str(base), 0, None))
    sem = asyncio.Semaphore(concurrency)
    timeout = ClientTimeout(total=timeout_sec)
    async with aiohttp.ClientSession(headers=HTML_HEADERS, timeout=timeout) as session:
        while q and len(results) < max_pages:
            url, depth, referer = q.popleft()
            if depth > max_depth:
                continue
            if url in seen:
                continue
            seen.add(url)
            try:
                async with sem:
                    page = await _visit_website_async(
                        url,
                        timeout_sec=timeout_sec,
                        session=session,
                        with_links=True,
                        referer=referer,
                    )
            except Exception as e:
                page = {"ok": False, "error": str(e), "url": url}
            canonical = None
            if isinstance(page, dict):
                canonical = page.get("final_url") or page.get("url") or url
            if canonical:
                seen.add(str(canonical))
            results.append({**page, "depth": depth, "referer": referer})
            if (
                isinstance(page, dict)
                and page.get("ok")
                and page.get("links")
                and depth < max_depth
            ):
                parent_final = str(canonical or url)
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
                    if link in seen:
                        continue
                    if len(results) + len(q) >= max_pages:
                        break
                    q.append((link, depth + 1, parent_final))
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
    return_mode: str = "detail",
) -> str:
    """DuckDuckGo-powered search that emits detailed, browser-style blocks.
    Output format per entry:
        [n] Title:
            URL: https://example.com
            Referer: DuckDuckGo search: <query>
            Detail: Extracted text plus safe links (truncated notice when needed).
    """
    q = query
    if site:
        q = f"{query} site:{site}"
    searcher = Searcher()
    try:
        results: List[SearchResult] = _run_async(searcher.search_duckduckgo(q, max_results=max_results)) or []
    except Exception:
        results = []
    referer_label = f"DuckDuckGo search: {query}"
    if not results:
        ddg_url = "https://duckduckgo.com/?q=" + quote_plus(q)
        detail = f"No results for: {q}"
        return _fmt_detail_block(1, "No results found", ddg_url, referer_label, detail)
    mode = (return_mode or "detail").strip().lower()
    if mode == "simple" or enrich_pages <= 0:
        blocks = []
        for idx, r in enumerate(results[:MAX_BLOCKS], start=1):
            snippet = r.meta_description or r.snippet or ""
            detail = _compose_detail(snippet, [], r.url)
            blocks.append(_fmt_detail_block(idx, r.title or r.url, r.url, referer_label, detail))
        return "\n\n".join(blocks)

    top_count = max(1, min(enrich_pages, max_results, MAX_BLOCKS))
    top_urls = [r.url for r in results[:top_count]]
    pages = _run_async(_visit_many_async(urls=top_urls, max_chars=15_000, timeout_sec=20, concurrency=6)) or []
    page_by_url: Dict[str, dict] = {}
    for p in pages:
        if isinstance(p, dict):
            key = p.get("final_url") or p.get("url")
            if key and key not in page_by_url:
                page_by_url[key] = p
    blocks: List[str] = []
    for idx, r in enumerate(results[:MAX_BLOCKS], start=1):
        page = None
        for key, data in page_by_url.items():
            if key == r.url or data.get("url") == r.url or data.get("final_url") == r.url:
                page = data
                break
        title = r.title or r.url
        target_url = r.url
        if page and page.get("ok"):
            title = page.get("title") or title
            target_url = page.get("final_url") or page.get("url") or target_url
            text_blob = page.get("text") or page.get("description") or r.meta_description or r.snippet or ""
            anchors = page.get("anchors") or []
            detail = _compose_detail(text_blob, anchors, target_url)
        elif page and not page.get("ok"):
            target_url = page.get("url") or target_url
            detail = _normalize_whitespace(page.get("error")) or "Fetch error"
        else:
            snippet = r.meta_description or r.snippet or ""
            detail = _compose_detail(snippet, [], target_url)
        blocks.append(_fmt_detail_block(idx, title, target_url, referer_label, detail))
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
    return_mode: str = "detail",
) -> str:
    """Breadth-first crawl that mimics a no-login browser session.
    Output format per page:
        [n] Title:
            URL: https://example.com
            Referer: None or parent URL
            Detail: Extracted text and sampled links (truncated notice when needed).
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
        return _fmt_detail_block(1, "No content", url, "None", "No pages visited")
    blocks: List[str] = []
    for idx, page in enumerate(pages[:MAX_BLOCKS], start=1):
        if not isinstance(page, dict):
            blocks.append(_fmt_detail_block(idx, "Fetch failed", url, "None", "Fetch error"))
            continue
        ok = bool(page.get("ok"))
        final_url = str(page.get("final_url") or page.get("url") or url)
        referer = page.get("referer")
        title = page.get("title") or final_url
        if not ok:
            detail = _normalize_whitespace(page.get("error")) or "Fetch error"
            blocks.append(_fmt_detail_block(idx, title, final_url, referer, detail))
            continue
        text_blob = page.get("text") or page.get("description") or ""
        anchors = page.get("anchors") or []
        detail = _compose_detail(text_blob, anchors, final_url)
        depth_marker = page.get("depth")
        if depth_marker not in (None, 0):
            prefix = f"Depth {depth_marker}: "
            detail = prefix + detail if detail else prefix.rstrip()
        blocks.append(_fmt_detail_block(idx, title, final_url, referer, detail))
    return "\n\n".join(blocks)

# ------------------------------
# Main (HTTP server)
# ------------------------------
if __name__ == "__main__":
    from GlobalConfig import GlobalConfig
    import asyncio
    
    if GlobalConfig.transport == "http":
        asyncio.run(mcp.run_http_async(GlobalConfig.port) if GlobalConfig.port else mcp.run_http_async()) 
    else:
        asyncio.run(mcp.run_stdio_async())
