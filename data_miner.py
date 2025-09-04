# data_miner.py
import os
import httpx
import asyncio
import uuid
from bs4 import BeautifulSoup
import feedparser
from typing import List, Dict, Optional
from utils import async_retry, logger

class DataMiner:
    def __init__(self, newsapi_key: Optional[str] = None, rss_feeds: Optional[List[str]] = None, max_articles: int = 30):
        self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY")
        self.rss_feeds = rss_feeds or os.getenv("RSS_FEEDS", "").split(",")
        self.max_articles = max_articles
        self.client = httpx.AsyncClient(timeout=15.0)

    async def close(self):
        await self.client.aclose()

    @async_retry(max_attempts=3)
    async def fetch_newsapi(self) -> List[Dict]:
        if not self.newsapi_key:
            return []
        url = "https://newsapi.org/v2/top-headlines"
        params = {"category": "health", "language": "en", "pageSize": min(self.max_articles, 100)}
        headers = {"Authorization": self.newsapi_key}
        r = await self.client.get(url, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()
        items = []
        for a in data.get("articles", []):
            items.append({
                "id": str(uuid.uuid4()),
                "title": a.get("title"),
                "url": a.get("url"),
                "source": a.get("source", {}).get("name"),
                "published": a.get("publishedAt"),
                "content": a.get("content") or ""
            })
        return items

    @async_retry(max_attempts=3)
    async def fetch_rss(self) -> List[Dict]:
        out = []
        for feed in filter(None, self.rss_feeds):
            try:
                r = await self.client.get(feed)
                r.raise_for_status()
                text = r.text
                # feedparser is sync; run in thread
                parsed = await asyncio.to_thread(feedparser.parse, text)
                for entry in parsed.entries[: self.max_articles]:
                    out.append({
                        "id": str(uuid.uuid4()),
                        "title": entry.get("title"),
                        "url": entry.get("link"),
                        "source": entry.get("source", {}).get("title") if entry.get("source") else None,
                        "published": entry.get("published"),
                        "content": entry.get("summary", "")  # summary may be HTML
                    })
            except Exception as e:
                logger.warning(f"RSS fetch failed for {feed}: {e}")
        return out

    @async_retry(max_attempts=3)
    async def fetch_article_text(self, url: str) -> str:
        try:
            r = await self.client.get(url, timeout=20.0)
            r.raise_for_status()
            html = r.text
            soup = BeautifulSoup(html, "html.parser")
            # basic heuristics: prefer <article>, else longest block of <p>
            article_tag = soup.find("article")
            if article_tag:
                text = " ".join(p.get_text(strip=True) for p in article_tag.find_all("p"))
                if len(text) > 200:
                    return text
            # fallback: longest paragraph cluster
            p_texts = [p.get_text(strip=True) for p in soup.find_all("p")]
            if not p_texts:
                return ""
            # join largest contiguous group
            p_texts_sorted = sorted(p_texts, key=lambda s: len(s), reverse=True)
            return " ".join(p_texts_sorted[:12])
        except Exception as e:
            logger.warning(f"Failed to fetch article text for {url}: {e}")
            return ""

    async def get_latest(self, limit: int = 20) -> List[Dict]:
        results = []
        # prefer NewsAPI then RSS and de-dup by url/title
        newsapi_items = await self.fetch_newsapi()
        rss_items = await self.fetch_rss()
        combined = newsapi_items + rss_items
        seen = set()
        for item in combined:
            if len(results) >= limit:
                break
            key = (item.get("url") or "") + (item.get("title") or "")
            if key in seen:
                continue
            seen.add(key)
            # fetch full text if we only have summary
            if not item.get("content") or len(item.get("content")) < 300:
                item["content"] = await self.fetch_article_text(item.get("url") or "") or item.get("content", "")
            results.append(item)
        return results
