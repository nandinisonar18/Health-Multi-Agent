# orchestrator.py
import os
import asyncio
import json
from utils import load_env, logger
from data_miner import DataMiner
from summarizer import MedicalSummarizer
from decision_maker import DecisionMaker

load_env()

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "5"))
MODEL = os.getenv("LANGCHAIN_MODEL", "gpt-3.5-turbo")

async def process_article(sem, article, summarizer: MedicalSummarizer, decider: DecisionMaker):
    async with sem:
        result = {"id": article.get("id"), "title": article.get("title"), "url": article.get("url"), "source": article.get("source")}
        try:
            summary_obj = await summarizer.summarize(article.get("title",""), article.get("url",""), article.get("content",""))
            result["summary"] = summary_obj
        except Exception as e:
            result["error_summarize"] = str(e)
            return result
        try:
            key_facts = summary_obj.get("key_facts") if isinstance(summary_obj, dict) else []
            classification = await decider.classify(article.get("title",""), summary_obj.get("summary", "") if isinstance(summary_obj, dict) else str(summary_obj), key_facts)
            result["classification"] = classification
        except Exception as e:
            result["error_classify"] = str(e)
        return result
    

async def main(limit=10):
    miner = DataMiner(rss_feeds=os.getenv("RSS_FEEDS").split(",") if os.getenv("RSS_FEEDS") else None,
                      newsapi_key=os.getenv("NEWSAPI_KEY"))
    summarizer = MedicalSummarizer(model_name=MODEL)
    decider = DecisionMaker(model_name=MODEL)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    try:
        articles = await miner.get_latest(limit)
        logger.info(f"Fetched {len(articles)} articles")
        tasks = [asyncio.create_task(process_article(sem, a, summarizer, decider)) for a in articles]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        # save
        out_path = "results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved results to {out_path}")
        return results
    finally:
        await miner.close()

if __name__ == "__main__":
    asyncio.run(main(limit=10))
