# summarizer.py
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import asyncio
from utils import async_retry, logger



class MedicalSummarizer:
    def __init__(self, model_name: str = None, temperature: float = 0.0):
        model_name = model_name or os.getenv("LANGCHAIN_MODEL", "gpt-3.5-turbo")
        self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"),model="llama-3.1-8b-instant")
        self.prompt = PromptTemplate(
            input_variables=["title", "url", "article_text"],
            template=(
                "You are a careful medical-news summarizer. Given the article text, produce:\n"
                "1) A concise summary (3-5 sentences).\n"
                "2) 3 bullet key facts (each 1 line).\n"
                "3) Short note: uncertainty/limitations (1-2 lines).\n"
                "4) A one-line consumer-facing recommendation: either 'Consult professional' or 'Information only' (do NOT give medical advice). \n\n"
                "Output must be a JSON object exactly with keys: summary, key_facts (list), uncertainty, recommendation.\n\n"
                "Title: {title}\nURL: {url}\n\nArticle:\n{article_text}\n\nRespond JSON only."
            )
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    @async_retry(max_attempts=2)
    async def summarize(self, title: str, url: str, article_text: str) -> dict:
        try:
            # use arun for async run
            raw = await self.chain.arun(title=title, url=url, article_text=article_text)
            # parse JSON
            import json, re
            # find JSON substring
            m = re.search(r'(\{.*\})', raw, re.S)
            if not m:
                logger.warning("Summarizer didn't return JSON - returning raw text")
                return {"raw": raw}
            parsed = json.loads(m.group(1))
            return parsed
        except Exception as e:
            logger.exception(f"Summarizer error for {url}: {e}")
            return {"error": str(e)}
