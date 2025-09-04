# decision_maker.py
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils import async_retry, safe_json_load, logger

class DecisionMaker:
    def __init__(self, model_name: str = None, temperature: float = 0.0):
        model_name = model_name or os.getenv("LANGCHAIN_MODEL", "gpt-3.5-turbo")
        self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"),model="llama-3.1-8b-instant")       
        self.prompt = PromptTemplate(
            input_variables=["title", "summary", "key_facts"],
            template=(
                "You are a classifier. Given the article title and summary, decide if the content should be labeled "
                "'Actionable Advice' or 'Informative'.\n\n"
                "DEFINITIONS:\n"
                "- Actionable Advice: the text gives explicit step-by-step or clear recommendations meant for immediate action.\n"
                "- Informative: background information, study results, data, or news without direct instructions.\n\n"
                "Output ONLY a JSON object with keys: label (one of 'Actionable Advice' or 'Informative'), confidence (0-1), reason (one sentence).\n\n"
                "Title: {title}\nSummary: {summary}\nKey facts: {key_facts}\n\nRespond JSON only."
            )
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    @async_retry(max_attempts=2)
    async def classify(self, title: str, summary: str, key_facts) -> dict:
        try:
            raw = await self.chain.arun(title=title, summary=summary, key_facts="\n".join(key_facts) if isinstance(key_facts, (list,tuple)) else str(key_facts))
            parsed = safe_json_load(raw)
            if parsed:
                return parsed
            # fallback wrap
            return {"label": "Informative", "confidence": 0.5, "reason": raw[:200]}
        except Exception as e:
            logger.exception(f"Classification error: {e}")
            return {"error": str(e)}
