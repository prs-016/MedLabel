from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.tools import build_agent_tools
from api.xai_client import get_xai_api_key

load_dotenv()


def _make_llm():
    """Prefer xAI (user .env) when configured; otherwise Gemini."""
    xai_key = get_xai_api_key()
    if xai_key:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "XAI_API_KEY is set but langchain-openai is not installed. "
                "Run: pip install langchain-openai"
            ) from e
        return ChatOpenAI(
            model=os.getenv("XAI_MODEL_TEXT", "grok-3"),
            api_key=xai_key,
            base_url=os.getenv("XAI_API_BASE", "https://api.x.ai/v1"),
            temperature=0,
        )

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_TEXT", "gemini-1.5-flash"),
            google_api_key=gemini_key,
            temperature=0,
        )

    raise ValueError(
        "No LLM API key found. Set XAI_API_KEY or GEMINI_API_KEY in .env."
    )


class MedAgent:
    def __init__(self, scanned_drug: str | None = None):
        """
        Parameters
        ----------
        scanned_drug:
            Generic/brand name from OCR for the medicine the user scanned
            (used as drug_a in ``interaction_check``).
        """
        self.scanned_drug = scanned_drug or ""
        self.llm = _make_llm()
        self.tools = build_agent_tools(scanned_drug=self.scanned_drug)
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def run_query(self, user_query: str) -> str:
        """Runs the agentic reasoning chain."""
        prefix = ""
        if self.scanned_drug:
            prefix = (
                f"The user scanned a medicine label identified as: {self.scanned_drug}. "
                "Use interaction_check for drug combination questions (pass the other drug only). "
                "Use adverse_events with the scanned drug name when asked about side effects.\n\n"
            )
        return self.agent.run(prefix + user_query)


if __name__ == "__main__":
    agent = MedAgent(scanned_drug="acetaminophen")
    print("Agent initialized.")
