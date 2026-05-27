from __future__ import annotations

import os

from dotenv import load_dotenv

from agent.tools import adverse_events, build_agent_tools, interaction_check, vector_search
from api.xai_client import get_xai_api_key

load_dotenv()


def _make_gemini_agent(scanned_drug: str):
    """LangChain ReAct agent (Gemini only — xAI does not support agent ``stop`` param)."""
    from langchain.agents import AgentType, initialize_agent
    from langchain_google_genai import ChatGoogleGenerativeAI

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is required for LangChain agent mode.")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL_TEXT", "gemini-1.5-flash"),
        google_api_key=gemini_key,
        temperature=0,
    )
    tools = build_agent_tools(scanned_drug=scanned_drug)
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )


def _is_interaction_question(question: str) -> bool:
    q = question.lower()
    return any(
        w in q
        for w in ("with", "together", "combine", "interaction", "mix", "take")
    )


def _is_adverse_events_question(question: str) -> bool:
    q = question.lower()
    return any(
        w in q
        for w in ("side effect", "adverse", "reaction", "symptom", "faers")
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
        self._use_xai = bool(get_xai_api_key())
        self.agent = None
        if not self._use_xai:
            self.agent = _make_gemini_agent(self.scanned_drug)

    def _run_tools_direct(self, user_query: str) -> str:
        """Route to tools without LangChain agent (required for xAI API)."""
        if _is_interaction_question(user_query) and self.scanned_drug:
            return interaction_check(user_query, scanned_drug=self.scanned_drug)
        if _is_adverse_events_question(user_query):
            return adverse_events(self.scanned_drug or user_query)
        return vector_search(user_query)

    def run_query(self, user_query: str) -> str:
        """Answer using tools; xAI uses direct routing, Gemini uses LangChain agent."""
        if self._use_xai:
            return self._run_tools_direct(user_query)

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
    print(agent.run_query("Can I take this with ibuprofen?"))
