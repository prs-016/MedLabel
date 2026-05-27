from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentType, initialize_agent
import os
from dotenv import load_dotenv

from agent.tools import build_agent_tools

load_dotenv()


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
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

        self.tools = build_agent_tools(scanned_drug=self.scanned_drug)

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def run_query(self, user_query: str) -> str:
        """Runs the agentic reasoning chain."""
        return self.agent.run(user_query)

if __name__ == "__main__":
    agent = MedAgent()
    print("Agent initialized.")
    # print(agent.run_query("What is the dose for Advil?"))
