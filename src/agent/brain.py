import os

from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_openai import ChatOpenAI

from api.xai_client import get_xai_api_key

load_dotenv()


class MedAgent:
    def __init__(self):
        api_key = get_xai_api_key()
        if not api_key:
            raise ValueError(
                "XAI_API_KEY (or xai_api_key) is required for the MedLabel agent."
            )
        self.llm = ChatOpenAI(
            model=os.getenv("XAI_MODEL_TEXT", "grok-3"),
            api_key=api_key,
            base_url=os.getenv("XAI_API_BASE", "https://api.x.ai/v1"),
            temperature=0,
        )

        # Real tools wired on thaneesh-interaction_check branch (merge to main first).
        self.tools = [
            Tool(
                name="vector_search",
                func=lambda x: "vector_search not wired on this branch yet.",
                description="Search FDA label chunks for dosage, warnings, and ingredients.",
            ),
            Tool(
                name="interaction_check",
                func=lambda x: "interaction_check lives on thaneesh-interaction_check PR.",
                description="Check drug-drug interactions when user asks about combining medicines.",
            ),
        ]

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
    print("Agent initialized (xAI).")
