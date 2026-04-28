from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import os
from dotenv import load_dotenv

load_dotenv()

class MedAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Week 4 Task: Define tools properly
        self.tools = [
            Tool(
                name="VectorSearch",
                func=lambda x: "Simulated FDA Label Context",
                description="Use this for questions about dosage, warnings, and ingredients."
            ),
            Tool(
                name="InteractionCheck",
                func=lambda x: "Simulated Interaction Result",
                description="Use this for 'Can I take this with X' questions."
            )
        ]
        
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def run_query(self, user_query):
        """
        Runs the agentic reasoning chain.
        """
        # Week 4 Task: Add system prompt and context grounding
        return self.agent.run(user_query)

if __name__ == "__main__":
    agent = MedAgent()
    print("Agent initialized.")
    # print(agent.run_query("What is the dose for Advil?"))
