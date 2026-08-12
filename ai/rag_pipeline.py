from ai.retriever import SurveillanceRetriever
from ai.prompts import build_prompt
from ai.llm_interface import GeminiLLM


class RAGPipeline:

    def __init__(self):

        self.retriever = SurveillanceRetriever()

        self.llm = GeminiLLM()

    # Ask Question

    def answer(self, question):

        events = self.retriever.search(
            question,
            top_k=5
        )

        prompt = build_prompt(
            question,
            events
        )

        answer = self.llm.generate_answer(
            prompt
        )

        return answer, events