# """
# qa/claude.py — Gemini-based answer generation (free tier).
# """

# from __future__ import annotations
# import google.generativeai as genai
# from config import GOOGLE_API_KEY

# genai.configure(api_key=GOOGLE_API_KEY)


# class ClaudeQA:
#     def __init__(self):
#         self._model = genai.GenerativeModel(
#             model_name="models/gemini-2.0-flash-lite",
#             system_instruction="""You are an expert software engineer assistant with deep knowledge of this codebase.
# When answering questions:
# 1. Base your answer ONLY on the code context provided.
# 2. Always cite the specific file and function/class name.
# 3. If the context doesn't contain enough information, say so explicitly.
# 4. Include short code snippets to support your answer."""
#         )

#     def ask_stream(self, system: str, messages: list[dict]):
#         prompt = messages[0]["content"]
#         response = self._model.generate_content(prompt, stream=True)
#         for chunk in response:
#             if chunk.text:
#                 yield chunk.text

#     def ask(self, system: str, messages: list[dict]) -> str:
#         prompt = messages[0]["content"]
#         response = self._model.generate_content(prompt)
#         return response.text

#     def answer(self, system: str, messages: list[dict]) -> str:
#         parts = []
#         for token in self.ask_stream(system, messages):
#             parts.append(token)
#         return "".join(parts)


import requests
import json

class ClaudeQA:
    def __init__(self):
        pass

    def ask_stream(self, system: str, messages: list[dict]):
        prompt = messages[0]["content"]
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": True},
            stream=True
        )
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("response"):
                    yield data["response"]

    def ask(self, system: str, messages: list[dict]) -> str:
        return "".join(self.ask_stream(system, messages))

    def answer(self, system: str, messages: list[dict]) -> str:
        return self.ask(system, messages)