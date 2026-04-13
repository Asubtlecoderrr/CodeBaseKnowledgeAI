# import time
# import google.generativeai as genai
# from config import EMBED_MODEL, GOOGLE_API_KEY

# genai.configure(api_key=GOOGLE_API_KEY)

# class Embedder:
#     def embed(self, chunks):
#         texts = [c.enriched for c in chunks]
#         all_vectors = []
#         for i, text in enumerate(texts):
#             vector = self._embed_with_retry(text, task_type="retrieval_document")
#             all_vectors.append(vector)
#             time.sleep(1.2)
#             if (i + 1) % 50 == 0:
#                 print(f"  [embedder] {i+1}/{len(texts)} chunks embedded...")
#         return all_vectors

#     def embed_query(self, query):
#         time.sleep(1.2)  # add this line
#         return self._embed_with_retry(query, task_type="retrieval_query")

#     def _embed_with_retry(self, text, task_type, retries=3):
#         for attempt in range(retries):
#             try:
#                 result = genai.embed_content(
#                     model=EMBED_MODEL,
#                     content=text,
#                     task_type=task_type,
#                 )
#                 return result["embedding"]
#             except Exception as e:
#                 if "429" in str(e) and attempt < retries - 1:
#                     wait = 60
#                     print(f"  [embedder] rate limit hit, waiting {wait}s...")
#                     time.sleep(wait)
#                 else:
#                     raise

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        print("[embedder] loading local model...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[embedder] model ready")

    def embed(self, chunks):
        texts = [c.enriched for c in chunks]
        vectors = self._model.encode(texts, show_progress_bar=True, batch_size=32)
        return vectors.tolist()

    def embed_query(self, query):
        vector = self._model.encode([query])
        return vector[0].tolist()