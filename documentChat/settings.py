CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Use only the following context to answer the question:

{context}

---

Answer the question based on the context above. If appropriate, use bullet points to organize your response:

Question: {question}

Respond clearly, concisely, and directly. If the question cannot be answered with the provided information, clearly state that.
"""