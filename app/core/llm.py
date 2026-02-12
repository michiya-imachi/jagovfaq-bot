import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def make_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    return ChatOpenAI(model=model, temperature=0)


def make_embeddings() -> OpenAIEmbeddings:
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)
