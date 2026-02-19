from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

from app.core.config import get_env_str


def make_llm() -> ChatOpenAI:
    model = get_env_str("OPENAI_MODEL", "gpt-5-mini")
    return ChatOpenAI(model=model, temperature=0)


def make_embeddings() -> OpenAIEmbeddings:
    model = get_env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def make_openai_client() -> OpenAI:
    return OpenAI()
