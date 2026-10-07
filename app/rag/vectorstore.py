import time
from pinecone import Pinecone, ServerlessSpec
from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embeddings-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embeddings-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "text-embeddings-ada-002": 1536,
    "all-minilm-l6-v2": 384,
    "gemini-embedding-001": 3072,
    "models/gemini-embedding-001": 3072,
    "gemini-embedding-2": 3072,
    "models/gemini-embedding-2": 3072,
    "text-embedding-004": 768,
    "models/text-embedding-004": 768,
}


def normalize_embedding_model_name(model_name: str) -> str:
    name = model_name.strip()
    if name.startswith("text-embeddings-"):
        return name.replace("text-embeddings-", "text-embedding-", 1)
    return name


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    
    # If using Gemini and OpenAI key is not provided, use Gemini embedding dimension
    if settings.gemini_api_key and not settings.openai_api_key:
        if "004" in name:
            return 768
        return 3072

    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "gemini-embedding" in normalized:
        return 3072
    if "text-embedding-3-small" in normalized or "text-embeddings-3-small" in normalized:
        return 1536
    if "text-embedding-3-large" in normalized or "text-embeddings-3-large" in normalized:
        return 3072
    if "text-embedding-ada-002" in normalized or "text-embeddings-ada-002" in normalized:
        return 1536
    if "all-minilm" in normalized:
        return 384

    return 1536


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        if settings.openai_api_key:
            from langchain_openai import OpenAIEmbeddings
            model_name = normalize_embedding_model_name(settings.embedding_model)
            _embeddings = OpenAIEmbeddings(
                model=model_name,
                api_key=settings.openai_api_key,
            )
        elif settings.gemini_api_key:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            model_name = "gemini-embedding-001"
            if "gemini-embedding" in settings.embedding_model:
                model_name = settings.embedding_model
            _embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=settings.gemini_api_key,
            )
        else:
            raise RuntimeError("Neither OPENAI_API_KEY nor GEMINI_API_KEY is configured in .env")
    return _embeddings


def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env or environment")

    desired_dimension = get_embedding_dimension()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    names = [x["name"] for x in pc.list_indexes()]

    if settings.pinecone_index_name in names:
        index_info = pc.describe_index(settings.pinecone_index_name)
        current_dimension = getattr(index_info, "dimension", None)
        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        if current_dimension is not None and current_dimension != desired_dimension:
            pc.delete_index(name=settings.pinecone_index_name)
            while settings.pinecone_index_name in [x["name"] for x in pc.list_indexes()]:
                time.sleep(1)

    if settings.pinecone_index_name not in [x["name"] for x in pc.list_indexes()]:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(settings.pinecone_index_name)


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_pinecone import PineconeVectorStore
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})


def add_documents(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)
