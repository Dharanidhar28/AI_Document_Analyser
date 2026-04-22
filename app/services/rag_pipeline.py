from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from huggingface_hub import InferenceClient
import os
import re
from dotenv import load_dotenv
from requests.exceptions import RequestException


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "deepset/roberta-base-squad2")
HF_PROVIDER = os.getenv("HF_PROVIDER", "hf-inference")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = InferenceClient(model=HF_MODEL_ID, provider=HF_PROVIDER, token=HF_TOKEN)


def _split_documents(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])

    # Limit to the first 15 chunks to avoid API timeouts on large documents
    if len(docs) > 15:
        docs = docs[:15]

    return docs


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _build_local_store(docs):
    return {
        "mode": "local",
        "docs": [doc.page_content for doc in docs if doc.page_content.strip()],
    }


def _score_text(query_tokens: set[str], text: str) -> tuple[int, int]:
    text_tokens = _tokenize(text)
    overlap = len(query_tokens & text_tokens)
    return overlap, len(text)


def _fallback_answer(question: str, contexts: list[str]) -> str:
    if not contexts:
        return "I could not find relevant information in the uploaded document."

    query_tokens = _tokenize(question)
    ranked_contexts = sorted(
        contexts,
        key=lambda context: _score_text(query_tokens, context),
        reverse=True,
    )
    best_context = ranked_contexts[0].strip()
    if not best_context:
        return "I could not find relevant information in the uploaded document."

    sentences = re.split(r"(?<=[.!?])\s+", best_context)
    ranked_sentences = sorted(
        [sentence.strip() for sentence in sentences if sentence.strip()],
        key=lambda sentence: _score_text(query_tokens, sentence),
        reverse=True,
    )

    if ranked_sentences and _score_text(query_tokens, ranked_sentences[0])[0] > 0:
        return ranked_sentences[0]

    return best_context[:300].strip()


def create_vector_store(text):
    docs = _split_documents(text)

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_url="https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    )

    try:
        vector_store = FAISS.from_documents(docs, embeddings)
    except (RequestException, ValueError):
        return _build_local_store(docs)

    return vector_store


def retrieve_context(vector_store, question, top_k=5):

    if vector_store is None:
        return []

    if isinstance(vector_store, dict) and vector_store.get("mode") == "local":
        docs = vector_store.get("docs", [])
        query_tokens = _tokenize(question)
        ranked_docs = sorted(
            docs,
            key=lambda doc: _score_text(query_tokens, doc),
            reverse=True,
        )
        return [doc for doc in ranked_docs[:top_k] if doc.strip()]

    docs = vector_store.similarity_search(question, k=top_k)

    if not docs:
        return []

    return [doc.page_content for doc in docs if doc.page_content.strip()]


def generate_answer(question: str, contexts: list[str]) -> str:
    if not contexts:
        return "I could not find relevant information in the uploaded document."

    best_answer = ""
    best_score = 0.0
    had_remote_error = False

    for context in contexts:
        try:
            response = client.question_answering(
                question=question,
                context=context,
                model=HF_MODEL_ID,
                handle_impossible_answer=True,
                top_k=3,
                max_answer_len=120,
            )
        except Exception:
            had_remote_error = True
            continue

        candidates = response if isinstance(response, list) else [response]
        for candidate in candidates:
            answer = (candidate.answer or "").strip()
            score = float(candidate.score or 0.0)
            if answer and score > best_score:
                best_answer = answer
                best_score = score

    if not best_answer or best_score < 0.08:
        if had_remote_error:
            return _fallback_answer(question, contexts)
        return "I could not find relevant information in the uploaded document."

    return best_answer
