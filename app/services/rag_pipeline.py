from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "deepset/roberta-base-squad2")
HF_PROVIDER = os.getenv("HF_PROVIDER", "hf-inference")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = InferenceClient(model=HF_MODEL_ID, provider=HF_PROVIDER, token=HF_TOKEN)


def create_vector_store(text):

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store


def retrieve_context(vector_store, question, top_k=5):

    if vector_store is None:
        return []

    docs = vector_store.similarity_search(question, k=top_k)

    if not docs:
        return []

    return [doc.page_content for doc in docs if doc.page_content.strip()]


def generate_answer(question: str, contexts: list[str]) -> str:
    if not contexts:
        return "I could not find relevant information in the uploaded document."

    best_answer = ""
    best_score = 0.0

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
            continue

        candidates = response if isinstance(response, list) else [response]
        for candidate in candidates:
            answer = (candidate.answer or "").strip()
            score = float(candidate.score or 0.0)
            if answer and score > best_score:
                best_answer = answer
                best_score = score

    if not best_answer or best_score < 0.08:
        return "I could not find relevant information in the uploaded document."

    return best_answer
