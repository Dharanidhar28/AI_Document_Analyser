from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from huggingface_hub import InferenceClient
import os
import re
from dotenv import load_dotenv


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "deepset/roberta-base-squad2")
HF_PROVIDER = os.getenv("HF_PROVIDER", "hf-inference")

client = InferenceClient(
    model=HF_MODEL_ID,
    provider=HF_PROVIDER,
    token=HF_TOKEN
)


def create_vector_store(text):

    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store


def split_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def _fallback_answer(question: str, contexts: list[str]) -> str:
    joined_context = "\n".join(contexts)
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", joined_context)
    phone_match = re.search(r"(?:\+?\d[\d\s-]{8,}\d)", joined_context)

    if "email" in question.lower() and email_match:
        return email_match.group(0)

    if "phone" in question.lower() or "mobile" in question.lower() or "contact" in question.lower():
        if phone_match:
            return re.sub(r"\s+", " ", phone_match.group(0)).strip()

    question_terms = {
        term for term in re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
        if term not in {"what", "which", "when", "where", "whose", "about", "from", "into"}
    }

    scored_lines = []
    for context in contexts:
        for raw_line in context.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if len(line) < 20:
                continue

            line_terms = set(re.findall(r"\b[a-zA-Z]{3,}\b", line.lower()))
            overlap = len(question_terms & line_terms)
            bonus = 0

            if "email" in question.lower() and "@" in line:
                bonus += 3
            if "phone" in question.lower() and any(ch.isdigit() for ch in line):
                bonus += 2
            if "skill" in question.lower() and "skill" in line.lower():
                bonus += 2
            if "name" in question.lower() and len(line.split()) <= 6:
                bonus += 1

            score = overlap + bonus
            if score > 0:
                scored_lines.append((score, line))

    if not scored_lines:
        first_context = re.sub(r"\s+", " ", contexts[0]).strip() if contexts else ""
        return first_context[:300] if first_context else "I could not find relevant information in the uploaded document."

    scored_lines.sort(key=lambda item: item[0], reverse=True)
    top_lines = []
    seen = set()
    for _, line in scored_lines:
        if line not in seen:
            top_lines.append(line)
            seen.add(line)
        if len(top_lines) == 2:
            break

    return " ".join(top_lines)


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
        return _fallback_answer(question, contexts)

    return best_answer
