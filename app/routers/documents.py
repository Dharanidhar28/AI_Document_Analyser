import os

from fastapi import APIRouter, UploadFile, File, HTTPException
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError
from requests.exceptions import RequestException

from app.services.pdf_parser import extract_text_from_pdf
from app.services.rag_pipeline import create_vector_store, retrieve_context, generate_answer

router = APIRouter()

vector_db = None


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    global vector_db
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was provided.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs("uploads", exist_ok=True)
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join("uploads", safe_filename)

    try:
        contents = file.file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        text = extract_text_from_pdf(file_path)
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="This PDF does not contain extractable text.",
            )

        vector_db = create_vector_store(text)
    except HTTPException:
        raise
    except InferenceTimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="The embedding service timed out while processing the document.",
        ) from exc
    except HfHubHTTPError as exc:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", 502) or 502
        raise HTTPException(
            status_code=status_code,
            detail=f"Hugging Face embedding error: {exc}",
        ) from exc
    except RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="The embedding service could not be reached while indexing the document.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process the uploaded PDF: {exc}",
        ) from exc

    return {"message": "Document processed successfully"}


@router.post("/ask")
def ask_question(question: str):
    if vector_db is None:
        raise HTTPException(status_code=400, detail="Please upload a document first")

    contexts = retrieve_context(vector_db, question)
    try:
        answer = generate_answer(question, contexts)
    except InferenceTimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="The Hugging Face inference model is currently unavailable or timed out.",
        ) from exc
    except HfHubHTTPError as exc:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", 502) or 502
        error_text = str(exc)
        raise HTTPException(
            status_code=status_code,
            detail=f"Hugging Face inference error: {error_text}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate an answer from the document: {exc}",
        ) from exc

    return {"answer": answer}
