from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import os

from app.routers import documents


app = FastAPI()

# Mount static files only if directory exists
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Load templates only if directory exists
if os.path.isdir("templates"):
    templates = Jinja2Templates(directory="templates")
else:
    templates = None

app.include_router(documents.router)


@app.get("/health")
def health_check():
    """Health check endpoint for deployment monitoring"""
    return {"status": "ok"}


@app.get("/")
def home(request: Request):
    if templates is None:
        return {"message": "AI Document Analyser is running. Template directory not found."}
    return templates.TemplateResponse("index.html", {"request": request})
