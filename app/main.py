from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import os

from app.routers import documents


app = FastAPI()

# Mount static files only if directory exists
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(documents.router)

# HTML content for the home page
html_content = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Document Analyzer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/style.css?v=2">
</head>

<body>
<div class="app-shell">
    <aside class="sidebar">
        <div class="brand-card">
            <p class="eyebrow">AI Document Analyzer</p>
            <h1>Chat with your document</h1>
            <p class="brand-copy">
                Upload a PDF, then ask questions in a chat-style workspace.
            </p>
        </div>

        <section class="panel upload-panel">
            <label class="upload-label" for="fileInput">Choose PDF</label>
            <input type="file" id="fileInput" accept=".pdf">
            <button id="uploadButton" class="primary-button" onclick="uploadFile()">Upload document</button>
            <p id="uploadStatus" class="status-text">No document uploaded yet.</p>
        </section>

        <section class="panel tips-panel">
            <p class="panel-title">Try asking</p>
            <button class="suggestion-chip" onclick="fillQuestion('What is the email address?')">What is the email address?</button>
            <button class="suggestion-chip" onclick="fillQuestion('What skills are listed?')">What skills are listed?</button>
            <button class="suggestion-chip" onclick="fillQuestion('Summarize this document')">Summarize this document</button>
        </section>
    </aside>

    <main class="chat-layout">
        <section id="chatWindow" class="chat-window">
            <div class="welcome-card">
                <p class="eyebrow">Ready</p>
                <h2>Start a conversation</h2>
                <p>Upload a PDF and ask a question. Your messages and answers will appear here like a chat.</p>
            </div>
        </section>

        <section class="composer-panel">
            <div class="composer">
                <input type="text" id="question" placeholder="Ask something about the uploaded document">
                <button id="askButton" class="primary-button" onclick="askQuestion()">Send</button>
            </div>
            <p id="helperText" class="helper-text">Upload a document first, then start asking questions.</p>
        </section>
    </main>
</div>

<script src="/static/script.js?v=2"></script>
</body>
</html>"""


@app.get("/")
def home(request: Request):
    return HTMLResponse(content=html_content, status_code=200)
