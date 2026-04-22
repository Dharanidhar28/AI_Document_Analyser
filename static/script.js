function getElement(id) {
    return document.getElementById(id);
}

async function readResponsePayload(response) {
    const contentType = response.headers.get("content-type") || "";
    const rawBody = await response.text();

    if (!rawBody) {
        return {};
    }

    if (contentType.includes("application/json")) {
        try {
            return JSON.parse(rawBody);
        } catch {
            return { detail: rawBody };
        }
    }

    return { detail: rawBody };
}

function addMessage(role, text) {
    const chatWindow = getElement("chatWindow");
    if (!chatWindow) {
        return null;
    }

    const welcomeCard = chatWindow.querySelector(".welcome-card");
    if (welcomeCard) {
        welcomeCard.remove();
    }

    const message = document.createElement("div");
    message.className = `message ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    message.appendChild(bubble);
    chatWindow.appendChild(message);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    return bubble;
}

function fillQuestion(question) {
    const questionInput = getElement("question");
    if (!questionInput) {
        return;
    }

    questionInput.value = question;
    questionInput.focus();
}

async function uploadFile() {
	const fileInput = getElement("fileInput");
    const uploadStatus = getElement("uploadStatus");
    const helperText = getElement("helperText");
    const uploadButton = getElement("uploadButton");

    if (!fileInput || !uploadStatus || !helperText || !uploadButton) {
        return;
    }

    const file = fileInput.files[0];

    if (!file) {
        uploadStatus.textContent = "Please choose a PDF file first.";
        return;
    }

	const formData = new FormData();

	formData.append("file", file);

    uploadButton.disabled = true;
    uploadStatus.textContent = "Uploading and indexing document. This can take a few moments.";

	try {
		const response = await fetch("/upload", {
			method: "POST",
			body: formData,
		});

		const data = await readResponsePayload(response);

        if (!response.ok) {
            throw new Error(data.detail || data.message || "Upload failed.");
        }

        uploadStatus.textContent = `${file.name} uploaded successfully.`;
        helperText.textContent = "Document is ready. Ask a question below.";
        addMessage("assistant", `Document ready: ${file.name}\nAsk me anything about it.`);
	} catch (error) {
        console.error(error);
        uploadStatus.textContent = error.message || "Upload failed.";
    } finally {
        uploadButton.disabled = false;
    }
}

async function askQuestion() {
    const questionInput = getElement("question");
    const helperText = getElement("helperText");
    const askButton = getElement("askButton");
    const chatWindow = getElement("chatWindow");

    if (!questionInput || !helperText || !askButton || !chatWindow) {
        return;
    }

    const question = questionInput.value;

    if (!question) {
        helperText.textContent = "Please enter a question first.";
        return;
    }

    addMessage("user", question);
    questionInput.value = "";
    askButton.disabled = true;
    helperText.textContent = "Thinking...";
    const answerBubble = addMessage("assistant", "Thinking...");

    try {
        const response = await fetch(`/ask?question=${encodeURIComponent(question)}`, {
            method: "POST"
        });

        const data = await readResponsePayload(response);
        if (!response.ok) {
            throw new Error(data.detail || "Error getting answer.");
        }

        answerBubble.textContent = data.answer;
        helperText.textContent = "Ask another question about the same document.";

    } catch (error) {
        console.error(error);
        if (answerBubble) {
            answerBubble.textContent = error.message || "Error getting answer.";
        }
        helperText.textContent = "Something went wrong while getting the answer.";
    } finally {
        askButton.disabled = false;
        questionInput.focus();
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const questionInput = getElement("question");
    if (!questionInput) {
        return;
    }

    questionInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            askQuestion();
        }
    });
});
