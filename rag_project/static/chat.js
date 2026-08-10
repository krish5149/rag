const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const queryInput = document.getElementById("query-input");
const sendButton = document.getElementById("send-button");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");

function addMessage(text, isUser = false) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${isUser ? "user" : "system"}`;

  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;

  wrapper.appendChild(content);
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function setQueryEnabled(enabled) {
  queryInput.disabled = !enabled;
  sendButton.disabled = !enabled;
  if (!enabled) {
    queryInput.placeholder = "Upload files first to enable query.";
  } else {
    queryInput.placeholder = "Ask something...";
  }
}

async function uploadFiles(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  uploadStatus.textContent = "Uploading files...";

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      uploadStatus.textContent = `Upload failed: ${data.detail || data.message || response.statusText}`;
      return false;
    }

    uploadStatus.textContent = `Uploaded ${data.length} file(s) successfully.`;
    return true;
  } catch (error) {
    uploadStatus.textContent = `Network error: ${error.message}`;
    return false;
  }
}

async function sendQuery(query) {
  addMessage(query, true);
  addMessage("Thinking...", false);

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const lastSystem = chatWindow.querySelector(".message.system:last-child .message-content");
    let data = null;
    let text = null;

    try {
      data = await response.json();
    } catch (parseError) {
      text = await response.text();
    }

    if (!response.ok) {
      const errorMessage = data?.detail || data?.message || text || response.statusText;
      if (lastSystem) lastSystem.textContent = `Error: ${errorMessage}`;
      return;
    }

    if (lastSystem) lastSystem.textContent = data?.answer || text || "No answer returned.";
  } catch (error) {
    const lastSystem = chatWindow.querySelector(".message.system:last-child .message-content");
    if (lastSystem) lastSystem.textContent = `Network error: ${error.message}`;
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const files = fileInput.files;
  if (!files || !files.length) {
    uploadStatus.textContent = "Please choose at least one file to upload.";
    return;
  }

  const success = await uploadFiles(files);
  if (success) {
    setQueryEnabled(true);
    addMessage(`Uploaded ${files.length} file(s). You can now ask a question.`, false);
  } else {
    setQueryEnabled(false);
  }
});

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  queryInput.value = "";
  sendQuery(query);
});

setQueryEnabled(false);
