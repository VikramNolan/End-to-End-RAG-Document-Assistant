/**
 * RAG Assistant — Intelligent Document AI
 * Comprehensive Frontend Client
 * Handles:
 * - Multi-session chat history (persisted in localStorage)
 * - Document ingestion (PDF, DOCX, TXT)
 * - Document Preview Modal (full text, chunks explorer, highlight navigation)
 * - Knowledge Base Management Hub (analytics KPIs, document table, bulk clear)
 * - Real-time SSE streaming from Groq LLM with source citations
 */

// Use an explicit override when the frontend and API are hosted separately.
const configuredApiBase = window.API_BASE_URL;
const isFilePage = window.location.protocol === "file:";
const isLocalFrontendServer = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname)
  && window.location.port
  && window.location.port !== "8000";
const API_BASE = configuredApiBase
  ? configuredApiBase.replace(/\/$/, "")
  : (isFilePage || isLocalFrontendServer)
    ? "http://127.0.0.1:8000"
    : window.location.origin;

// LocalStorage Keys
const STORAGE_KEY_SESSIONS = "rag_assistant_sessions_v2";
const STORAGE_KEY_ACTIVE_SESSION = "rag_assistant_active_session_v2";

// Application State
const state = {
  sessions: [],
  currentSessionId: null,
  documents: [],
  docSearchQuery: "",
  kbTableSearchQuery: "",
  isGenerating: false,
  systemStatus: {
    groqConfigured: false,
    vectorMode: "pinecone",
    llmModel: "Groq Llama-3.3"
  },
  activePreviewDoc: null,
  activePreviewTab: "full-text",
  highlightedChunkIndex: null
};

// DOM Elements: Core
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const uploadProgress = document.getElementById("upload-progress");
const progressBarFill = document.getElementById("progress-bar-fill");
const progressStatusText = document.getElementById("progress-status-text");
const docList = document.getElementById("doc-list");
const emptyDocsPlaceholder = document.getElementById("empty-docs-placeholder");
const docSearchInput = document.getElementById("doc-search-input");
const tabDocCount = document.getElementById("tab-doc-count");
const tabHistoryCount = document.getElementById("tab-history-count");

// DOM Elements: Navigation Tabs
const tabBtnDocs = document.getElementById("tab-btn-docs");
const tabBtnHistory = document.getElementById("tab-btn-history");
const panelDocs = document.getElementById("panel-docs");
const panelHistory = document.getElementById("panel-history");

// DOM Elements: Chat Workspace
const chatForm = document.getElementById("chat-form");
const chatTextarea = document.getElementById("chat-textarea");
const sendBtn = document.getElementById("send-btn");
const messagesFlow = document.getElementById("messages-flow");
const welcomeScreen = document.getElementById("welcome-screen");
const typingIndicator = document.getElementById("typing-indicator");
const chatContainer = document.getElementById("chat-messages-container");
const clearChatBtn = document.getElementById("clear-chat-btn");
const exportChatBtn = document.getElementById("export-chat-btn");
const activeContextLabel = document.getElementById("active-context-label");
const currentSessionTitle = document.getElementById("current-session-title");
const btnRenameSession = document.getElementById("btn-rename-session");
const btnNewChat = document.getElementById("btn-new-chat");
const historyList = document.getElementById("history-list");

// DOM Elements: Status
const llmStatusDot = document.getElementById("llm-status-dot");
const llmModelName = document.getElementById("llm-model-name");
const vectorStatusDot = document.getElementById("vector-status-dot");
const vectorModeLabel = document.getElementById("vector-mode-label");
const mobileToggle = document.getElementById("mobile-toggle");
const sidebar = document.getElementById("sidebar");

// DOM Elements: Document Preview Modal
const docPreviewModal = document.getElementById("doc-preview-modal");
const previewDocTitle = document.getElementById("preview-doc-title");
const previewTypeIcon = document.getElementById("preview-type-icon");
const previewBadgeChunks = document.getElementById("preview-badge-chunks");
const previewBadgeSize = document.getElementById("preview-badge-size");
const previewBadgeChars = document.getElementById("preview-badge-chars");
const previewChunkCount = document.getElementById("preview-chunk-count");
const docFullTextContent = document.getElementById("doc-full-text-content");
const previewChunksContainer = document.getElementById("preview-chunks-container");
const previewTextSearch = document.getElementById("preview-text-search");
const previewChunksFilter = document.getElementById("preview-chunks-filter");
const chunksInfoText = document.getElementById("chunks-info-text");
const btnCopyDocText = document.getElementById("btn-copy-doc-text");
const btnDownloadDocText = document.getElementById("btn-download-doc-text");
const closePreviewBtn = document.getElementById("close-preview-btn");
const btnClosePreviewFooter = document.getElementById("btn-close-preview-footer");

// DOM Elements: Knowledge Base Manager Modal
const kbManagerModal = document.getElementById("kb-manager-modal");
const openKbManagerBtn = document.getElementById("open-kb-manager-btn");
const openKbHeaderBtn = document.getElementById("open-kb-header-btn");
const footerKbBtn = document.getElementById("footer-kb-btn");
const closeKbManagerBtn = document.getElementById("close-kb-manager-btn");
const kbManagerCloseFooter = document.getElementById("kb-manager-close-footer");
const kbTableBody = document.getElementById("kb-table-body");
const kbTableSearch = document.getElementById("kb-table-search");
const kbUploadBtn = document.getElementById("kb-upload-btn");
const kbClearAllBtn = document.getElementById("kb-clear-all-btn");
const kpiDocsCount = document.getElementById("kpi-docs-count");
const kpiChunksCount = document.getElementById("kpi-chunks-count");
const kpiVectorMode = document.getElementById("kpi-vector-mode");
const kpiEmbeddingDim = document.getElementById("kpi-embedding-dim");

// DOM Elements: Settings Modal
const settingsModal = document.getElementById("settings-modal");
const openSettingsBtn = document.getElementById("open-settings-btn");
const closeSettingsBtn = document.getElementById("close-settings-btn");
const modalOkBtn = document.getElementById("modal-ok-btn");
const diagFastapi = document.getElementById("diag-fastapi");
const diagGroq = document.getElementById("diag-groq");
const diagVector = document.getElementById("diag-vector");
const diagIndex = document.getElementById("diag-index");
const diagEmbedding = document.getElementById("diag-embedding");

/* ==========================================================================
   Initialization
   ========================================================================== */
document.addEventListener("DOMContentLoaded", () => {
  initSessions();
  initEventListeners();
  checkSystemHealth();
  fetchDocumentList();
});

function initEventListeners() {
  // Mobile Sidebar Toggle
  if (mobileToggle) {
    mobileToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }

  // Sidebar Tab Navigation
  tabBtnDocs.addEventListener("click", () => switchSidebarTab("docs"));
  tabBtnHistory.addEventListener("click", () => switchSidebarTab("history"));

  // Dropzone File Upload
  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileSelect);

  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      uploadFile(files[0]);
    }
  });

  // Filter Document List in Sidebar
  if (docSearchInput) {
    docSearchInput.addEventListener("input", (e) => {
      state.docSearchQuery = e.target.value.toLowerCase().trim();
      renderDocumentList();
    });
  }

  // Sessions Management
  btnNewChat.addEventListener("click", () => createNewSession());
  btnRenameSession.addEventListener("click", () => renameCurrentSession());
  currentSessionTitle.addEventListener("click", () => renameCurrentSession());
  clearChatBtn.addEventListener("click", () => clearCurrentSession());
  exportChatBtn.addEventListener("click", () => exportCurrentSession());

  // Textarea auto-resize & keyboard submission
  chatTextarea.addEventListener("input", autoResizeTextarea);
  chatTextarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Chat Form Submission
  chatForm.addEventListener("submit", handleChatSubmit);

  // Starter Prompt Chips
  document.querySelectorAll(".prompt-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const promptText = chip.getAttribute("data-prompt");
      if (promptText) {
        chatTextarea.value = promptText;
        autoResizeTextarea();
        chatForm.dispatchEvent(new Event("submit"));
      }
    });
  });

  // Document Preview Modal Controls
  closePreviewBtn.addEventListener("click", closeDocumentPreview);
  btnClosePreviewFooter.addEventListener("click", closeDocumentPreview);
  docPreviewModal.addEventListener("click", (e) => {
    if (e.target === docPreviewModal) closeDocumentPreview();
  });

  // Preview Tabs
  document.querySelectorAll(".preview-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      const targetView = tab.getAttribute("data-view");
      switchPreviewTab(targetView);
    });
  });

  // Preview Actions
  btnCopyDocText.addEventListener("click", copyFullDocumentText);
  btnDownloadDocText.addEventListener("click", downloadDocumentText);
  if (previewTextSearch) {
    previewTextSearch.addEventListener("input", handlePreviewTextSearch);
  }
  if (previewChunksFilter) {
    previewChunksFilter.addEventListener("input", handlePreviewChunksFilter);
  }

  // Knowledge Base Management Modal Controls
  [openKbManagerBtn, openKbHeaderBtn, footerKbBtn].forEach(btn => {
    if (btn) {
      btn.addEventListener("click", openKbManager);
    }
  });
  closeKbManagerBtn.addEventListener("click", closeKbManager);
  kbManagerCloseFooter.addEventListener("click", closeKbManager);
  kbManagerModal.addEventListener("click", (e) => {
    if (e.target === kbManagerModal) closeKbManager();
  });
  kbUploadBtn.addEventListener("click", () => fileInput.click());
  kbClearAllBtn.addEventListener("click", clearEntireKnowledgeBase);
  if (kbTableSearch) {
    kbTableSearch.addEventListener("input", (e) => {
      state.kbTableSearchQuery = e.target.value.toLowerCase().trim();
      renderKbTable();
    });
  }

  // Settings Modal Controls
  openSettingsBtn.addEventListener("click", () => {
    checkSystemHealth();
    settingsModal.classList.remove("hidden");
  });
  // Also wire the header gear icon button
  const openSettingsHeaderBtn = document.getElementById("open-settings-header-btn");
  if (openSettingsHeaderBtn) {
    openSettingsHeaderBtn.addEventListener("click", () => {
      checkSystemHealth();
      settingsModal.classList.remove("hidden");
    });
  }
  closeSettingsBtn.addEventListener("click", () => settingsModal.classList.add("hidden"));
  modalOkBtn.addEventListener("click", () => settingsModal.classList.add("hidden"));
  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) settingsModal.classList.add("hidden");
  });
}

/* ==========================================================================
   Sidebar Tabs Switching (Documents vs Chat History)
   ========================================================================== */
function switchSidebarTab(tabName) {
  if (tabName === "docs") {
    tabBtnDocs.classList.add("active");
    tabBtnHistory.classList.remove("active");
    panelDocs.classList.add("active");
    panelHistory.classList.remove("active");
  } else {
    tabBtnHistory.classList.add("active");
    tabBtnDocs.classList.remove("active");
    panelHistory.classList.add("active");
    panelDocs.classList.remove("active");
    renderHistoryList();
  }
}

/* ==========================================================================
   Multi-Session Chat History Management
   ========================================================================== */
function initSessions() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_SESSIONS);
    if (saved) {
      state.sessions = JSON.parse(saved);
    }
  } catch (err) {
    console.warn("Could not load sessions from localStorage:", err);
    state.sessions = [];
  }

  const activeId = localStorage.getItem(STORAGE_KEY_ACTIVE_SESSION);
  if (state.sessions.length > 0) {
    const found = state.sessions.find(s => s.id === activeId);
    state.currentSessionId = found ? found.id : state.sessions[0].id;
  } else {
    // Create first session
    const initialSession = {
      id: "session_" + Date.now(),
      title: "New Conversation",
      createdAt: Date.now(),
      messages: []
    };
    state.sessions = [initialSession];
    state.currentSessionId = initialSession.id;
    saveSessions();
  }

  loadActiveSessionToUI();
  renderHistoryList();
}

function saveSessions() {
  try {
    localStorage.setItem(STORAGE_KEY_SESSIONS, JSON.stringify(state.sessions));
    if (state.currentSessionId) {
      localStorage.setItem(STORAGE_KEY_ACTIVE_SESSION, state.currentSessionId);
    }
  } catch (err) {
    console.error("Failed to save sessions to localStorage:", err);
  }
  tabHistoryCount.textContent = state.sessions.length;
}

function getCurrentSession() {
  return state.sessions.find(s => s.id === state.currentSessionId);
}

function createNewSession() {
  const newSession = {
    id: "session_" + Date.now(),
    title: "New Conversation",
    createdAt: Date.now(),
    messages: []
  };
  state.sessions.unshift(newSession);
  state.currentSessionId = newSession.id;
  saveSessions();

  loadActiveSessionToUI();
  renderHistoryList();
  chatTextarea.focus();
  if (window.innerWidth <= 860) sidebar.classList.remove("open");
}

function switchSession(sessionId) {
  if (state.currentSessionId === sessionId) return;
  state.currentSessionId = sessionId;
  saveSessions();
  loadActiveSessionToUI();
  renderHistoryList();
  if (window.innerWidth <= 860) sidebar.classList.remove("open");
}

function deleteSession(sessionId, e) {
  if (e) e.stopPropagation();
  if (!confirm("Are you sure you want to delete this chat session?")) return;

  state.sessions = state.sessions.filter(s => s.id !== sessionId);

  if (state.sessions.length === 0) {
    createNewSession();
  } else {
    if (state.currentSessionId === sessionId) {
      state.currentSessionId = state.sessions[0].id;
    }
    saveSessions();
    loadActiveSessionToUI();
    renderHistoryList();
  }
}

function renameCurrentSession() {
  const current = getCurrentSession();
  if (!current) return;
  const newTitle = prompt("Enter conversation title:", current.title);
  if (newTitle && newTitle.trim()) {
    current.title = newTitle.trim();
    saveSessions();
    currentSessionTitle.textContent = current.title;
    renderHistoryList();
  }
}

function clearCurrentSession() {
  const current = getCurrentSession();
  if (!current) return;
  if (!confirm("Clear all messages in this conversation?")) return;

  current.messages = [];
  saveSessions();
  loadActiveSessionToUI();
}

function exportCurrentSession() {
  const current = getCurrentSession();
  if (!current || current.messages.length === 0) {
    alert("This conversation has no messages to export.");
    return;
  }

  let mdContent = `# ${current.title}\n`;
  mdContent += `*Date: ${new Date(current.createdAt).toLocaleString()}*\n\n---\n\n`;

  current.messages.forEach(msg => {
    const roleName = msg.role === "user" ? "👤 User" : "🧠 RAG Assistant";
    mdContent += `### ${roleName}\n\n${msg.content}\n\n`;

    if (msg.sources && msg.sources.length > 0) {
      mdContent += `**Retrieved Sources:**\n`;
      msg.sources.forEach((s, idx) => {
        mdContent += `- [${s.filename || "Doc"}, Chunk #${s.chunk_index} (Match: ${Math.round((s.score || 0) * 100)}%)]: "${s.text_snippet}"\n`;
      });
      mdContent += "\n";
    }
    mdContent += "---\n\n";
  });

  const blob = new Blob([mdContent], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const cleanTitle = current.title.toLowerCase().replace(/[^a-z0-9]/g, "_");
  a.href = url;
  a.download = `chat_${cleanTitle}_${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

function loadActiveSessionToUI() {
  const current = getCurrentSession();
  if (!current) return;

  currentSessionTitle.textContent = current.title || "Conversation";
  messagesFlow.innerHTML = "";

  if (current.messages.length === 0) {
    welcomeScreen.classList.remove("hidden");
  } else {
    welcomeScreen.classList.add("hidden");
    current.messages.forEach(msg => {
      if (msg.role === "user") {
        appendUserMessage(msg.content, false);
      } else {
        appendAssistantMessageFromHistory(msg.content, msg.sources || []);
      }
    });
  }
  scrollToBottom();
}

function renderHistoryList() {
  tabHistoryCount.textContent = state.sessions.length;
  historyList.innerHTML = "";

  state.sessions.forEach(sess => {
    const item = document.createElement("div");
    item.className = `history-item ${sess.id === state.currentSessionId ? "active" : ""}`;

    const dateStr = formatRelativeTime(sess.createdAt);
    const msgCount = (sess.messages || []).length;

    item.innerHTML = `
      <div class="history-item-left">
        <span class="history-item-title" title="${escapeHtml(sess.title)}">${escapeHtml(sess.title)}</span>
        <div class="history-item-meta">
          <span>${msgCount} message${msgCount !== 1 ? "s" : ""}</span>
          <span>•</span>
          <span>${dateStr}</span>
        </div>
      </div>
      <div class="history-item-actions">
        <button class="history-action-btn edit-btn" title="Rename session">
          <i class="fa-solid fa-pen"></i>
        </button>
        <button class="history-action-btn del-btn" title="Delete session">
          <i class="fa-regular fa-trash-can"></i>
        </button>
      </div>
    `;

    item.addEventListener("click", () => switchSession(sess.id));

    item.querySelector(".edit-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      const newTitle = prompt("Rename session:", sess.title);
      if (newTitle && newTitle.trim()) {
        sess.title = newTitle.trim();
        saveSessions();
        if (sess.id === state.currentSessionId) {
          currentSessionTitle.textContent = sess.title;
        }
        renderHistoryList();
      }
    });

    item.querySelector(".del-btn").addEventListener("click", (e) => {
      deleteSession(sess.id, e);
    });

    historyList.appendChild(item);
  });
}

/* ==========================================================================
   System Health & Diagnostics
   ========================================================================== */
async function checkSystemHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("Health check returned status " + res.status);
    const data = await res.json();

    state.systemStatus.groqConfigured = data.groq_configured;
    state.systemStatus.vectorMode = data.vector_store_mode;

    // Update UI status badges
    if (data.groq_configured) {
      llmStatusDot.className = "status-dot green";
      llmModelName.textContent = "Groq Active";
      diagGroq.textContent = "Connected (API Key OK)";
      diagGroq.style.color = "var(--accent-green)";
    } else {
      llmStatusDot.className = "status-dot amber";
      llmModelName.textContent = "Key Required";
      diagGroq.textContent = "Missing GROQ_API_KEY in .env";
      diagGroq.style.color = "var(--accent-amber)";
    }

    if (data.vector_store_mode === "pinecone") {
      vectorStatusDot.className = "status-dot green";
      vectorModeLabel.textContent = "Pinecone Cloud";
      diagVector.textContent = "Connected (Pinecone Cloud)";
      diagVector.style.color = "var(--accent-green)";
      if (kpiVectorMode) kpiVectorMode.textContent = "Pinecone Cloud";
    } else {
      vectorStatusDot.className = "status-dot cyan";
      vectorModeLabel.textContent = "Local Store";
      diagVector.textContent = "Active (Local Cosine Store)";
      diagVector.style.color = "var(--accent-cyan)";
      if (kpiVectorMode) kpiVectorMode.textContent = "Local Store";
    }

    diagFastapi.textContent = "Online (Ready)";
    diagFastapi.style.color = "var(--accent-green)";
  } catch (err) {
    console.warn("Could not reach backend health check:", err);
    llmStatusDot.className = "status-dot amber";
    vectorStatusDot.className = "status-dot amber";
    diagFastapi.textContent = "Disconnected (Check backend terminal)";
    diagFastapi.style.color = "var(--accent-rose)";
  }
}

/* ==========================================================================
   Document Management (Upload, List, Delete, Preview)
   ========================================================================== */
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) {
    uploadFile(file);
  }
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  uploadProgress.classList.remove("hidden");
  progressStatusText.textContent = `Uploading & chunking "${file.name}"...`;

  try {
    const res = await fetch(`${API_BASE}/api/documents/upload`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Upload failed with status " + res.status);
    }

    const data = await res.json();
    progressStatusText.textContent = `Indexed ${data.document.total_chunks} chunks successfully!`;
    setTimeout(() => {
      uploadProgress.classList.add("hidden");
      fileInput.value = "";
    }, 2000);

    // Refresh doc list and table
    await fetchDocumentList();
    if (!kbManagerModal.classList.contains("hidden")) {
      renderKbTable();
    }
  } catch (err) {
    console.error("Upload error:", err);
    progressStatusText.textContent = `Error: ${err.message}`;
    progressStatusText.style.color = "var(--accent-rose)";
    setTimeout(() => {
      uploadProgress.classList.add("hidden");
      progressStatusText.style.color = "var(--accent-cyan)";
      fileInput.value = "";
    }, 3500);
  }
}

async function fetchDocumentList() {
  try {
    const res = await fetch(`${API_BASE}/api/documents/`);
    if (!res.ok) throw new Error("Failed to fetch documents");
    const data = await res.json();

    state.documents = data.documents || [];
    renderDocumentList();
    updateKnowledgeBaseStats();
  } catch (err) {
    console.error("Error fetching documents:", err);
  }
}

function updateKnowledgeBaseStats() {
  const totalDocs = state.documents.length;
  const totalChunks = state.documents.reduce((sum, d) => sum + (d.chunk_count || 0), 0);
  tabDocCount.textContent = totalDocs;

  if (kpiDocsCount) kpiDocsCount.textContent = totalDocs;
  if (kpiChunksCount) kpiChunksCount.textContent = totalChunks;

  activeContextLabel.textContent = totalDocs === 0
    ? "Ready to search knowledge base (0 docs)"
    : `Knowledge Base: ${totalDocs} doc${totalDocs > 1 ? "s" : ""} (${totalChunks} chunks)`;
}

function renderDocumentList() {
  tabDocCount.textContent = state.documents.length;

  let filtered = state.documents;
  if (state.docSearchQuery) {
    filtered = state.documents.filter(d =>
      (d.filename || "").toLowerCase().includes(state.docSearchQuery)
    );
  }

  if (state.documents.length === 0) {
    emptyDocsPlaceholder.classList.remove("hidden");
    docList.innerHTML = "";
    docList.appendChild(emptyDocsPlaceholder);
    return;
  }

  emptyDocsPlaceholder.classList.add("hidden");
  docList.innerHTML = "";

  if (filtered.length === 0) {
    const noResults = document.createElement("div");
    noResults.className = "empty-docs";
    noResults.innerHTML = `<p>No documents match "${escapeHtml(state.docSearchQuery)}"</p>`;
    docList.appendChild(noResults);
    return;
  }

  filtered.forEach(doc => {
    const card = document.createElement("div");
    card.className = "doc-card";

    const ext = (doc.file_type || doc.filename.split(".").pop() || "txt").toLowerCase();
    let badgeClass = "doc-badge txt";
    let iconClass = "fa-file-lines";

    if (ext === "pdf") {
      badgeClass = "doc-badge pdf";
      iconClass = "fa-file-pdf";
    } else if (ext === "docx" || ext === "doc") {
      badgeClass = "doc-badge docx";
      iconClass = "fa-file-word";
    }

    const formattedSize = formatBytes(doc.file_size || 0);

    card.innerHTML = `
      <div class="doc-left">
        <div class="${badgeClass}">
          <i class="fa-regular ${iconClass}"></i>
        </div>
        <div class="doc-meta">
          <span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
          <span class="doc-subtext">
            <span>${doc.chunk_count} chunks</span>
            <span>•</span>
            <span>${formattedSize}</span>
          </span>
        </div>
      </div>
      <div class="doc-actions">
        <button class="doc-action-btn preview-btn" title="Preview document & chunks" data-doc-id="${doc.doc_id}">
          <i class="fa-regular fa-eye"></i>
        </button>
        <button class="doc-action-btn delete-btn" title="Delete document" data-doc-id="${doc.doc_id}">
          <i class="fa-regular fa-trash-can"></i>
        </button>
      </div>
    `;

    // Click card to open preview
    card.addEventListener("click", () => {
      openDocumentPreview(doc.doc_id);
    });

    // Preview button
    card.querySelector(".preview-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      openDocumentPreview(doc.doc_id);
    });

    // Delete button
    card.querySelector(".delete-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteDocument(doc.doc_id, doc.filename);
    });

    docList.appendChild(card);
  });
}

async function deleteDocument(docId, filename) {
  if (!confirm(`Are you sure you want to delete "${filename}" from the knowledge base?`)) {
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/documents/${docId}`, {
      method: "DELETE"
    });
    if (!res.ok) throw new Error("Delete failed");
    await fetchDocumentList();
    if (!kbManagerModal.classList.contains("hidden")) {
      renderKbTable();
    }
  } catch (err) {
    alert("Failed to delete document: " + err.message);
  }
}

async function clearEntireKnowledgeBase() {
  if (!confirm("CAUTION: This will delete ALL documents and their embeddings from the vector store. Are you sure?")) {
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/documents/clear`, {
      method: "DELETE"
    });
    if (!res.ok) throw new Error("Clear failed");
    await fetchDocumentList();
    renderKbTable();
    alert("Knowledge base successfully cleared.");
  } catch (err) {
    alert("Failed to clear knowledge base: " + err.message);
  }
}

/* ==========================================================================
   Document Preview Modal (Full Text & Chunks Explorer)
   ========================================================================== */
async function openDocumentPreview(docId, highlightChunkIdx = null) {
  state.highlightedChunkIndex = highlightChunkIdx;

  try {
    const res = await fetch(`${API_BASE}/api/documents/${docId}/preview`);
    if (!res.ok) throw new Error("Failed to load document preview");
    const data = await res.json();
    const preview = data.preview;

    state.activePreviewDoc = preview;

    // Header info
    previewDocTitle.textContent = preview.filename || "Document Preview";
    const ext = (preview.file_type || "").toLowerCase();
    previewTypeIcon.innerHTML = ext === "pdf"
      ? `<i class="fa-regular fa-file-pdf" style="color: var(--accent-rose);"></i>`
      : ext === "docx" || ext === "doc"
      ? `<i class="fa-regular fa-file-word" style="color: #60a5fa;"></i>`
      : `<i class="fa-regular fa-file-lines" style="color: var(--accent-green);"></i>`;

    previewBadgeChunks.textContent = `${preview.total_chunks || (preview.chunks || []).length} Chunks`;
    previewBadgeSize.textContent = formatBytes(preview.file_size || 0);
    previewBadgeChars.textContent = `${(preview.char_count || (preview.full_text || "").length).toLocaleString()} Chars`;
    previewChunkCount.textContent = (preview.chunks || []).length;

    // Render Full Text
    docFullTextContent.textContent = preview.full_text || "No text available.";

    // Render Chunks Explorer
    renderChunksList(preview.chunks || []);

    // Render Metadata
    document.getElementById("meta-doc-id").textContent = preview.doc_id || "-";
    document.getElementById("meta-filename").textContent = preview.filename || "-";
    document.getElementById("meta-file-type").textContent = (preview.file_type || "-").toUpperCase();
    document.getElementById("meta-file-size").textContent = formatBytes(preview.file_size || 0);
    document.getElementById("meta-chars").textContent = (preview.char_count || 0).toLocaleString();
    document.getElementById("meta-chunks").textContent = preview.total_chunks || 0;
    document.getElementById("meta-uploaded-at").textContent = preview.uploaded_at || "-";

    // If highlighted chunk requested, switch directly to chunks tab and scroll
    if (highlightChunkIdx !== null) {
      switchPreviewTab("chunks-view");
      setTimeout(() => {
        const targetChunkCard = document.getElementById(`chunk-card-${highlightChunkIdx}`);
        if (targetChunkCard) {
          targetChunkCard.scrollIntoView({ behavior: "smooth", block: "center" });
          targetChunkCard.classList.add("highlighted");
        }
      }, 200);
    } else {
      switchPreviewTab("full-text");
    }

    docPreviewModal.classList.remove("hidden");
  } catch (err) {
    alert("Could not open preview: " + err.message);
  }
}

function closeDocumentPreview() {
  docPreviewModal.classList.add("hidden");
  state.activePreviewDoc = null;
  state.highlightedChunkIndex = null;
}

function switchPreviewTab(viewName) {
  state.activePreviewTab = viewName;

  document.querySelectorAll(".preview-tab").forEach(tab => {
    tab.classList.toggle("active", tab.getAttribute("data-view") === viewName);
  });

  document.querySelectorAll(".preview-view-panel").forEach(panel => {
    panel.classList.toggle("active", panel.id === `view-${viewName}`);
  });
}

function renderChunksList(chunks) {
  previewChunksContainer.innerHTML = "";
  chunksInfoText.textContent = `Total ${chunks.length} chunks indexed`;

  if (chunks.length === 0) {
    previewChunksContainer.innerHTML = `<p class="empty-docs">No chunk segments available.</p>`;
    return;
  }

  chunks.forEach(chunk => {
    const card = document.createElement("div");
    card.className = `chunk-card ${state.highlightedChunkIndex === chunk.chunk_index ? "highlighted" : ""}`;
    card.id = `chunk-card-${chunk.chunk_index}`;

    card.innerHTML = `
      <div class="chunk-card-header">
        <span class="chunk-index-badge">Chunk #${chunk.chunk_index} • ${chunk.length || chunk.text.length} chars</span>
        <button class="chunk-copy-btn" title="Copy chunk text">
          <i class="fa-regular fa-copy"></i> Copy
        </button>
      </div>
      <div class="chunk-text">${escapeHtml(chunk.text)}</div>
    `;

    card.querySelector(".chunk-copy-btn").addEventListener("click", () => {
      navigator.clipboard.writeText(chunk.text).then(() => {
        const btn = card.querySelector(".chunk-copy-btn");
        btn.innerHTML = `<i class="fa-solid fa-check"></i> Copied!`;
        setTimeout(() => {
          btn.innerHTML = `<i class="fa-regular fa-copy"></i> Copy`;
        }, 1500);
      });
    });

    previewChunksContainer.appendChild(card);
  });
}

function handlePreviewTextSearch(e) {
  const query = e.target.value.toLowerCase();
  const text = state.activePreviewDoc ? state.activePreviewDoc.full_text || "" : "";
  if (!query) {
    docFullTextContent.textContent = text;
    return;
  }

  // Highlight matches in text
  const regex = new RegExp(`(${escapeRegex(query)})`, "gi");
  const highlighted = escapeHtml(text).replace(regex, '<mark style="background: rgba(6, 182, 212, 0.4); color: #fff; border-radius: 2px;">$1</mark>');
  docFullTextContent.innerHTML = highlighted;
}

function handlePreviewChunksFilter(e) {
  const query = e.target.value.toLowerCase().trim();
  const chunks = (state.activePreviewDoc && state.activePreviewDoc.chunks) || [];

  if (!query) {
    renderChunksList(chunks);
    return;
  }

  const filtered = chunks.filter(c => c.text.toLowerCase().includes(query));
  chunksInfoText.textContent = `Found ${filtered.length} of ${chunks.length} chunks`;
  renderChunksList(filtered);
}

function copyFullDocumentText() {
  if (!state.activePreviewDoc || !state.activePreviewDoc.full_text) return;
  navigator.clipboard.writeText(state.activePreviewDoc.full_text).then(() => {
    btnCopyDocText.innerHTML = `<i class="fa-solid fa-check"></i> <span>Copied!</span>`;
    setTimeout(() => {
      btnCopyDocText.innerHTML = `<i class="fa-regular fa-copy"></i> <span>Copy Text</span>`;
    }, 2000);
  });
}

function downloadDocumentText() {
  if (!state.activePreviewDoc) return;
  const docId = state.activePreviewDoc.doc_id;
  window.location.href = `${API_BASE}/api/documents/${docId}/download`;
}

/* ==========================================================================
   Knowledge Base Management Hub Modal
   ========================================================================== */
function openKbManager() {
  renderKbTable();
  updateKnowledgeBaseStats();
  kbManagerModal.classList.remove("hidden");
}

function closeKbManager() {
  kbManagerModal.classList.add("hidden");
}

function renderKbTable() {
  kbTableBody.innerHTML = "";

  let list = state.documents;
  if (state.kbTableSearchQuery) {
    list = list.filter(d => (d.filename || "").toLowerCase().includes(state.kbTableSearchQuery));
  }

  if (list.length === 0) {
    kbTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-dim); padding: 30px;">No documents in knowledge base</td></tr>`;
    return;
  }

  list.forEach(doc => {
    const tr = document.createElement("tr");
    const ext = (doc.file_type || doc.filename.split(".").pop() || "txt").toUpperCase();
    const tagClass = ext === "PDF" ? "tag-pdf" : ext.startsWith("DOC") ? "tag-docx" : "tag-txt";

    tr.innerHTML = `
      <td><strong>${escapeHtml(doc.filename)}</strong></td>
      <td><span class="tag-badge ${tagClass}">${ext}</span></td>
      <td><span style="color: var(--accent-cyan); font-weight: 600;">${doc.chunk_count}</span></td>
      <td>${formatBytes(doc.file_size || 0)}</td>
      <td style="color: var(--text-dim);">${escapeHtml(doc.uploaded_at || "-")}</td>
      <td>
        <div class="table-action-btns">
          <button class="doc-action-btn preview-btn" title="Inspect Document" data-doc-id="${doc.doc_id}">
            <i class="fa-regular fa-eye"></i>
          </button>
          <a class="doc-action-btn" title="Download Text" href="${API_BASE}/api/documents/${doc.doc_id}/download" download>
            <i class="fa-solid fa-download"></i>
          </a>
          <button class="doc-action-btn delete-btn" title="Delete from index" data-doc-id="${doc.doc_id}">
            <i class="fa-regular fa-trash-can"></i>
          </button>
        </div>
      </td>
    `;

    tr.querySelector(".preview-btn").addEventListener("click", () => {
      closeKbManager();
      openDocumentPreview(doc.doc_id);
    });

    tr.querySelector(".delete-btn").addEventListener("click", () => {
      deleteDocument(doc.doc_id, doc.filename);
    });

    kbTableBody.appendChild(tr);
  });
}

/* ==========================================================================
   Chat & RAG Generation
   ========================================================================== */
function autoResizeTextarea() {
  chatTextarea.style.height = "auto";
  chatTextarea.style.height = `${Math.min(chatTextarea.scrollHeight, 180)}px`;
}

async function handleChatSubmit(e) {
  e.preventDefault();
  const query = chatTextarea.value.trim();
  if (!query || state.isGenerating) return;

  const current = getCurrentSession();
  if (!current) return;

  // Auto-rename "New Conversation" on first question
  if (current.title === "New Conversation" && current.messages.length === 0) {
    current.title = query.length > 36 ? query.slice(0, 36) + "..." : query;
    currentSessionTitle.textContent = current.title;
    saveSessions();
    renderHistoryList();
  }

  // Hide welcome screen
  welcomeScreen.classList.add("hidden");

  // Append user message to UI and session state
  appendUserMessage(query, true);

  // Clear input
  chatTextarea.value = "";
  autoResizeTextarea();

  // Disable send button and show typing indicator
  state.isGenerating = true;
  sendBtn.disabled = true;
  typingIndicator.classList.remove("hidden");
  scrollToBottom();

  // Create empty assistant message element
  const assistantMsgObj = createAssistantMessageElement();

  try {
    // Attempt SSE Streaming first
    const recentHistory = current.messages.slice(-6).map(m => ({ role: m.role, content: m.content }));
    const response = await fetch(`${API_BASE}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        history: recentHistory
      })
    });

    if (!response.ok) {
      throw new Error(`Server returned error code ${response.status}`);
    }

    typingIndicator.classList.add("hidden");

    let fullAnswer = "";
    let retrievedSources = [];

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          try {
            const parsed = JSON.parse(jsonStr);

            if (parsed.type === "metadata") {
              retrievedSources = parsed.sources || [];
              if (retrievedSources.length > 0) {
                renderSourcesAccordion(assistantMsgObj.sourcesContainer, retrievedSources);
              }
            } else if (parsed.type === "token") {
              fullAnswer += parsed.content;
              assistantMsgObj.contentElem.innerHTML = renderMarkdown(fullAnswer);
              scrollToBottom();
            } else if (parsed.type === "done") {
              // Generation complete
            }
          } catch (jsonErr) {
            console.warn("Error parsing stream chunk:", jsonErr);
          }
        }
      }
    }

    // Save assistant response to session
    current.messages.push({
      role: "assistant",
      content: fullAnswer,
      sources: retrievedSources
    });
    saveSessions();

    // Enable copy & regenerate buttons
    setupCopyButton(assistantMsgObj.copyBtn, fullAnswer);
    setupRegenerateButton(assistantMsgObj.regenBtn, query);

  } catch (err) {
    console.error("Streaming error, falling back to standard endpoint:", err);

    // Fallback to non-streaming POST /api/chat
    try {
      const recentHistory = current.messages.slice(-6).map(m => ({ role: m.role, content: m.content }));
      const fallbackRes = await fetch(`${API_BASE}/api/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          history: recentHistory
        })
      });

      typingIndicator.classList.add("hidden");

      if (!fallbackRes.ok) {
        throw new Error("Chat request failed.");
      }

      const resData = await fallbackRes.json();
      assistantMsgObj.contentElem.innerHTML = renderMarkdown(resData.answer || "");

      if (resData.sources && resData.sources.length > 0) {
        renderSourcesAccordion(assistantMsgObj.sourcesContainer, resData.sources);
      }

      current.messages.push({
        role: "assistant",
        content: resData.answer,
        sources: resData.sources || []
      });
      saveSessions();

      setupCopyButton(assistantMsgObj.copyBtn, resData.answer);
      setupRegenerateButton(assistantMsgObj.regenBtn, query);

    } catch (fallbackErr) {
      typingIndicator.classList.add("hidden");
      assistantMsgObj.contentElem.innerHTML = `<p style="color: var(--accent-rose);"><strong>Error:</strong> ${escapeHtml(fallbackErr.message || err.message)}</p>`;
    }
  } finally {
    state.isGenerating = false;
    sendBtn.disabled = false;
    chatTextarea.focus();
    scrollToBottom();
    renderHistoryList();
  }
}

function appendUserMessage(text, saveToState = false) {
  if (saveToState) {
    const current = getCurrentSession();
    if (current) {
      current.messages.push({ role: "user", content: text });
      saveSessions();
    }
  }

  const row = document.createElement("div");
  row.className = "message-row user";
  row.innerHTML = `
    <div class="message-avatar user-avatar">
      <i class="fa-solid fa-user"></i>
    </div>
    <div class="message-content-wrapper">
      <div class="message-bubble">
        <p>${escapeHtml(text)}</p>
      </div>
    </div>
  `;
  messagesFlow.appendChild(row);
  scrollToBottom();
}

function appendAssistantMessageFromHistory(text, sources) {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.innerHTML = `
    <div class="message-avatar ai-avatar">
      <i class="fa-solid fa-brain"></i>
    </div>
    <div class="message-content-wrapper">
      <div class="message-bubble">
        ${renderMarkdown(text)}
      </div>
      <div class="sources-accordion"></div>
      <div class="message-actions">
        <button class="copy-btn" title="Copy answer">
          <i class="fa-regular fa-copy"></i>
          <span>Copy</span>
        </button>
      </div>
    </div>
  `;

  if (sources && sources.length > 0) {
    renderSourcesAccordion(row.querySelector(".sources-accordion"), sources);
  }

  setupCopyButton(row.querySelector(".copy-btn"), text);
  messagesFlow.appendChild(row);
}

function createAssistantMessageElement() {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.innerHTML = `
    <div class="message-avatar ai-avatar">
      <i class="fa-solid fa-brain"></i>
    </div>
    <div class="message-content-wrapper">
      <div class="message-bubble" id="assistant-bubble">
        <span class="loading-cursor">▍</span>
      </div>
      <div class="sources-accordion"></div>
      <div class="message-actions">
        <button class="copy-btn" title="Copy answer">
          <i class="fa-regular fa-copy"></i>
          <span>Copy</span>
        </button>
        <button class="regen-btn action-btn-sm" title="Regenerate response">
          <i class="fa-solid fa-rotate"></i>
          <span>Regenerate</span>
        </button>
      </div>
    </div>
  `;

  messagesFlow.appendChild(row);
  scrollToBottom();

  return {
    contentElem: row.querySelector(".message-bubble"),
    sourcesContainer: row.querySelector(".sources-accordion"),
    copyBtn: row.querySelector(".copy-btn"),
    regenBtn: row.querySelector(".regen-btn")
  };
}

function renderSourcesAccordion(container, sources) {
  if (!sources || sources.length === 0) return;

  const toggleBtn = document.createElement("button");
  toggleBtn.className = "sources-toggle-btn";
  toggleBtn.innerHTML = `
    <i class="fa-solid fa-database"></i>
    <span>Retrieved Context (${sources.length} sources)</span>
    <i class="fa-solid fa-chevron-down" style="font-size: 0.7em;"></i>
  `;

  const cardsContainer = document.createElement("div");
  cardsContainer.className = "sources-cards-container hidden";

  sources.forEach((src, idx) => {
    const card = document.createElement("div");
    card.className = "source-card";
    const scorePercent = Math.round((src.score || 0) * 100);

    card.innerHTML = `
      <div class="source-header">
        <div class="source-title">
          <i class="fa-regular fa-file-lines"></i>
          <span>${escapeHtml(src.filename || "Document")} (Chunk #${src.chunk_index !== undefined ? src.chunk_index : idx})</span>
        </div>
        <span class="source-score-badge">Match: ${scorePercent}%</span>
      </div>
      <div class="source-snippet">
        "${escapeHtml(src.text_snippet || "")}"
      </div>
      <div class="source-preview-hint">
        <i class="fa-solid fa-arrow-up-right-from-square"></i>
        <span>Click to preview chunk in document viewer</span>
      </div>
    `;

    // Click source card to navigate directly to document preview
    card.addEventListener("click", () => {
      // Find matching doc_id by filename if not directly on src
      let docId = src.doc_id;
      if (!docId && state.documents.length > 0) {
        const found = state.documents.find(d => d.filename === src.filename);
        if (found) docId = found.doc_id;
      }
      if (docId) {
        openDocumentPreview(docId, src.chunk_index !== undefined ? src.chunk_index : idx);
      } else {
        alert("Original document data is being indexed or not found.");
      }
    });

    cardsContainer.appendChild(card);
  });

  toggleBtn.addEventListener("click", () => {
    cardsContainer.classList.toggle("hidden");
    const chevron = toggleBtn.querySelector(".fa-chevron-down, .fa-chevron-up");
    if (cardsContainer.classList.contains("hidden")) {
      chevron.className = "fa-solid fa-chevron-down";
    } else {
      chevron.className = "fa-solid fa-chevron-up";
    }
  });

  container.innerHTML = "";
  container.appendChild(toggleBtn);
  container.appendChild(cardsContainer);
}

function setupCopyButton(btn, text) {
  if (!btn) return;
  btn.addEventListener("click", () => {
    navigator.clipboard.writeText(text).then(() => {
      btn.innerHTML = `<i class="fa-solid fa-check"></i> <span>Copied!</span>`;
      setTimeout(() => {
        btn.innerHTML = `<i class="fa-regular fa-copy"></i> <span>Copy</span>`;
      }, 2000);
    });
  });
}

function setupRegenerateButton(btn, query) {
  if (!btn) return;
  btn.addEventListener("click", () => {
    if (state.isGenerating) return;
    chatTextarea.value = query;
    autoResizeTextarea();
    chatForm.dispatchEvent(new Event("submit"));
  });
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

/* ==========================================================================
   Utilities (Markdown, Escaping, Formatting)
   ========================================================================== */
function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeRegex(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function formatBytes(bytes) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

function formatRelativeTime(timestamp) {
  if (!timestamp) return "";
  const now = Date.now();
  const diffSec = Math.floor((now - timestamp) / 1000);

  if (diffSec < 60) return "Just now";
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return new Date(timestamp).toLocaleDateString();
}

// Lightweight Markdown Renderer for AI responses
function renderMarkdown(raw) {
  if (!raw) return "";

  let html = raw;

  // Code blocks: ```lang\ncode\n```
  html = html.replace(/```([a-zA-Z0-9]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
  });

  // Inline code: `code`
  html = html.replace(/`([^`]+)`/g, (match, code) => {
    return `<code>${escapeHtml(code)}</code>`;
  });

  // Bold: **text**
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // Italic: *text*
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");

  // Headers
  html = html.replace(/^### (.*$)/gim, '<h4 style="margin: 12px 0 6px 0; color: #38bdf8;">$1</h4>');
  html = html.replace(/^## (.*$)/gim, '<h3 style="margin: 14px 0 8px 0; color: #818cf8;">$1</h3>');
  html = html.replace(/^# (.*$)/gim, '<h2 style="margin: 16px 0 10px 0; color: #a78bfa;">$1</h2>');

  // Bullet Lists and Paragraphs
  const paragraphs = html.split("\n\n");
  const formattedParagraphs = paragraphs.map(para => {
    const lines = para.split("\n");
    const isBulletList = lines.every(l => l.trim().startsWith("- ") || l.trim().startsWith("* ") || l.trim() === "");
    const isNumberedList = lines.every(l => /^\d+\.\s/.test(l.trim()) || l.trim() === "");

    if (isBulletList && lines.some(l => l.trim().length > 0)) {
      const items = lines
        .filter(l => l.trim().length > 0)
        .map(l => `<li>${l.replace(/^[-*]\s+/, "")}</li>`)
        .join("");
      return `<ul>${items}</ul>`;
    } else if (isNumberedList && lines.some(l => l.trim().length > 0)) {
      const items = lines
        .filter(l => l.trim().length > 0)
        .map(l => `<li>${l.replace(/^\d+\.\s+/, "")}</li>`)
        .join("");
      return `<ol>${items}</ol>`;
    } else {
      return `<p>${para.replace(/\n/g, "<br>")}</p>`;
    }
  });

  return formattedParagraphs.join("");
}
