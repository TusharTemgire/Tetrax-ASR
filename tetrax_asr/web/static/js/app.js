// Tetrax ASR Real-Time Speech Intelligence Console

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initStreaming();
  initFileUpload();
  checkSystemStatus();
});

// Tab Navigation
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(btn => {
    btn.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");
    });
  });
}

// Global Streaming State
let isStreaming = false;
let socket = null;
let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let analyser = null;
let animationFrameId = null;

function initStreaming() {
  const streamBtn = document.getElementById("streamBtn");
  const streamStatus = document.getElementById("streamStatus");
  const captionsBox = document.getElementById("captionsBox");
  const canvas = document.getElementById("visualizer");
  const canvasCtx = canvas.getContext("2d");

  streamBtn.addEventListener("click", async () => {
    if (!isStreaming) {
      try {
        await startStreaming(canvas, canvasCtx);
      } catch (err) {
        console.error("Failed to start audio stream:", err);
        alert(`Could not access microphone: ${err.message}`);
      }
    } else {
      stopStreaming();
    }
  });
}

async function startStreaming(canvas, canvasCtx) {
  const streamBtn = document.getElementById("streamBtn");
  const streamStatus = document.getElementById("streamStatus");
  const interimContainer = document.getElementById("interimCaption");

  // 1. Get User Microphone
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: { ideal: 16000 },
      echoCancellation: true,
      noiseSuppression: true
    }
  });

  // 2. Setup AudioContext
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioContext.createMediaStreamSource(mediaStream);

  // Setup Analyser for visualizer
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);
  drawVisualizer(canvas, canvasCtx);

  // 3. Connect WebSocket with selected language
  const selectedLang = document.getElementById("streamLangSelect")?.value || "mr";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/v1/audio/stream?language=${selectedLang}`;
  socket = new WebSocket(wsUrl);

  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    isStreaming = true;
    streamBtn.classList.add("active");
    streamBtn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
        <rect x="6" y="6" width="12" height="12" rx="2"></rect>
      </svg>
      Stop Listening
    `;
    const langLabel = selectedLang === "mr" ? "मराठी (Marathi)" : "हिंदी (Hindi)";
    streamStatus.textContent = `Listening (${langLabel} Active)`;
    streamStatus.style.color = "#10b981";
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleStreamMessage(data);
    } catch (e) {
      console.warn("Non-JSON WS message:", event.data);
    }
  };

  socket.onerror = (err) => {
    console.error("WebSocket error:", err);
    streamStatus.textContent = "Connection error";
    streamStatus.style.color = "#f43f5e";
  };

  socket.onclose = () => {
    if (isStreaming) {
      stopStreaming();
    }
  };

  // 4. Downsample & Send PCM Audio Chunks
  const bufferSize = 4096;
  scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);
  analyser.connect(scriptProcessor);
  scriptProcessor.connect(audioContext.destination);

  scriptProcessor.onaudioprocess = (e) => {
    if (!isStreaming || !socket || socket.readyState !== WebSocket.OPEN) return;

    const inputData = e.inputBuffer.getChannelData(0);
    const targetSampleRate = 16000;
    const currentSampleRate = audioContext.sampleRate;

    // Resample if necessary
    const resampled = downsampleBuffer(inputData, currentSampleRate, targetSampleRate);
    // Convert Float32 to Int16 PCM
    const pcm16 = floatTo16BitPCM(resampled);

    socket.send(pcm16.buffer);
  };
}

function stopStreaming() {
  isStreaming = false;
  const streamBtn = document.getElementById("streamBtn");
  const streamStatus = document.getElementById("streamStatus");
  const interimContainer = document.getElementById("interimCaption");

  if (socket) {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: "stop" }));
    }
    socket.close();
    socket = null;
  }

  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }

  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }

  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }

  streamBtn.classList.remove("active");
  streamBtn.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"></path>
      <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"></path>
    </svg>
    Start Live Speech Stream
  `;
  streamStatus.textContent = "Ready to Stream";
  streamStatus.style.color = "var(--text-muted)";
  if (interimContainer) interimContainer.textContent = "";
}

function handleStreamMessage(data) {
  const committedContainer = document.getElementById("committedCaptions");
  const interimContainer = document.getElementById("interimCaption");
  const captionsBox = document.getElementById("captionsBox");

  if (data.event === "interim") {
    interimContainer.innerHTML = `${escapeHtml(data.text)} <span class="cursor-blink"></span>`;
    captionsBox.scrollTop = captionsBox.scrollHeight;
  } else if (data.event === "final") {
    interimContainer.textContent = "";
    const div = document.createElement("div");
    div.className = "caption-segment";
    div.innerHTML = `
      <span class="caption-time">[${formatSeconds(data.start)} - ${formatSeconds(data.end)}]</span>
      <span class="caption-committed">${escapeHtml(data.text)}</span>
    `;
    committedContainer.appendChild(div);
    captionsBox.scrollTop = captionsBox.scrollHeight;
  }
}

// Audio visualizer drawing loop
function drawVisualizer(canvas, canvasCtx) {
  if (!analyser) return;

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function render() {
    animationFrameId = requestAnimationFrame(render);
    analyser.getByteFrequencyData(dataArray);

    canvasCtx.fillStyle = "rgba(11, 15, 25, 0.4)";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    const barWidth = (canvas.width / bufferLength) * 2.5;
    let barHeight;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      barHeight = (dataArray[i] / 255) * canvas.height;
      const gradient = canvasCtx.createLinearGradient(0, canvas.height, 0, 0);
      gradient.addColorStop(0, "#0284c7");
      gradient.addColorStop(1, "#38bdf8");

      canvasCtx.fillStyle = gradient;
      canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
      x += barWidth + 1;
    }
  }

  render();
}

// Audio Resampling & Conversion Utilities
function downsampleBuffer(buffer, sampleRate, outSampleRate) {
  if (outSampleRate === sampleRate) {
    return buffer;
  }
  if (outSampleRate > sampleRate) {
    throw new Error("Downsampling rate must be smaller than original rate");
  }
  const sampleRateRatio = sampleRate / outSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function floatTo16BitPCM(input) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return output;
}

// File Upload Handling
function initFileUpload() {
  const dropzone = document.getElementById("uploadDropzone");
  const fileInput = document.getElementById("audioFileInput");
  const audioPreview = document.getElementById("audioPreview");
  const audioPlayer = document.getElementById("audioPlayer");
  const transcribeBtn = document.getElementById("transcribeBtn");
  const fileResultCard = document.getElementById("fileResultCard");
  let selectedFile = null;

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    document.getElementById("selectedFileName").textContent = `${file.name} (${(file.size / (1024*1024)).toFixed(2)} MB)`;
    audioPlayer.src = URL.createObjectURL(file);
    audioPreview.style.display = "block";
    transcribeBtn.disabled = false;
  }

  transcribeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    transcribeBtn.disabled = true;
    transcribeBtn.innerHTML = `
      <svg class="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <circle cx="12" cy="12" r="10" stroke-width="4" stroke-opacity="0.25"></circle>
        <path d="M12 2a10 10 0 0 1 10 10" stroke-width="4"></path>
      </svg>
      Transcribing...
    `;

    const selectedLang = document.getElementById("uploadLangSelect")?.value || "mr";
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", "Tetrax-ASR-Open-764M-A19M");
    formData.append("language", selectedLang);
    formData.append("timestamps", "true");

    try {
      const response = await fetch("/v1/audio/transcriptions", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Transcription request failed");
      }

      const result = await response.json();
      displayFileTranscriptionResult(result);
    } catch (err) {
      alert(`Transcription Error: ${err.message}`);
    } finally {
      transcribeBtn.disabled = false;
      transcribeBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
          <path d="M8 5v14l11-7z"></path>
        </svg>
        Transcribe Audio
      `;
    }
  });
}

function displayFileTranscriptionResult(res) {
  const card = document.getElementById("fileResultCard");
  const fullText = document.getElementById("fileFullText");
  const segmentsContainer = document.getElementById("fileSegments");
  
  document.getElementById("metaDuration").textContent = `${res.duration}s`;
  document.getElementById("metaLatency").textContent = `${res.processing_time || 0}s`;
  document.getElementById("metaRTF").textContent = res.rtf !== null ? `${res.rtf}x` : "-";
  document.getElementById("metaModel").textContent = res.model;

  fullText.textContent = res.text || "(No speech detected)";

  segmentsContainer.innerHTML = "";
  if (res.segments && res.segments.length > 0) {
    res.segments.forEach(seg => {
      const row = document.createElement("div");
      row.className = "caption-segment";
      row.innerHTML = `
        <span class="caption-time">[${formatSeconds(seg.start)} - ${formatSeconds(seg.end)}]</span>
        <span>${escapeHtml(seg.text)}</span>
      `;
      segmentsContainer.appendChild(row);
    });
  }

  card.style.display = "block";
  card.scrollIntoView({ behavior: "smooth" });
}

// System Status Polling
async function checkSystemStatus() {
  try {
    const res = await fetch("/v1/health");
    if (!res.ok) return;
    const data = await res.json();
    
    const pill = document.getElementById("systemHealthPill");
    const statusText = document.getElementById("systemHealthText");
    const dot = pill.querySelector(".status-dot");
    const modelBadge = document.getElementById("activeModelBadge");

    if (data.status === "healthy") {
      dot.classList.remove("inactive");
      statusText.textContent = "Engine Ready";
    } else {
      dot.classList.add("inactive");
      statusText.textContent = "Degraded";
    }

    if (data.is_finetuned_adapter) {
      modelBadge.innerHTML = `<span class="model-pill">Tetrax-ASR-Open-764M-A19M (मराठी • हिंदी Active)</span>`;
    } else {
      modelBadge.innerHTML = `<span class="model-pill" style="color:#94a3b8; border-color:rgba(255,255,255,0.1)">Whisper Base (Awaiting LoRA)</span>`;
    }
  } catch (e) {
    console.warn("Could not check health:", e);
  }
}

// Helpers
function formatSeconds(sec) {
  if (sec === undefined || sec === null) return "00:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 10);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms}`;
}

function escapeHtml(text) {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return text.replace(/[&<>"']/g, m => map[m]);
}
