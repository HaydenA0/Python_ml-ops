const imageInput = document.getElementById("image-input");
const predictForm = document.getElementById("predict-form");
const previewImage = document.getElementById("image-preview");
const previewPlaceholder = document.getElementById("preview-placeholder");
const fileNameLabel = document.getElementById("file-name");
const apiStatus = document.getElementById("api-status");
const formMessage = document.getElementById("form-message");
const submitButton = document.getElementById("submit-button");
const resultCard = document.getElementById("result-card");
const emptyState = document.getElementById("empty-state");
const predictedClass = document.getElementById("predicted-class");
const predictedProbability = document.getElementById("predicted-probability");
const appliedThreshold = document.getElementById("applied-threshold");
const requestId = document.getElementById("request-id");
const modelVersionLabel = document.getElementById("model-version");
const latencyMs = document.getElementById("latency-ms");
const device = document.getElementById("device");
const preprocessing = document.getElementById("preprocessing");
const baseProbsSection = document.getElementById("base-probs-section");
const baseProbsGrid = document.getElementById("base-probs-grid");
const clinicalSection = document.getElementById("clinical-section");
const clinicalGrid = document.getElementById("clinical-grid");
const thresholdInput = document.getElementById("threshold");
const thresholdValue = document.getElementById("threshold-value");

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = thresholdInput.value;
});

async function fetchHealth() {
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    apiStatus.textContent = payload.status === "ok" ? "API Ready" : "Model Missing";
    apiStatus.classList.toggle("status-error", payload.status !== "ok");
    modelVersionLabel.textContent = payload.model_version || "Unknown model version";
  } catch (error) {
    apiStatus.textContent = "API Unreachable";
    apiStatus.classList.add("status-error");
  }
}

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = () => {
    previewImage.src = reader.result;
    previewImage.classList.add("visible");
    previewPlaceholder.classList.add("hidden");
    fileNameLabel.textContent = file.name;
  };
  reader.readAsDataURL(file);
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const [, base64] = String(reader.result).split(",");
      resolve(base64);
    };
    reader.onerror = () => reject(new Error("Unable to read the selected file."));
    reader.readAsDataURL(file);
  });
}

function setLoadingState(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Analyzing..." : "Analyze Image";
}

function renderBaseProbs(payload) {
  if (!payload.base_model_probabilities) {
    baseProbsSection.classList.add("hidden");
    return;
  }
  baseProbsSection.classList.remove("hidden");
  baseProbsGrid.innerHTML = "";
  const entries = Object.entries(payload.base_model_probabilities);
  const maxVal = Math.max(...entries.map(([, v]) => v));
  for (const [name, prob] of entries) {
    const pct = (prob * 100).toFixed(1);
    const bar = document.createElement("div");
    bar.className = "prob-bar";
    bar.innerHTML = `
      <span class="prob-bar-label">${name}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width: ${pct}%"></div>
      </div>
      <span class="prob-bar-value">${pct}%</span>
    `;
    baseProbsGrid.appendChild(bar);
  }
}

function renderClinicalMetadata(payload) {
  if (!payload.clinical_metadata) {
    clinicalSection.classList.add("hidden");
    return;
  }
  clinicalSection.classList.remove("hidden");
  clinicalGrid.innerHTML = "";
  const rows = [
    { label: "Age", value: payload.clinical_metadata.age ?? "-" },
    { label: "Age (normalized)", value: payload.clinical_metadata.age_norm ?? "-" },
    { label: "Sex", value: payload.clinical_metadata.sex ?? "-" },
    { label: "Sex (encoded)", value: payload.clinical_metadata.sex_enc ?? "-" },
    { label: "Position", value: payload.clinical_metadata.position ?? "-" },
    { label: "Position (encoded)", value: payload.clinical_metadata.pos_enc ?? "-" },
  ];
  for (const { label, value } of rows) {
    const item = document.createElement("div");
    item.className = "clinical-item";
    item.innerHTML = `<span class="clinical-item-label">${label}</span><span class="clinical-item-value">${value}</span>`;
    clinicalGrid.appendChild(item);
  }
}

function renderResult(payload) {
  predictedClass.textContent = payload.predicted_class;
  predictedProbability.textContent = `${(payload.probability * 100).toFixed(2)}%`;
  appliedThreshold.textContent = Number(payload.threshold).toFixed(2);
  requestId.textContent = payload.request_id;
  modelVersionLabel.textContent = payload.model_version;
  latencyMs.textContent = `${payload.latency_ms.toFixed(1)} ms`;
  device.textContent = payload.device || "-";
  preprocessing.textContent = payload.preprocessing?.resized_to
    ? `${payload.preprocessing.resized_to}` : "-";
  renderBaseProbs(payload);
  renderClinicalMetadata(payload);
  resultCard.classList.remove("hidden");
  emptyState.classList.add("hidden");
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) return;
  showPreview(file);
  formMessage.textContent = "";
});

predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = imageInput.files?.[0];
  if (!file) {
    formMessage.textContent = "Select an image before running inference.";
    return;
  }

  setLoadingState(true);
  formMessage.textContent = "Sending image to the model...";

  try {
    const imageBase64 = await readFileAsBase64(file);
    const body = {
      image_base64: imageBase64,
      threshold: thresholdInput.value === "" ? null : Number(thresholdInput.value),
    };
    const age = document.getElementById("age").value;
    const sex = document.getElementById("sex").value;
    const position = document.getElementById("position").value;
    if (age) body.age = Number(age);
    if (sex) body.sex = sex;
    if (position) body.position = position;

    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const payload = await response.json();
    if (!response.ok) {
      const detail = payload.detail || "Prediction request failed.";
      throw new Error(detail);
    }

    renderResult(payload);
    formMessage.textContent = "Prediction completed successfully.";
  } catch (error) {
    formMessage.textContent = error.message;
  } finally {
    setLoadingState(false);
  }
});

fetchHealth();
