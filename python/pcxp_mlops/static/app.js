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

function renderResult(payload) {
  predictedClass.textContent = payload.predicted_class;
  predictedProbability.textContent = `${(payload.probability * 100).toFixed(2)}%`;
  appliedThreshold.textContent = Number(payload.threshold).toFixed(2);
  requestId.textContent = payload.request_id;
  modelVersionLabel.textContent = payload.model_version;
  resultCard.classList.remove("hidden");
  emptyState.classList.add("hidden");
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) {
    return;
  }
  showPreview(file);
  formMessage.textContent = "";
});

predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = imageInput.files?.[0];
  const thresholdInput = document.getElementById("threshold").value;
  if (!file) {
    formMessage.textContent = "Select an image before running inference.";
    return;
  }

  setLoadingState(true);
  formMessage.textContent = "Sending image to the model...";

  try {
    const imageBase64 = await readFileAsBase64(file);
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        image_base64: imageBase64,
        threshold: thresholdInput === "" ? null : Number(thresholdInput)
      })
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
