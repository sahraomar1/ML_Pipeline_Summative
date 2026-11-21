const byId = (id) => document.getElementById(id);
const percent = (value) => (value * 100).toFixed(1) + "%";

// Call the API and return the answer. Show the error if it fails.
async function callApi(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Request failed");
  return data;
}

// ---------- 1. Model status ----------

function formatUptime(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m ${seconds % 60}s`;
}

async function refreshStatus() {
  try {
    const health = await callApi("/health");
    byId("status").textContent = health.status;
    byId("uptime").textContent = formatUptime(health.uptime_seconds);
    byId("last-trained").textContent = new Date(health.model_last_trained).toLocaleString();
    byId("waiting").textContent = health.new_uploads_waiting;
  } catch (error) {
    byId("status").textContent = "down";
  }
}

// ---------- 2. Predict ----------

function showPreview(file) {
  byId("preview").src = URL.createObjectURL(file);
}

function showPrediction(result) {
  const rows = Object.entries(result.probabilities)
    .map(([name, value]) => `<li>${name}: ${percent(value)}</li>`)
    .join("");
  byId("predict-result").innerHTML = `
    <div class="label ${result.label}">${result.label}</div>
    <p>Confidence: ${percent(result.confidence)}</p>
    <p><strong>Next step:</strong> ${result.claim_route}</p>
    <ul>${rows}</ul>`;
}

async function predict() {
  const file = byId("predict-file").files[0];
  if (!file) return alert("Choose a photo first.");
  showPreview(file);
  byId("predict-result").textContent = "Predicting...";
  const form = new FormData();
  form.append("file", file);
  try {
    showPrediction(await callApi("/predict", { method: "POST", body: form }));
  } catch (error) {
    byId("predict-result").innerHTML = `<p class="error">${error.message}</p>`;
  }
}

// ---------- 3. Data insights ----------

const CLASS_COLORS = ["#2e7d32", "#f0a202", "#c62828"];

function drawBarChart(canvasId, title, labels, values) {
  new Chart(byId(canvasId), {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: CLASS_COLORS }] },
    options: { plugins: { legend: { display: false }, title: { display: true, text: title } } },
  });
}

// Write one sentence for each feature.
function writeStory(insights) {
  const classes = Object.keys(insights);
  const sentences = {
    count: "Number of photos",
    brightness: "Brightness",
    contrast: "Contrast",
    edges: "Edge strength",
  };
  byId("story").innerHTML = Object.entries(sentences)
    .map(([feature, name]) => {
      const sorted = [...classes].sort((a, b) => insights[b][feature] - insights[a][feature]);
      const top = sorted[0];
      const bottom = sorted[sorted.length - 1];
      return `<li><strong>${name}:</strong> highest for ${top} (${insights[top][feature]}), lowest for ${bottom} (${insights[bottom][feature]}).</li>`;
    })
    .join("");
}

async function loadInsights() {
  const insights = await callApi("/insights");
  const classes = Object.keys(insights);
  const valuesOf = (feature) => classes.map((name) => insights[name][feature]);
  drawBarChart("chart-count", "Number of training photos", classes, valuesOf("count"));
  drawBarChart("chart-brightness", "Average brightness (0-255)", classes, valuesOf("brightness"));
  drawBarChart("chart-contrast", "Average contrast", classes, valuesOf("contrast"));
  drawBarChart("chart-edges", "Average edge strength", classes, valuesOf("edges"));
  writeStory(insights);
}

// ---------- 4. Upload and retrain ----------

async function uploadImages() {
  const files = byId("upload-files").files;
  if (files.length === 0) return alert("Choose some photos first.");
  const form = new FormData();
  form.append("label", byId("upload-label").value);
  for (const file of files) form.append("files", file);
  byId("upload-result").textContent = "Uploading...";
  try {
    const result = await callApi("/upload", { method: "POST", body: form });
    byId("upload-result").textContent =
      `Saved ${result.saved} images. ${result.new_uploads_waiting} images are waiting for retraining.`;
    if (result.auto_retrain_started) watchRetraining();
    refreshStatus();
  } catch (error) {
    byId("upload-result").textContent = error.message;
  }
}

async function startRetraining() {
  try {
    await callApi("/retrain", { method: "POST" });
    watchRetraining();
  } catch (error) {
    byId("retrain-result").textContent = error.message;
  }
}

// Check every 3 seconds until retraining is done.
function watchRetraining() {
  byId("retrain-button").disabled = true;
  byId("retrain-result").textContent = "Retraining is running...";
  const timer = setInterval(async () => {
    const run = await callApi("/retrain/status");
    if (run.status === "running") return;
    clearInterval(timer);
    byId("retrain-button").disabled = false;
    byId("retrain-result").textContent =
      `Retraining ${run.status}. ${run.message} (${run.image_count} images used)`;
    refreshStatus();
  }, 3000);
}

// ---------- 5. Evaluate ----------

function confusionTable(result) {
  const header = result.class_names.map((name) => `<th>predicted ${name}</th>`).join("");
  const rows = result.confusion_matrix
    .map((row, i) => `<tr><th>actual ${result.class_names[i]}</th>${row.map((n) => `<td>${n}</td>`).join("")}</tr>`)
    .join("");
  return `<table><tr><th></th>${header}</tr>${rows}</table>`;
}

function showEvaluation(result) {
  byId("evaluate-result").innerHTML = `
    <div class="tiles">
      <div class="tile"><span>Accuracy</span><strong>${percent(result.accuracy)}</strong></div>
      <div class="tile"><span>Precision</span><strong>${percent(result.precision)}</strong></div>
      <div class="tile"><span>Recall</span><strong>${percent(result.recall)}</strong></div>
      <div class="tile"><span>F1 score</span><strong>${percent(result.f1_score)}</strong></div>
      <div class="tile"><span>Loss</span><strong>${result.loss.toFixed(3)}</strong></div>
    </div>
    ${confusionTable(result)}`;
}

async function evaluateModel() {
  byId("evaluate-result").textContent = "Evaluating... this can take a minute.";
  try {
    showEvaluation(await callApi("/evaluate"));
  } catch (error) {
    byId("evaluate-result").innerHTML = `<p class="error">${error.message}</p>`;
  }
}

// ---------- Start ----------

byId("predict-button").addEventListener("click", predict);
byId("upload-button").addEventListener("click", uploadImages);
byId("retrain-button").addEventListener("click", startRetraining);
byId("evaluate-button").addEventListener("click", evaluateModel);

refreshStatus();
setInterval(refreshStatus, 5000);
loadInsights();