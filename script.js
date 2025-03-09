// DOM elements
const uploadArea = document.getElementById("uploadArea");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const predictionsList = document.getElementById("predictions");
const loader = document.getElementById("loader");

const modelUri = 'mobilenetv3-large/model.json';
const labelUri = 'mobilenetv3-large/imagenet-simple-labels.json';

// Load MobileNetV3 model
let model;
async function loadModel() {
  loader.style.display = "block"; // Show loader
  try {
    const response = await fetch(modelUri);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    model = await tf.loadGraphModel(modelUri);
    console.log("MobileNetV3 loaded!");
  } catch (error) {
    console.error("Loading failed:", error);
  }
  loader.style.display = "none"; // Hide loader
  if (model) {
    uploadArea.classList.remove("disabled");
    imageInput.disabled = false;
  }
}
loadModel();

// Drag-and-drop functionality
uploadArea.addEventListener("dragover", (e) => {
  if (!uploadArea.classList.contains("disabled")) {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  }
});
uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});
uploadArea.addEventListener("drop", (e) => {
  if (!uploadArea.classList.contains("disabled")) {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    processImage(file);
  }
});
uploadArea.addEventListener("click", () => {
  if (!uploadArea.classList.contains("disabled")) {
    imageInput.click();
  }
});
imageInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) processImage(file);
});

// Display predictions with probabilities, colors, and dynamic bar widths
function displayPredictions(predictions) {
  predictionsList.innerHTML = "";
  const topThree = predictions.slice(0, 3);
  topThree.forEach(pred => {
    const li = document.createElement("li");
    const span = document.createElement("span");
    const label = pred.className || `Class ${predictions.indexOf(pred)}`; // Fallback label
    const probability = (pred.probability * 100).toFixed(1);
    span.textContent = `${label}: ${probability}%`;

    if (probability >= 70) {
      li.classList.add("high-confidence");
    } else if (probability >= 30) {
      li.classList.add("medium-confidence");
    } else {
      li.classList.add("low-confidence");
    }

    li.style.setProperty("--bar-width", `${probability}%`);

    li.appendChild(span);
    predictionsList.appendChild(li);
  });

  const styleSheet = document.styleSheets[0];
  styleSheet.insertRule(`
    #predictions li::before {
      width: var(--bar-width);
    }
  `, styleSheet.cssRules.length);
}

// Process the uploaded image
async function processImage(file) {
  const imgUrl = URL.createObjectURL(file);
  preview.src = imgUrl;
  preview.style.display = "block";
  predictionsList.innerHTML = "<li>Analyzing image...</li>";

  preview.onload = async () => {
    const imgTensor = tf.browser.fromPixels(preview)
      .resizeNearestNeighbor([224, 224]) // MobileNetV3 expects 224x224
      .toFloat()
      .div(tf.scalar(255.0)) // Normalize to [0, 1]
      .expandDims();

    try {
      const predictions = await model.predict(imgTensor); // Raw output (logits or probabilities)
      const probabilities = tf.softmax(predictions).dataSync(); // Convert to probabilities if logits
      const labels = await fetch(labelUri).then(r => r.json());

      const topK = Array.from(probabilities)
        .map((prob, idx) => ({ probability: prob, className: labels[idx] || `Class ${idx}` }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 3);

      console.log("Predictions:", topK);
      displayPredictions(topK);
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      tf.dispose(imgTensor); // Clean up
    }
  };
}