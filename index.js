let model;
let globalIndex = 0;
let models = [];
let datas = [];
let data = [];

function indexToPunch(index) {
  switch (index) {
    case 1:
      return "Straight";
    case 2:
      return "Hook";
    case 3:
      return "Body";
    case 4:
      return "UpperCut";
    default:
      return "None";
  }
}

async function fetchStorage() {
  const url =
    "https://raw.githubusercontent.com/punch-boxing/punch-assets/main/";
  const selector = document.getElementById("selector");

  try {
    // 3 or 5 D
    // 1 to 8 L
    // 5 to 20 W
    for (let i = 3; i <= 5; i += 2) {
      for (let j = 1; j <= 8; j++) {
        for (let k = 5; k <= 20; k += 5) {
          const modelUrl = `${url}models/GRU_${i}D_${j}L_${k}W/model.json`;
          const response = await fetch(modelUrl);
          if (!response.ok) {
            console.error(
              `Failed to fetch model ${i}_${j}_${k}:`,
              response.statusText
            );
            continue;
          }
          const modelData = await response.json();
          console.log(`Model ${i}_${j}_${k} data:`, modelData);
          models.push({
            name: `GRU_${i}D_${j}L_${k}W`,
            url: modelUrl,
          });
        }
      }
    }

    for (let i = 1; i <= 15; i++) {
      const dataUrl = `${url}datas/${i}.csv`;
      const response = await fetch(dataUrl);
      if (!response.ok) {
        console.error(`Failed to fetch data ${i}:`, response.statusText);
        continue;
      }
      const dataText = await response.text();
      const dataRows = dataText
        .trim()
        .split("\n")
        .map((row) => row.split(",").map((val) => val.trim()));
      datas.push({
        name: `Data ${i}`,
        url: dataUrl,
        rows: dataRows,
      });
    }
  } catch (error) {
    console.error("Error fetching models:", error);
  }
  selector.innerHTML = `<div>
        Select Model
        <select id="model-select"></select>
        <button id="load-model-btn" onclick="loadModel()">Load Model</button>
      </div>
      <div>
        Select Test Data
        <select id="data-select"></select>
        <button id="load-data-btn" onclick="loadData()">Load Data</button>
      </div>`;

  const modelSelect = document.getElementById("model-select");
  const dataSelect = document.getElementById("data-select");
  dataSelect.innerHTML = datas
    .map((data) => `<option value="${data.url}">${data.name}</option>`)
    .join("");
  modelSelect.innerHTML = models
    .map((model) => `<option value="${model.url}">${model.name}</option>`)
    .join("");
}

async function loadModel() {
  try {
    model = await tf.loadLayersModel(
      `${document.getElementById("model-select").value}`
    );

    // Compile the model for evaluation
    model.compile({
      optimizer: "adam",
      loss: "sparseCategoricalCrossentropy",
      metrics: ["accuracy"],
    });

    document.getElementById("model-status").innerHTML =
      "✅ Model loaded successfully!";

    // Display model information
    const modelInfo = document.getElementById("model-info");
    modelInfo.innerHTML = `
                    <h3>Model Information:</h3>
                    <p><strong>Input Shape:</strong> ${JSON.stringify(
                      model.inputs[0].shape
                    )}</p>
                    <p><strong>Output Shape:</strong> ${JSON.stringify(
                      model.outputs[0].shape
                    )}</p>
                    <p><strong>Total Parameters:</strong> ${model.countParams()}</p>
                    <p><strong>Model Type:</strong> ${
                      model.constructor.name
                    }</p>
                `;

    // Log model summary
    console.log("Model loaded:", model);
    model.summary();
  } catch (error) {
    document.getElementById("model-status").innerHTML =
      "❌ Error loading model: " + error.message;
    console.error("Error loading model:", error);
  }
}

async function loadData() {
  try {
    const dataUrl = document.getElementById("data-select").value;
    const response = await fetch(dataUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch data from ${dataUrl}`);
    }
    const dataText = await response.text();
    const dataRows = dataText
      .trim()
      .split("\n")
      .map((row) =>
        row.split(",").map((val) => {
          const num = parseFloat(val);
          return isNaN(num) ? val.trim() : num;
        })
      );
    data = dataRows;
    document.getElementById("data-status").innerHTML =
      "✅ Data loaded successfully!";
  } catch (error) {
    document.getElementById("data-status").innerHTML =
      "❌ Error loading data: " + error.message;
    console.error("Error loading data:", error);
  }
}

async function makePrediction() {
  if (!model) {
    alert("Model not loaded yet!");
    return;
  }

  try {
    globalIndex++;
    const prediction = model.predict(
      tf.tensor(data[globalIndex].slice(2, 7)).reshape([1, 1, 5])
    );
    const predictionData = await prediction.data();
    console.log(data[globalIndex]);
    predictedIndex = predictionData.indexOf(Math.max(...predictionData));
    const resultDiv = document.getElementById("prediction-result");
    resultDiv.innerHTML = `
                    <h3>Prediction Results:</h3>
                    <p><strong>Input Values:</strong></p>
                    <ul>
                        ${data[globalIndex]
                          .slice(2, 7)
                          .map((val) => `<li>${val.toFixed(6)}</li>`)
                          .join("")}
                    </ul>
                    <p><strong>Prediction Values:</strong></p>
                    <ul>
                        ${Array.from(predictionData)
                          .map(
                            (val, idx) =>
                              `<li>Class ${idx}: ${val.toFixed(6)} (${(
                                val * 100
                              ).toFixed(2)}%)</li>`
                          )
                          .join("")}
                    </ul>
                    <p><strong>Predicted Class:</strong> ${predictedIndex}</p>
                    <p><strong>Predicted Punch:</strong> ${indexToPunch(
                      predictedIndex
                    )}</p>
                    <p><strong>Real Data: ${data[globalIndex][7]}</strong></p>
                    <p><strong>Is Correct:</strong> ${
                      indexToPunch(predictedIndex) === data[globalIndex][7]
                        ? "✅ Yes"
                        : "❌ No"
                    }</p>
                `;
  } catch (error) {
    document.getElementById("prediction-result").innerHTML =
      "❌ Error making prediction: " + error.message;
    console.error("Error making prediction:", error);
  }
}

async function evaluateModel() {
  if (!model) {
    alert("Model not loaded yet!");
    return;
  }

  try {
    // Filter out rows with non-numeric values
    const filteredData = data.filter((row) => {
      if (typeof row !== "number") {
        return row === "Straight"
          ? 1
          : row === "Hook"
          ? 2
          : row === "Body"
          ? 3
          : row === "UpperCut"
          ? 4
          : 0;
      }
      return false;
    });

    // Evaluate model
    const evalResult = model.evaluate(
      tf.tensor(filteredData.map((row) => row.slice(2, 7))).reshape([-1, 1, 5]),
      tf.tensor(filteredData.map((row) => row[7]))
    );
    console.log("Evaluation results:", evalResult);
  } catch (error) {
    document.getElementById("prediction-result").innerHTML =
      "❌ Error evaluating model: " + error.message;
    console.error("Error evaluating model:", error);
  }
}

// Load the model when the page loads
window.onload = fetchStorage;
