// URL model ONNX
const MODEL_URL = './model3 copy.onnx';

// DOM Elements
const imageUpload = document.getElementById('imageUpload');
const uploadedImage = document.getElementById('uploadedImage');
const resultsDiv = document.getElementById('results');

// Variables
let model;

// Load the ONNX model
async function loadModel() {
    try {
        model = await ort.InferenceSession.create(MODEL_URL);
        console.log('Model loaded successfully');
        console.log('Input Names:', model.inputNames);
        console.log('Output Names:', model.outputNames);
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

// Preprocess image to match model input
function preprocessImage(image) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Set canvas size and draw image
    canvas.width = 177;
    canvas.height = 177;
    ctx.drawImage(image, 0, 0, 177, 177);

    // Extract image data and normalize
    const imageData = ctx.getImageData(0, 0, 177, 177);
    const { data } = imageData;
    const input = new Float32Array(3 * 177 * 177);

    for (let i = 0; i < data.length; i += 4) {
        const pixelIndex = i / 4;
        input[pixelIndex] = data[i] / 255.0; // R
        input[pixelIndex + 177 * 177] = data[i + 1] / 255.0; // G
        input[pixelIndex + 2 * 177 * 177] = data[i + 2] / 255.0; // B
    }

    return new ort.Tensor('float32', input, [1, 3, 177, 177]);
}

// Upload and classify image
function uploadAndClassifyImage() {
    const file = imageUpload.files[0];
    const reader = new FileReader();

    // Display uploaded image
    reader.onloadend = async function () {
        uploadedImage.src = reader.result;

        const image = new Image();
        image.src = reader.result;

        image.onload = async () => {
            resultsDiv.innerHTML = '<p>Classifying...</p>';

            try {
                const tensor = preprocessImage(image);

                // Use model input and output names
                const feeds = { [model.inputNames[0]]: tensor };
                const output = await model.run(feeds);

                const probabilities = output[model.outputNames[0]].data;
                const classNames = [
                    'anggur', 'apel', 'belimbing', 'jeruk', 'kiwi',
                    'mangga', 'nanas', 'pisang', 'semangka', 'stroberi'
                ];

                // Find class with the highest probability
                const maxIndex = probabilities.indexOf(Math.max(...probabilities));

                // Display classification results with updated taskbar and percentages
                resultsDiv.innerHTML = '';
                probabilities.forEach((prob, index) => {
                    resultsDiv.innerHTML += `
                        <div class="result">
                            <span>${classNames[index].toUpperCase()}</span>
                            <div class="bar-container">
                                <div class="bar" style="width: ${(prob * 100).toFixed(2)}%;"></div>
                            </div>
                            <span>${(prob * 100).toFixed(2)}%</span>
                        </div>
                    `;
                });

                // Output the highest probability and the class name
                resultsDiv.innerHTML += `
                    <h3>Most Likely Class:</h3>
                    <p>${classNames[maxIndex].toUpperCase()} with ${(probabilities[maxIndex] * 100).toFixed(2)}%</p>
                `;
            } catch (error) {
                console.error('Error classifying image:', error);
                resultsDiv.innerHTML = '<p>Error classifying image. Check console for details.</p>';
            }
        };
    };

    if (file) {
        reader.readAsDataURL(file);
    }
}

// Load model on page load
loadModel();

// Add event listener to classify image on file upload
imageUpload.addEventListener('change', uploadAndClassifyImage);
