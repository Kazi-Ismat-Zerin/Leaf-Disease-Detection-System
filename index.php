<!DOCTYPE html>
<html lang="en">
<head>
    <title>Leaf Disease Detection</title>
    <style>
        body { font-family: sans-serif; text-align: center; padding: 50px; background: #f4f4f4; }
        .card { background: white; padding: 20px; border-radius: 10px; display: inline-block; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        #result { margin-top: 20px; font-weight: bold; color: #2e7d32; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Leaf Disease Detector</h2>
        <input type="file" id="imageInput" accept="image/*">
        <button onclick="uploadImage()">Analyze Leaf</button>
        <div id="result"></div>
    </div>

    <script>
        async function uploadImage() {
            const fileInput = document.getElementById('imageInput');
            const resultDiv = document.getElementById('result');
            
            if (fileInput.files.length === 0) return alert("Please select an image");

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            resultDiv.innerText = "Processing...";

            try {
                const response = await fetch('http://localhost:8000/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                resultDiv.innerHTML = Prediction: ${data.class} <br> Confidence: ${(data.confidence * 100).toFixed(2)}%;
            } catch (error) {
                resultDiv.innerText = "Error connecting to Backend API.";
            }
        }
    </script>
</body>
</html>