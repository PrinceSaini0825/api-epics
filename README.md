# 🔥 Fire Detection API

A FastAPI-based REST API for fire detection using PyTorch CNN models — converted from Google Colab.

## Project Structure

```
fire-detection-api/
├── models/
│   ├── __init__.py
│   └── model.py          ← ModelA, ModelB, HybridFireDetector
├── api/
│   ├── __init__.py
│   └── main.py           ← FastAPI app + all endpoints
├── scripts/
│   └── train.py          ← Training script
├── weights/              ← Put your .pth files here (git-ignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Step 1 – Setup

```bash
# Clone / open in VS Code, then create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2 – Get your trained weights

**Option A — Re-train locally:**
```bash
python scripts/train.py \
  --data_dir /path/to/fire_dataset \
  --output_dir ./weights
```

Your dataset folder must follow this layout:
```
fire_dataset/
    fire_images/
        fire.1.png
        fire.2.png
        ...
    non_fire_images/
        non_fire.1.png
        ...
```

**Option B — Download from Google Drive (Colab weights):**
Download `best_ModelA.pth`, `best_ModelB.pth`, `best_Hybrid.pth` from your Google Drive
and place them in the `./weights/` folder.

---

## Step 3 – Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be live at: **http://localhost:8000**

Interactive docs (Swagger UI): **http://localhost:8000/docs**

---

## API Endpoints

### `GET /` — Health check
Returns API status and which models are loaded.

### `GET /models` — List models
```json
{ "available_models": ["model_a", "model_b", "hybrid"] }
```

### `POST /predict` — Predict from image file
Upload an image (jpg/png/webp) and get a prediction.

**Example (curl):**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/image.jpg" \
  -F "model=hybrid"
```

**Response:**
```json
{
  "model_used": "hybrid",
  "label": "fire_images",
  "is_fire": true,
  "confidence": 0.1872,
  "threshold": 0.5
}
```

### `POST /predict/base64` — Predict from base64 string
Useful for mobile apps that send images as base64.

**Request body:**
```json
{
  "image_base64": "<base64-encoded-image>",
  "model": "hybrid"
}
```

---

## Using the API in your app

### JavaScript / React Native / Next.js

**File upload:**
```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('model', 'hybrid');

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData,
});
const result = await response.json();
console.log(result.label, result.is_fire, result.confidence);
```

**Base64 (e.g. from camera roll):**
```javascript
const response = await fetch('http://localhost:8000/predict/base64', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image_base64: base64String,  // strip "data:image/jpeg;base64," prefix first
    model: 'hybrid',
  }),
});
const result = await response.json();
```

### Python (requests)
```python
import requests

with open("fire_image.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        data={"model": "hybrid"},
    )
print(r.json())
```

### Flutter / Dart
```dart
var request = http.MultipartRequest(
  'POST', Uri.parse('http://localhost:8000/predict'),
);
request.files.add(await http.MultipartFile.fromPath('file', imagePath));
request.fields['model'] = 'hybrid';
var response = await request.send();
var body = await response.stream.bytesToString();
print(jsonDecode(body));
```

---

## Deploying to production

For a public API (not just localhost), deploy with one of these:

| Platform     | Command / Notes |
|--------------|-----------------|
| **Railway**  | `railway up` — auto-detects Python, set `WEIGHTS_DIR` env var |
| **Render**   | Add as a Web Service, start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| **Docker**   | See below |
| **VPS / EC2**| Run behind nginx with gunicorn + uvicorn workers |

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
```bash
docker build -t fire-detection-api .
docker run -p 8000:8000 -v $(pwd)/weights:/app/weights fire-detection-api
```

---

## Notes on confidence score

The model outputs a sigmoid probability between 0 and 1.
- **Low value (< 0.5) → fire** (`fire_images` is class index 0)
- **High value (≥ 0.5) → non-fire**

The `is_fire` field in the response handles this logic for you.
