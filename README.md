# SAM Ticket Detection Service

This project uses Segment Anything locally to segment kitchen tickets and return
polygon coordinates. It supports the original Meta SAM ViT-B path and a faster
MobileSAM path behind the same FastAPI contract. It does not use Azure Vision,
YOLO, or Detectron2. The production HTTP API does not use OCR and does not
require Tesseract.

The detector keeps the current automatic-mask behavior, polygon extraction, and
light paper-region filtering. It does not force rectangles, so tilted, partial,
hanging, cropped, and irregular tickets can still be returned.

## Project Structure

```text
app.py              FastAPI HTTP service
ticket_detector.py  reusable SAM detection and filtering logic
detect_tickets.py   local/manual CLI test script
models/             SAM checkpoint location
requirements.txt    Python dependencies
requirements-mobilesam.txt optional MobileSAM dependency
requirements-ocr.txt optional OCR dependency for local CLI debugging only
README.md           setup and usage
```

## Windows Local Setup

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

Install CPU PyTorch, then the project dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the SAM ViT-B checkpoint:

```powershell
New-Item -ItemType Directory -Force .\models
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile .\models\sam_vit_b_01ec64.pth
```

Optional MobileSAM setup:

MobileSAM is installed from the official
[MobileSAM repository](https://github.com/ChaoningZhang/MobileSAM) and uses the
`vit_t` checkpoint.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-mobilesam.txt
Invoke-WebRequest -Uri https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt -OutFile .\models\mobile_sam.pt
```

## macOS/Linux Server Setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch and dependencies:

```bash
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

Do not install Tesseract for the server. The FastAPI service only loads SAM and
returns ticket polygons as JSON.

Download the SAM ViT-B checkpoint:

```bash
mkdir -p models
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o models/sam_vit_b_01ec64.pth
```

Optional MobileSAM setup:

MobileSAM is installed from the official
[MobileSAM repository](https://github.com/ChaoningZhang/MobileSAM) and uses the
`vit_t` checkpoint.

```bash
python -m pip install -r requirements-mobilesam.txt
curl -L https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt -o models/mobile_sam.pt
```

For a Mac mini server, CPU mode is supported explicitly with `SAM_DEVICE=cpu`.

## Running Locally

Use the CLI when you want annotated images, JSON files, optional crops, or
prompt-click testing:

```powershell
.\.venv\Scripts\python.exe detect_tickets.py "Test Images\IMG_2437.jpeg" --mode auto --output-dir sam_outputs
```

Recommended CPU-friendly settings:

```powershell
.\.venv\Scripts\python.exe detect_tickets.py "Test Images\IMG_2437.jpeg" --mode auto --output-dir sam_outputs --max-dim 768 --points-per-side 12 --points-per-batch 4
```

MobileSAM CLI test:

```powershell
.\.venv\Scripts\python.exe detect_tickets.py "Test Images\IMG_2437.jpeg" --backend mobilesam --mode auto --output-dir sam_outputs --max-dim 768 --points-per-side 12 --points-per-batch 4
```

Prompt mode is still available for manual testing:

```powershell
.\.venv\Scripts\python.exe detect_tickets.py "Test Images\IMG_2437.jpeg" --mode prompt --points "900,1200;1750,1200" --output-dir sam_prompt_outputs
```

The CLI saves annotated images and JSON. The HTTP API does not save images,
crops, output folders, or OCR results.

## Running As A Server

Set an API key and start Uvicorn:

```powershell
$env:SAM_API_KEY = "change-me"
$env:MODEL_BACKEND = "sam"
$env:SAM_DEVICE = "cpu"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

For MobileSAM on older CPUs or low-cost cloud instances:

```powershell
$env:SAM_API_KEY = "change-me"
$env:MODEL_BACKEND = "mobilesam"
$env:SAM_DEVICE = "cpu"
$env:SAM_MAX_DIM = "768"
$env:SAM_POINTS_PER_SIDE = "12"
$env:SAM_POINTS_PER_BATCH = "4"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

macOS:

```bash
export SAM_API_KEY="change-me"
export MODEL_BACKEND=sam
export SAM_DEVICE=cpu
source .venv/bin/activate
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

The selected model is loaded once at startup and reused for requests. Startup
logs include model load time. Each `/detect-tickets` success log includes
image decode, resize/preprocess, model inference, mask filtering, and total
request time.
OCR is not initialized or invoked by the server.

## API

`POST /detect-tickets` decodes the input image, runs ticket detection with the
selected backend, and returns JSON polygons. It does not run OCR.

### Health

```bash
curl http://localhost:8000/health
```

### Detect From Multipart Upload

```bash
curl -X POST http://localhost:8000/detect-tickets \
  -H "x-api-key: change-me" \
  -F "image=@Test Images/IMG_2437.jpeg"
```

PowerShell:

```powershell
curl.exe -X POST http://localhost:8000/detect-tickets `
  -H "x-api-key: change-me" `
  -F "image=@Test Images\IMG_2437.jpeg"
```

### Detect From Image URL

```bash
curl -X POST http://localhost:8000/detect-tickets \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"imageUrl":"https://example.com/tickets.jpg"}'
```

Bearer auth is also accepted:

```bash
curl -X POST http://localhost:8000/detect-tickets \
  -H "Authorization: Bearer change-me" \
  -F "image=@Test Images/IMG_2437.jpeg"
```

## API Response

The API returns JSON only, using original image coordinates:

```json
{
  "imageWidth": 3984,
  "imageHeight": 1882,
  "ticketCount": 4,
  "tickets": [
    {
      "id": 1,
      "score": 0.92274,
      "bbox": {
        "x": 0,
        "y": 57,
        "width": 1023,
        "height": 1137
      },
      "polygon": [
        { "x": 0, "y": 57 },
        { "x": 986, "y": 99 },
        { "x": 1022, "y": 1193 }
      ]
    }
  ]
}
```

Tickets are sorted left-to-right. Polygon points keep the detector's current
integer coordinate precision.

## Environment Variables

```text
SAM_API_KEY                    required for POST /detect-tickets
MODEL_BACKEND                  default: sam; use sam or mobilesam
SAM_CHECKPOINT                 default: models/sam_vit_b_01ec64.pth
SAM_MODEL_TYPE                 default: vit_b
MOBILESAM_CHECKPOINT           default: models/mobile_sam.pt
MOBILESAM_MODEL_TYPE           default: vit_t
SAM_DEVICE                     default: auto; use cpu for CPU-only servers
SAM_MAX_IMAGE_BYTES            default: 26214400
SAM_ALLOW_IMAGE_URL            default: true
SAM_IMAGE_URL_TIMEOUT_SECONDS  default: 15
SAM_MAX_DIM                    default: 1024
SAM_POINTS_PER_SIDE            default: 16
SAM_POINTS_PER_BATCH           default: 8
SAM_SAME_WIDTH_TOLERANCE       default: 0.0
SAM_REJECT_HIGH_DETECTIONS     default: true
SAM_MIN_BBOX_CENTER_Y_RATIO    default: 0.25
```

Most production SAM filtering thresholds can also be overridden with `SAM_...`
variables that match the names in `DetectorConfig` inside `ticket_detector.py`.

`MODEL_BACKEND=sam` uses `SAM_CHECKPOINT` and `SAM_MODEL_TYPE`.
`MODEL_BACKEND=mobilesam` uses `MOBILESAM_CHECKPOINT` and
`MOBILESAM_MODEL_TYPE`.

## Optional Local OCR Debugging

OCR is not part of the production API. For local CLI experiments only, install
the optional Python OCR dependency and a system Tesseract package separately:

```bash
python -m pip install -r requirements-ocr.txt
python detect_tickets.py "Test Images/IMG_2437.jpeg" --mode auto --ocr-min-confidence 10
```

The normal production path is:

1. .NET/Azure backend uploads an image or sends `imageUrl`.
2. This SAM service returns ticket polygons as JSON.
3. The tablet or web app draws those polygons on the original image for user
   confirmation.
