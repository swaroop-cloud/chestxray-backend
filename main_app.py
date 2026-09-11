from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from prediction import load_thresholds, predict

app = FastAPI(title="Chest X-Ray Multi-Label Classifier")

# Allow the Netlify frontend to call this API.
# Once you have your real Netlify URL, replace "*" with it, e.g.
# allow_origins=["https://your-site.netlify.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

THRESHOLDS_PATH = Path(__file__).parent.parent / "outputs" / "disease_thresholds.csv"
THRESHOLDS = load_thresholds(THRESHOLDS_PATH)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Chest X-ray classifier API is running."}


@app.post("/predict")
async def analyze_xray(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Please upload a PNG or JPEG image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return predict(image_bytes, THRESHOLDS)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
