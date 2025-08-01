from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import shutil
import os
import tempfile
import time
import logging

import config
from model_loader import load_models
from image_processing import process_file
from utils import to_base64

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

# Ensure upload folder exists
os.makedirs(config.UPLOAD_FOLDER_PATH, exist_ok=True)
logging.info(f"Upload folder ready: {config.UPLOAD_FOLDER_PATH}")

# Load models
logging.info("----- Initializing ANPR Application - Loading Models -----")
try:
    plate_detection_model, char_seg_model, char_recog_model, device, ocr_font_path = load_models()
    models_loaded = all([plate_detection_model, char_seg_model, char_recog_model])
except Exception as e:
    logging.error(f"Model loading failed: {e}", exc_info=True)
    plate_detection_model, char_seg_model, char_recog_model, device, ocr_font_path = None, None, None, "cpu", None
    models_loaded = False

# FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
templates.env.filters["to_base64"] = to_base64
templates.env.globals.update(zip=zip)

@app.get("/", response_class=HTMLResponse, name="upload_file_route")
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    original_filename = file.filename
    _, file_extension = os.path.splitext(original_filename)
    file_extension = file_extension.lower()

    if file_extension not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(config.ALLOWED_EXTENSIONS)
        msg = f"Unsupported file type: {file_extension}. Allowed: {allowed}"
        logging.warning(msg)
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "error": msg
        })

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, dir=config.UPLOAD_FOLDER_PATH) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
            logging.info(f"Saved temp file: {temp_path}")

        if not models_loaded:
            msg = "Models are not loaded correctly. Cannot process."
            logging.error(msg)
            return templates.TemplateResponse("upload.html", {
                "request": request,
                "error": msg
            })

        start_time = time.time()
        results = process_file(
            temp_path,
            plate_detection_model,
            char_seg_model,
            char_recog_model,
            device,
            ocr_font_path
        )
        duration = time.time() - start_time
        logging.info(f"Processed {original_filename} in {duration:.2f}s. Plates found: {len(results)}")

        return templates.TemplateResponse("results.html", {
            "request": request,
            "results": results,
            "filename": original_filename
        })

    except Exception as e:
        logging.error(f"Processing failed: {e}", exc_info=True)
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "error": f"Error occurred during processing: {str(e)}"
        })

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logging.info(f"Removed temp file: {temp_path}")
            except Exception as e:
                logging.warning(f"Failed to delete temp file: {e}")

