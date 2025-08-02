#  Nepali Licence Plate Recognition System(NLPR)

This is an Automatic Number Plate Recognition  system designed specifically for Nepali license plates. The system utilizes multiple machine learning models in a pipeline architecture to detect license plates, segment characters, and perform optical character recognition.

## 🔍 Overview

This project implements a complete ANPR pipeline:

1. **Plate Detection** - Detects license plates from images or video frames
2. **Segmentation** - Segments individual characters from the detected plate
3. **Character Recognition** - Recognizes the segmented characters

The system is deployed as a Fastapi web application for easy interaction.

## 📂 Project Structure

```
NLPR/
├── app/          # Fastapi web application
    ├── main.py           # Main Fastapi application
    ├── config.py        # Application configuration
    ├── model_loader.py  # Model loading utilities
    ├── image_processing.py  # Image processing pipeline
    ├── templates/       # HTML templates
    └── static/          # Static assets (CSS, JS, images)


```



### Prerequisites

1. Python 3.10 or higher


### Clone the Repository

```bash
git clone https://github.com/simalshakha/nepali-licence_plate_recognization.git
cd NLPR
```

### Install Dependencies with UV

```bash
pip install -r requirements.txt
```

This will install all dependencies defined in the `requirements.txt` file.

## 🏃 Running the Application

Start the Fastapi application:

```bash
cd api
uvicorn main:app --reload --port 5001
```

The web interface will be available at `http://127.0.0.1:5001/`

## 🔄 Pipeline Process
Model pipeline: `Plate Detection → Segmentation → Character recognization`
