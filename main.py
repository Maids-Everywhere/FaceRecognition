import tempfile
import cv2
import numpy as np
from fastapi import FastAPI, File
from fastapi.responses import FileResponse

from computer_vision import make_prediction

app = FastAPI()

@app.post("/inference")
async def inference(image: bytes = File(...)):
    predicted_image = make_prediction(np.fromstring(image, np.uint8))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_output_image:
        temp_output_image_path = temp_output_image.name
        cv2.imwrite(temp_output_image_path, predicted_image)
    return FileResponse(temp_output_image_path, media_type="image/jpeg")
