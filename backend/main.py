import numpy as np
import cv2,shutil
import matplotlib.pyplot as plt
from Models.model import app,swapper
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from Img_gen.img import Image_Generator
from PIL import Image

app_fastapi = FastAPI()

@app_fastapi.post("/upload/")
async def upload_image(file:UploadFile = File(...),prompt:str=Form(...)):
  with open("uploaded_image.jpeg", "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)

  input1 = cv2.imread("uploaded_image.jpeg")

  image = Image_Generator(prompt)
  input2 = np.array(image)
  
  faces = app.get(input1)
  res = input1.copy()
  input_face = app.get(input2)
  input_face = input_face[0]
  for face in faces:
    res = swapper.get(res,face,input_face,paste_back = True)

  output_image_path = "Replaced_image.JPG"
  cv2.imwrite(output_image_path, res)

  return FileResponse(output_image_path, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000)