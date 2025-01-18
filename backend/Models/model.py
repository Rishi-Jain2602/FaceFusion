import os
import gdown
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))


def download_model_from_gdrive(file_id, destination_path="models/inswapper_128.onnx"):
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    gdrive_url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(destination_path):
        print(f"Downloading model from Google Drive (ID: {file_id})...")
        gdown.download(gdrive_url, destination_path, quiet=False)
        print(f"Model downloaded successfully and saved to {destination_path}.")
    else:
        print(f"Model already exists at {destination_path}.")

file_id = "1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF"
download_model_from_gdrive(file_id)
swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx')
