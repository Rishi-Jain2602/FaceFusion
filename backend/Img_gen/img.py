import requests,io
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv()
HUGGINGFACE_API = os.environ["HUGGINGFACE_API_TOKEN"]

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API}"}

def Image_Generator(query):
	payload = {"inputs":query}
	response = requests.post(API_URL, headers=headers, json=payload)
	image_bytes = response.content
	image = Image.open(io.BytesIO(image_bytes))
	return image

