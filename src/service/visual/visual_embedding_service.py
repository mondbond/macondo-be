from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import chromadb

# Load model + processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Init Chroma client
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="images")
