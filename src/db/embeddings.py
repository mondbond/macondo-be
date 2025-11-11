from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def embed_image(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features[0].tolist()


def embed_text(text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features[0].tolist()


# CLIP transformer embedding for image or text
def clip_transformer_embedding(image: Image.Image = None, text: str = None):
    if image:
        return embed_image(image)
    elif text:
        return embed_text(text)
    else:
        return None
