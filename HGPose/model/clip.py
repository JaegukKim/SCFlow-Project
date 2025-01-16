from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn

# CLIP 모델과 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def CLIP(obj_name):
    text = "a photo of a " + obj_name
    input = processor(text=text, return_tensors="pt", padding=True, truncation=True) 

    with torch.no_grad():
        text_feature = model.get_text_features(**input)

    return text_feature