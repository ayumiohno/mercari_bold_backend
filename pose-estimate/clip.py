from transformers import CLIPProcessor, CLIPModel
import cv2

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def clip_inference(image, texts: list[str]) -> list[float]:
    inputs = processor(text=texts, images=image,
                       return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs.tolist()[0]
