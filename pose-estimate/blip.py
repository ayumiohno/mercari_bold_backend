from transformers import AutoProcessor, BlipForQuestionAnswering
import cv2

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

def blip_inference(image, prompt: str) -> str:
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    result = processor.decode(outputs[0], skip_special_tokens=True)
    return result.lower()
