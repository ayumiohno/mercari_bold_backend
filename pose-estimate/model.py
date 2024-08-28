from blip import blip_inference
from clip import clip_inference

def pose_inference(image) -> str:
    texts = ["They thumbs up.", "They are not posing."]
    probs = clip_inference(image, texts)
    if probs[0] > 0.9:
        print(f"thumbs up: {probs[0]} {probs[1]}")
        return f"thumbs up"
    prompt = "Are they shoulder to shoulder?"
    if blip_inference(image, prompt) == "yes":
        print("shoulder")
        return "shoulder"
    print("unknown")
    return "unknown"
