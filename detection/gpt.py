import os
from openai import OpenAI
import base64
import cv2

client = OpenAI(
    # This is the default and can be omitted
    api_key="",
)

def get_title(image):
    base64_image = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    # {"type": "text", "text": f"Explain the title and price of the product ({product_type}) in the image to sell in Mercari US. " + "Answer only in the format title:{}, price:{}."},
                    # {"type": "text", "text": f"List products in the image. Only list the general name and box coordinates. " + "Answer and list only in the format [{general_name:{str}, price:{float}}, ...]" + " Sort by center of screen and proximity to camera."},
                    # {"type": "text", "text": f"List products in the image. Only list the general name and predicted price. " + "Answer and list only in the format [{general_name:{str}, price:{float}}, ...]" + " Sort by center of screen and proximity to camera."},
                    {"type": "text", "text": "Explain a product in center of screen and proximate to camera. Only answer the general name(str, noun) within 15 characters."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

def get_title_and_price(image, product_type):
    base64_image = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Explain the title(in str) and price(in float) of the product ({product_type}) in the image to sell in Mercari US. " + "Answer only in the format {title}, {price}."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    img = cv2.imread("../sungrass.jpg")
    # img = cv2.imread("../water.jpg")
    print(get_title_and_price(img,  "sun glass"))
