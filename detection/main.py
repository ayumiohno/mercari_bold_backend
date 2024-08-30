from fastapi import FastAPI
import uvicorn
from fastapi import File, UploadFile
from PIL import Image
import io
import cv2
import numpy as np

from gpt import get_title_and_price, get_title
from model import segmentize
import json

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/detect")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
    
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        product = None
        for _ in range(5):
            try:
                product = get_title(image_cv)
                break
            except Exception as e:
                print(e)
        
        if not product:
            return {"message": "No products detected"}, 400
        print(product)
        cripped_image, box = segmentize(image_cv, product)

        title, price = None, None
        for i in range(5):
            try:
                res = get_title_and_price(image_cv, cripped_image)
                if "sorry" in res:
                    pass
                price = float(res.split(",")[-1])
                title = ",".join(res.split(",")[:-1])
                break
            except Exception as e:
                print(e)
        if price is None:
            price = 0.0
        if title is None:
            title = "Unknown"
        return {"title": title or product, "price": price, "box": box, "width": image_cv.shape[1], "height": image_cv.shape[0]}
    except Exception as e:
        print(e)
        return {"message": "There was an error uploading the file"}, 400
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000, log_level="debug")

