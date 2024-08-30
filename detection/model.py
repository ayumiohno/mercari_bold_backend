import sys
import time
sys.path.insert(0, 'third_party/CenterNet2/')

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder
import numpy as np

text_encoder = build_text_encoder(pretrain=True)
text_encoder.eval()

cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file(
    "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
cfg.MODEL.DEVICE='cpu'

predictor = DefaultPredictor(cfg)

def get_detic_predictor(custom_vocabulary):
    metadata = MetadataCatalog.get(str(time.time()))
    metadata.thing_classes = custom_vocabulary.split(',')
    prompt='a '
    texts = [prompt + x for x in metadata.thing_classes]
    classifier = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    num_classes = len(metadata.thing_classes)
    reset_cls_test(predictor.model, classifier, num_classes)
    return predictor


# 正方形に切り取る
def crip_image(image, box: list[float]):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width < height:
        x1 -= (height - width) / 2
        x2 += (height - width) / 2
    else:
        y1 -= (width - height) / 2
        y2 += (width - height) / 2
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    cripped = image[int(y1):int(y2), int(x1):int(x2)]
    # max_dim = max(width, height)
    # padded_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    # padded_image[:cripped.shape[0], :cripped.shape[1]] = cripped
    # return padded_image 
    return cripped, [float(x1), float(y1), float(x2), float(y2)]

def plastic_bottle(image) -> int:
    res = plastic_bottle_predictor(image)
    box = res["instances"].pred_boxes.tensor[0].cpu().numpy()
    print(box)
    crip = crip_image(image, box)
    print(res)
    return len(res["instances"]), crip


def segmentize(image, prompt):
    predictor = get_detic_predictor(prompt)
    res = predictor(image)
    box = [0, 0, image.shape[1], image.shape[0]]
    if len(res["instances"]) != 0:
        box = res["instances"].pred_boxes.tensor[0].cpu().numpy()
    return crip_image(image, box)

# if __name__ == "__main__":
#     import cv2
#     img = cv2.imread("../water.jpg")
#     start = time.time()
#     print(predict(img, "plastic bottle"))
#     print(time.time() - start)
    
#     img = cv2.imread("../sungrass.jpg")
#     start = time.time()
#     print(predict(img, "sun glass"))
#     print(time.time() - start)
