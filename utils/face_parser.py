import os
import cv2

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from PIL import Image

# pip install transformers==4.37.0
image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")


def face_seg_hf(img, net, cropped_size: int):
    device = next(net.parameters()).device
    face_area = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17]
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inputs = image_processor(images=pil_image, return_tensors="pt").to(device)
    outputs = net(**inputs)
    logits = outputs.logits  # shape (1, num_labels, ~height/4, ~width/4)
    predicted_classes = torch.argmax(logits, dim=1, keepdim=True)  # (1, 1, height/4, width/4)

    mask = torch.zeros_like(predicted_classes, dtype=torch.float)
    for category in face_area:
        mask[predicted_classes == category] = 1.0

    mask_resized = TF.resize(mask, size=[cropped_size, cropped_size], interpolation=InterpolationMode.NEAREST)

    return mask_resized
