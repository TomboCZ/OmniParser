"""Batch CPU inference for Microsoft's MIT-licensed icon_detect_v3 detector.

Reads a JSON list of image filenames on stdin. Writes one list of xyxy pixel
boxes and confidence scores per image on stdout, in input order. Model and
decoding specification: microsoft/OmniParser-v2.0 model card, revision f55d075.
"""
import argparse
import json
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms


def detect(model, path):
    with Image.open(path) as image:
        width, height = image.size
        scale = 1280 / max(width, height)
        rgb = image.convert("RGB").resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (1280, 1280), (114, 114, 114))
    padded.paste(rgb)
    tensor = torch.from_numpy(np.asarray(padded).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
    with torch.inference_mode():
        outputs = model(tensor)
    boxes, scores = [], []
    for stride, logits, offsets in zip((8, 16, 32), outputs[::2], outputs[1::2]):
        confidence = logits[0, 0].sigmoid()
        y, x = torch.where(confidence > 0.05)
        dx1, dy1, dx2, dy2 = offsets[0, :, y, x]
        center_x, center_y = x + 0.5, y + 0.5
        boxes.append(torch.stack((center_x - dx1, center_y - dy1, center_x + dx2, center_y + dy2), 1) * stride / scale)
        scores.append(confidence[y, x])
    boxes, scores = torch.cat(boxes), torch.cat(scores)
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)
    return [{"box": boxes[i].tolist(), "score": round(float(scores[i]), 4)} for i in nms(boxes, scores, 0.45)[:80]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = torch.jit.load(args.weights, map_location="cpu").eval()
    json.dump([detect(model, path) for path in json.load(sys.stdin)], sys.stdout)
