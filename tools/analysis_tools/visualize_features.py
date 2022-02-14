import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import cv2

def draw_features(im, features, save_path):
    im = im.detach().cpu().numpy()[0]
    im = im.transpose(1, 2, 0)
    features = features.detach().cpu().numpy()[0]
    features = np.abs(features)
    gray = features.sum(0)
    gray /= gray.shape[0]
    gray = np.clip(gray, 0, 1)*255.
    gray = gray.astype(np.uint8)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    cv2.imwrite(save_path, heatmap)
