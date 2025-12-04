# src/gradcam_heatmap.py
import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def _find_target_layer(model):
    # try canonical SwinV2 layer
    try:
        return model.backbone.layers[-1].blocks[-1].norm2
    except Exception:
        # fallback to last module with weight
        for m in reversed(list(model.modules())):
            if hasattr(m, "weight") and getattr(m, "weight") is not None:
                return m
    return None

def generate_gradcam(model, input_tensor, target_cat=None, use_cuda=True):
    """
    input_tensor: single 1x3xHxW torch tensor (on device)
    returns: (grayscale_cam_uint8 HxW, overlay_bgr uint8)
    """
    device = input_tensor.device
    target_layer = _find_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda and device.type == "cuda")

    # if target not provided, pick predicted class of fracture head
    if target_cat is None:
        with torch.no_grad():
            out = model(input_tensor)
            probs = torch.softmax(out[0], dim=1)
            target_cat = int(torch.argmax(probs, dim=1).item())

    targets = [ClassifierOutputTarget(target_cat)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # HxW float 0..1
    grayscale_cam = np.clip(grayscale_cam, 0, 1)

    # convert tensor -> rgb image normalized 0..1
    img_np = input_tensor[0].detach().cpu().numpy().transpose(1,2,0)
    # unnormalize if normalized with ImageNet (assume it was normalized)
    # If input was normalized, caller should pass already unnormalized image, here we assume normalized and revert
    img = img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
    img = np.clip(img, 0, 1)

    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    return (grayscale_cam * 255).astype("uint8"), cam_bgr
