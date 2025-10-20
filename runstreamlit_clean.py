import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet18
import segmentation_models_pytorch as smp
from utils import preprocess_classification, preprocess_segmentation, overlay_mask

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.title("🩺 Breast Cancer Detection from Mammograms")

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classification model
    model_cls = resnet18(weights=None)
    model_cls.fc = nn.Linear(model_cls.fc.in_features, 2)
    cls_checkpoint = "/content/drive/MyDrive/mammogram_project/resnet18_cls_cbisd.pth"
    model_cls.load_state_dict(torch.load(cls_checkpoint, map_location=device))
    model_cls.to(device)
    model_cls.eval()

    # Segmentation model
    model_seg = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    seg_checkpoint = "/content/drive/MyDrive/mammogram_project/lesion_segmentation_unet.pth"
    model_seg.load_state_dict(torch.load(seg_checkpoint, map_location=device))
    model_seg.to(device)
    model_seg.eval()

    return model_cls, model_seg, device

model_cls, model_seg, device = load_models()

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader("Upload a Mammogram Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)

    # Convert grayscale to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # -------------------------
    # Classification
    # -------------------------
    img_cls_tensor = preprocess_classification(img).to(device)
    if img_cls_tensor.ndim == 3:
        img_cls_tensor = img_cls_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model_cls(img_cls_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    if pred_class.item() == 1:
        st.success(f"Cancerous (Confidence: {confidence.item()*100:.2f}%)")
    else:
        st.info(f"Non-cancerous (Confidence: {confidence.item()*100:.2f}%)")

    # -------------------------
    # Segmentation
    # -------------------------
    img_seg_tensor = preprocess_segmentation(img).to(device)
    if img_seg_tensor.ndim == 3:
        img_seg_tensor = img_seg_tensor.unsqueeze(0)

    with torch.no_grad():
        mask = torch.sigmoid(model_seg(img_seg_tensor)).squeeze().cpu().numpy()

    # Threshold mask for clear binary lesion
    mask_binary = (mask > 0.5).astype(np.uint8) * 255

    # Resize mask to original image size
    mask_resized = cv2.resize(mask_binary, (img.width, img.height), interpolation=cv2.INTER_NEAREST)

    # Overlay mask
    overlayed = overlay_mask(img, mask_resized)

    st.markdown("### 🩻 Tumor Segmentation Result")
    st.image(overlayed, caption="Segmentation Overlay", use_container_width=True)

    # Optional: show raw binary mask
    st.markdown("### Raw Binary Mask")
    st.image(mask_resized, caption="Mask", use_container_width=True)
