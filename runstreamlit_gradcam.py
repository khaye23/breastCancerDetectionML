import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision.transforms import functional as TF
import numpy as np

# =========================
# 1. DEVICE SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# =========================
# 2. MODEL LOADING
# =========================
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1  # binary segmentation: lesion vs background
    ).to(device)
    model.load_state_dict(torch.load("best_seg_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# =========================
# 3. IMAGE PREPROCESSING
# =========================
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    return transform(image)

# =========================
# 4. PREDICTION FUNCTION
# =========================
def predict_mask(model, img_tensor):
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_tensor)
        mask = torch.sigmoid(pred).cpu().squeeze(0).squeeze(0).numpy()
    return mask

# =========================
# 5. SIMPLE MALIGNANCY PREDICTION
# =========================
def predict_malignancy(mask, threshold=0.3):
    """
    Simple heuristic: average intensity of predicted mask
    - >= threshold → malignant
    - < threshold → benign
    """
    score = mask.mean()
    if score >= threshold:
        return "Malignant (cancerous)", score
    else:
        return "Benign (non-cancerous.)", score

# =========================
# 6. STREAMLIT APP
# =========================
st.title("🩺 Mammogram Tumor Detection and Malignancy Prediction")
st.markdown("Upload a mammogram image, the app predicts tumor regions and estimates if the lesion is benign or malignant.")

uploaded_file = st.file_uploader("Upload a mammogram image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_tensor = preprocess_image(image)

    st.subheader("Prediction")
    with st.spinner("Running tumor detection..."):
        mask = predict_mask(model, img_tensor)
        diagnosis, score = predict_malignancy(mask)

        # Visualization
        img_vis = TF.to_pil_image(img_tensor)

        fig, axes = plt.subplots(1,2, figsize=(10,5))
        axes[0].imshow(img_vis)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(img_vis)
        axes[1].imshow(mask, cmap="jet", alpha=0.5)
        axes[1].set_title(f"Predicted Lesion Mask\n{diagnosis} (score={score:.2f})")
        axes[1].axis("off")

        st.pyplot(fig)

    st.success(f" Estimated: **{diagnosis}**")
    st.markdown("""
    **Interpretation:**  
    - Red/Yellow areas → High likelihood of lesion  
    - Blue/transparent → Background / no lesion  
    - The app provides a **simple estimate** of malignancy based on lesion prominence.
    """)

else:
    st.info("Please upload a mammogram image to start.")
