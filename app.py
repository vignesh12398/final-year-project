import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="COVID-19 Lung CT Scan Image Segmentation",
    page_icon="🫁",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "attention_unet_model.keras",
        compile=False
    )

    return model


try:

    model = load_model()

    st.sidebar.success(
        "Model loaded successfully"
    )

except Exception as e:

    st.error(
        "Could not load the model."
    )

    st.exception(e)

    st.stop()


# =========================================================
# MAIN TITLE
# =========================================================

st.title(
    "🫁 COVID-19 Lung CT Scan Image Segmentation"
)

st.write(
    "Upload a CT scan image to segment and localize "
    "possible infection regions using the trained "
    "Attention U-Net model."
)


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload CT Image",
    type=["png", "jpg", "jpeg"]
)


# =========================================================
# IMAGE PROCESSING
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("L")

    original_image = np.array(image)


    # =====================================================
    # ORIGINAL CT IMAGE
    # =====================================================

    st.subheader(
        "1️⃣ Original CT Image"
    )

    st.image(
        original_image,
        clamp=True,
        width=500
    )


    # =====================================================
    # PREPROCESSING
    # =====================================================

    resized_image = cv2.resize(
        original_image,
        (128, 128)
    )

    resized_image = (
        resized_image.astype(np.float32) / 255.0
    )

    input_image = np.expand_dims(
        resized_image,
        axis=0
    )

    input_image = np.expand_dims(
        input_image,
        axis=-1
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    with st.spinner(
        "Analyzing CT image..."
    ):

        prediction = model.predict(
            input_image,
            verbose=0
        )

    prediction = prediction.squeeze()


    # =====================================================
    # CREATE BINARY MASK
    # =====================================================

    threshold = 0.05

    pred_mask = (
        prediction > threshold
    ).astype(np.uint8)


    # =====================================================
    # GROUND TRUTH MASK
    # =====================================================

    st.subheader(
        "2️⃣ Ground Truth Mask"
    )

    # Get uploaded file name
    file_name = uploaded_file.name

    # Ground truth directory
    ground_truth_dir = "ground_truth"

    ground_truth_path = os.path.join(
        ground_truth_dir,
        file_name
    )


    if os.path.exists(ground_truth_path):

        # Read ground truth
        ground_truth = cv2.imread(
            ground_truth_path,
            cv2.IMREAD_GRAYSCALE
        )

        # Resize to model output size
        ground_truth = cv2.resize(
            ground_truth,
            (128, 128),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert to binary mask
        ground_truth_mask = (
            ground_truth > 127
        ).astype(np.uint8)


        st.image(
            ground_truth_mask * 255,
            caption="Ground Truth Infection Mask",
            clamp=True,
            width=500
        )

        ground_truth_available = True

    else:

        st.warning(
            f"Ground truth mask not found for "
            f"'{file_name}'."
        )

        st.info(
            "Make sure the ground truth mask is stored "
            f"at: {ground_truth_path}"
        )

        ground_truth_available = False


    # =====================================================
    # PREDICTED INFECTION MASK
    # =====================================================

    st.subheader(
        "3️⃣ Predicted Infection Mask"
    )

    st.image(
        pred_mask * 255,
        caption="Attention U-Net Prediction",
        clamp=True,
        width=500
    )


    # =====================================================
    # COMPARISON
    # =====================================================

    if ground_truth_available:

        st.subheader(
            "4️⃣ Ground Truth vs Prediction"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                ground_truth_mask * 255,
                caption="Ground Truth",
                clamp=True,
                width=500
            )

        with col2:

            st.image(
                pred_mask * 255,
                caption="Predicted Mask",
                clamp=True,
                width=500
            )


    # =====================================================
    # EXPLAINABLE INFECTION LOCALIZATION
    # =====================================================

    st.subheader(
        "5️⃣ Explainable Infection Localization"
    )

    st.write(
        "The highlighted regions show where the "
        "Attention U-Net predicts possible infection."
    )


    # Resize mask back to original image size
    mask_original_size = cv2.resize(
        pred_mask,
        (
            original_image.shape[1],
            original_image.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )


    # Convert grayscale image to RGB
    original_rgb = cv2.cvtColor(
        original_image,
        cv2.COLOR_GRAY2RGB
    )


    # Create red overlay
    overlay = np.zeros_like(
        original_rgb
    )

    overlay[:, :, 0] = 255


    overlay_image = original_rgb.copy()

    infection_region = (
        mask_original_size == 1
    )


    # Highlight predicted infection regions
    overlay_image[infection_region] = cv2.addWeighted(
        original_rgb[infection_region],
        0.5,
        overlay[infection_region],
        0.5,
        0
    )


    st.image(
        overlay_image,
        caption="Possible infection regions highlighted",
        width=500
    )


    # =====================================================
    # INFECTION QUANTIFICATION
    # =====================================================

    infected_pixels = int(
        np.sum(pred_mask)
    )

    total_pixels = int(
        pred_mask.shape[0] *
        pred_mask.shape[1]
    )


    infection_percentage = (
        infected_pixels /
        total_pixels
    ) * 100


    # =====================================================
    # INFECTION EXTENT
    # =====================================================

    if infection_percentage == 0:

        infection_extent = (
            "No Detected Infection"
        )

    elif infection_percentage <= 1:

        infection_extent = "Low"

    elif infection_percentage <= 5:

        infection_extent = "Moderate"

    else:

        infection_extent = "High"


    # =====================================================
    # FINAL PREDICTION
    # =====================================================

    if infected_pixels > 50:

        result = "INFECTED"

    else:

        result = "NORMAL"


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.subheader(
        "6️⃣ Prediction"
    )

    if result == "INFECTED":

        st.error(
            "🔴 Possible Infection Detected"
        )

    else:

        st.success(
            "🟢 No Significant Infection Detected"
        )


    # =====================================================
    # INFECTION ANALYSIS
    # =====================================================

    st.subheader(
        "7️⃣ Infection Analysis"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Infected Pixels",
            infected_pixels
        )


    with col2:

        st.metric(
            "Infection Percentage",
            f"{infection_percentage:.2f}%"
        )


    with col3:

        st.metric(
            "Infection Extent",
            infection_extent
        )


    # =====================================================
    # GROUND TRUTH METRICS
    # =====================================================

    if ground_truth_available:

        st.subheader(
            "8️⃣ Segmentation Comparison"
        )

        gt_pixels = int(
            np.sum(ground_truth_mask)
        )

        predicted_pixels = int(
            np.sum(pred_mask)
        )


        # Intersection
        intersection = np.logical_and(
            ground_truth_mask,
            pred_mask
        ).sum()


        # Union
        union = np.logical_or(
            ground_truth_mask,
            pred_mask
        ).sum()


        # Dice Score
        dice = (
            (2 * intersection) /
            (gt_pixels + predicted_pixels + 1e-7)
        )


        # IoU
        iou = (
            intersection /
            (union + 1e-7)
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "GT Pixels",
                gt_pixels
            )


        with col2:

            st.metric(
                "Predicted Pixels",
                predicted_pixels
            )


        with col3:

            st.metric(
                "Dice Score",
                f"{dice:.4f}"
            )


        with col4:

            st.metric(
                "IoU",
                f"{iou:.4f}"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Attention U-Net based CT image segmentation "
    "for research and educational purposes. "
    "The infection extent categories are project-defined "
    "and are not clinically validated."
)
