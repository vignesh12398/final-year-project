import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lesion Analysis",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "attention_unet_model.keras",
        compile=False
    )

    return model


try:

    model = load_model()

    st.sidebar.success("Model loaded successfully")

except Exception as e:

    st.error("Could not load the model.")

    st.exception(e)

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🔬 Infection Region Analysis")

st.write(
    "Analyze individual predicted infection regions "
    "identified by the Attention U-Net segmentation model."
)


# ============================================================
# UPLOAD CT IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload CT Image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("L")

    original_image = np.array(image)


    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    resized_image = cv2.resize(
        original_image,
        (128, 128)
    )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    resized_image = (
        resized_image.astype(
            np.float32
        ) / 255.0
    )


    # --------------------------------------------------------
    # PREPARE MODEL INPUT
    # --------------------------------------------------------

    input_image = np.expand_dims(
        resized_image,
        axis=0
    )

    input_image = np.expand_dims(
        input_image,
        axis=-1
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    with st.spinner(
        "Detecting infection regions..."
    ):

        prediction = model.predict(
            input_image,
            verbose=0
        )


    prediction = prediction.squeeze()


    # ========================================================
    # CREATE BINARY MASK
    # ========================================================

    threshold = 0.05

    pred_mask = (
        prediction > threshold
    ).astype(
        np.uint8
    )


    # ========================================================
    # CONNECTED COMPONENT ANALYSIS
    # ========================================================

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            pred_mask,
            connectivity=8
        )
    )


    # ========================================================
    # MINIMUM REGION SIZE
    # ========================================================

    min_region_size = 10


    regions = []


    # ========================================================
    # EXTRACT REGIONS
    # ========================================================

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]


        # Ignore very small regions

        if area < min_region_size:
            continue


        x = stats[
            label,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            label,
            cv2.CC_STAT_TOP
        ]

        width = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        height = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ]


        center_x = centroids[
            label
        ][0]

        center_y = centroids[
            label
        ][1]


        regions.append({

            "area": int(area),

            "x": int(x),

            "y": int(y),

            "width": int(width),

            "height": int(height),

            "center_x": center_x,

            "center_y": center_y

        })


    # ========================================================
    # SUMMARY
    # ========================================================

    number_of_regions = len(
        regions
    )


    infected_pixels = int(
        np.sum(pred_mask)
    )


    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    st.subheader(
        "Lesion Analysis Results"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Detected Regions",
            number_of_regions
        )


    with col2:

        st.metric(
            "Total Infected Pixels",
            infected_pixels
        )


    with col3:

        if number_of_regions > 0:

            largest_region = max(
                regions,
                key=lambda r: r["area"]
            )

            st.metric(
                "Largest Region",
                f"{largest_region['area']} pixels"
            )

        else:

            st.metric(
                "Largest Region",
                "0 pixels"
            )


    # ========================================================
    # REGION VISUALIZATION
    # ========================================================

    if number_of_regions > 0:

        st.subheader(
            "Detected Infection Regions"
        )


        # Resize image to model dimensions

        region_image = cv2.resize(
            original_image,
            (128, 128)
        )


        # Convert grayscale → RGB

        region_image = cv2.cvtColor(
            region_image,
            cv2.COLOR_GRAY2RGB
        )


        # ----------------------------------------------------
        # DRAW REGIONS
        # ----------------------------------------------------

        for index, region in enumerate(
            regions
        ):

            x = region["x"]

            y = region["y"]

            width = region["width"]

            height = region["height"]


            # Draw bounding box

            cv2.rectangle(
                region_image,
                (x, y),
                (
                    x + width,
                    y + height
                ),
                (255, 0, 0),
                1
            )


            # Region label

            cv2.putText(
                region_image,
                f"R{index + 1}",
                (
                    x,
                    max(y - 4, 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1
            )


        st.image(
            region_image,
            caption="Individual predicted infection regions",
            width=600
        )


        # ====================================================
        # INDIVIDUAL REGION DETAILS
        # ====================================================

        st.subheader(
            "Region Details"
        )


        for index, region in enumerate(
            regions
        ):

            region_percentage = (

                region["area"] /
                infected_pixels

            ) * 100 if infected_pixels > 0 else 0


            st.markdown(
                f"### Region {index + 1}"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Area",
                    f"{region['area']} pixels"
                )


            with col2:

                st.metric(
                    "Area Contribution",
                    f"{region_percentage:.2f}%"
                )


            with col3:

                st.metric(
                    "Region Size",
                    f"{region['width']} × "
                    f"{region['height']}"
                )


        # ====================================================
        # LARGEST REGION
        # ====================================================

        largest_region = max(
            regions,
            key=lambda r: r["area"]
        )


        largest_percentage = (

            largest_region["area"] /
            infected_pixels

        ) * 100 if infected_pixels > 0 else 0


        st.subheader(
            "Largest Predicted Infection Region"
        )


        st.info(
            f"""
            **Area:** {largest_region['area']} pixels

            **Contribution to total predicted infection:**
            {largest_percentage:.2f}%

            **Center location:**
            ({largest_region['center_x']:.1f},
            {largest_region['center_y']:.1f})

            **Bounding box:**
            {largest_region['width']} ×
            {largest_region['height']} pixels
            """
        )


    else:

        st.info(
            "No sufficiently large predicted "
            "infection regions were detected."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Region analysis is based on connected components "
    "of the predicted segmentation mask and is intended "
    "for research and educational purposes."
)