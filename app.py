import streamlit as st
from ultralytics import SAM
import cv2
import numpy as np
from PIL import Image
import torch
import os

# Page config
st.set_page_config(
    page_title="MobileSAM Segmentation Tool",
    page_icon="🎯",
    layout="wide"
)

# Initialize model with caching
@st.cache_resource
def load_model():
    """Load MobileSAM model once and cache it"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SAM('mobile_sam.pt')
    return model, device

model, device = load_model()

# Title and description
st.title("🎯 MobileSAM Image Segmentation Tool")
st.markdown("Upload an image or enter a file path to segment objects")

# Device info
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
st.sidebar.info(f"🖥️ Running on: **{device_name}**")

# Input method selection
input_method = st.sidebar.radio(
    "Choose Input Method:",
    ["File Uploader", "File Path"]
)

image = None
image_path = None

# Input handling
if input_method == "File Uploader":
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload an image to segment"
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_path = temp_path
        image = Image.open(uploaded_file)

elif input_method == "File Path":
    path_input = st.text_input(
        "Enter image file path:",
        placeholder="e.g., ./images/sample.jpg or C:/Users/Documents/image.png",
        help="Enter the full or relative path to your image"
    )

    if path_input:
        if os.path.exists(path_input):
            image_path = path_input
            image = Image.open(path_input)
            st.success(f"✅ Image loaded from: {path_input}")
        else:
            st.error(f"❌ File not found: {path_input}")

# Display original image
if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        # Get image dimensions
        img_width, img_height = image.size
        st.caption(f"Image size: {img_width} x {img_height} pixels")

    # Segmentation settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Segmentation Settings")

    segmentation_mode = st.sidebar.selectbox(
        "Mode:",
        ["Automatic (Segment Everything)", "Box Prompt", "Point Prompt"]
    )

    # Mode-specific inputs BEFORE segment button
    points_input = None
    box_input = None

    if segmentation_mode == "Point Prompt":
        st.sidebar.markdown("#### Point Coordinates")
        st.sidebar.info("Enter points as [[x1,y1], [x2,y2]]")
        points_input = st.sidebar.text_area(
            "Points (JSON format):",
            value="[[300, 300]]",
            height=100,
            help="Example: [[100,150], [200,250]]"
        )

        # Option to mark foreground or background
        point_type = st.sidebar.radio(
            "Point Type:",
            ["Foreground (object)", "Background (exclude)"]
        )

    elif segmentation_mode == "Box Prompt":
        st.sidebar.markdown("#### Bounding Box Coordinates")
        st.sidebar.info("Enter box as: x1, y1, x2, y2")
        st.sidebar.markdown("Where (x1,y1) is top-left and (x2,y2) is bottom-right")

        # Individual input fields for better UX
        box_col1, box_col2 = st.sidebar.columns(2)
        with box_col1:
            x1 = st.number_input("x1 (left):", min_value=0, max_value=img_width, value=100, key="x1")
            y1 = st.number_input("y1 (top):", min_value=0, max_value=img_height, value=100, key="y1")
        with box_col2:
            x2 = st.number_input("x2 (right):", min_value=0, max_value=img_width, value=min(500, img_width), key="x2")
            y2 = st.number_input("y2 (bottom):", min_value=0, max_value=img_height, value=min(500, img_height), key="y2")

        box_input = f"{x1},{y1},{x2},{y2}"

        # Show box coordinates summary
        st.sidebar.success(f"📦 Box: [{x1}, {y1}, {x2}, {y2}]")

        # Visualize bbox on original image
        if st.sidebar.checkbox("Preview Bounding Box", value=True):
            img_with_box = np.array(image.copy())
            cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (255, 0, 0), 3)
            with col1:
                st.image(img_with_box, caption="Image with Bounding Box Preview", use_container_width=True)

    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        use_retina = st.checkbox("High Resolution (Retina Masks)", value=True, 
                                 help="Better for complex edges")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.9, 0.05)

    # Segment button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Segment Image", type="primary", use_container_width=True):
        with st.spinner("Segmenting image..."):
            try:
                results = None

                if segmentation_mode == "Automatic (Segment Everything)":
                    results = model(
                        image_path,
                        device=device,
                        retina_masks=use_retina,
                        conf=conf_threshold,
                        iou=iou_threshold
                    )

                elif segmentation_mode == "Point Prompt":
                    try:
                        import json
                        points = json.loads(points_input)

                        # Set labels based on user selection
                        if point_type == "Foreground (object)":
                            labels = [1] * len(points)
                        else:
                            labels = [0] * len(points)

                        st.info(f"Using {len(points)} point(s) as {point_type.lower()}")

                        results = model.predict(
                            image_path,
                            points=points,
                            labels=labels,
                            device=device,
                            retina_masks=use_retina
                        )
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON format for points")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Error with point prompt: {str(e)}")
                        st.stop()

                elif segmentation_mode == "Box Prompt":
                    try:
                        # Parse the bounding box coordinates
                        bbox = [int(x.strip()) for x in box_input.split(',')]

                        if len(bbox) != 4:
                            st.error("❌ Bounding box must have exactly 4 values: x1, y1, x2, y2")
                            st.stop()

                        # Validate bbox coordinates
                        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                            st.error("❌ Invalid box: x2 must be > x1 and y2 must be > y1")
                            st.stop()

                        st.info(f"Using bounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")

                        # Use the bounding box for segmentation
                        results = model.predict(
                            image_path,
                            bboxes=[bbox],  # Pass as list of bboxes
                            device=device,
                            retina_masks=use_retina
                        )
                    except ValueError:
                        st.error("❌ Invalid box format. Use numbers only: x1,y1,x2,y2")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Error with box prompt: {str(e)}")
                        st.stop()

                # Process results
                if results and results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()

                    # Create visualization
                    img_array = np.array(image)
                    output = img_array.copy()

                    # Overlay masks with different colors
                    for i, mask in enumerate(masks):
                        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                        mask_resized = cv2.resize(
                            mask.astype(np.uint8),
                            (img_array.shape[1], img_array.shape[0])
                        )
                        colored_mask = np.zeros_like(img_array)
                        colored_mask[mask_resized > 0] = color
                        output = cv2.addWeighted(output, 1, colored_mask, 0.5, 0)

                    # Draw bounding box on output if box mode
                    if segmentation_mode == "Box Prompt":
                        bbox_coords = [int(x.strip()) for x in box_input.split(',')]
                        cv2.rectangle(output, 
                                    (bbox_coords[0], bbox_coords[1]), 
                                    (bbox_coords[2], bbox_coords[3]), 
                                    (0, 255, 0), 2)

                    # Display result
                    with col2:
                        st.subheader("Segmented Output")
                        st.image(output, use_container_width=True)

                    # Statistics
                    st.success(f"✅ Segmented **{len(masks)}** object(s)")

                    # Download button
                    output_pil = Image.fromarray(output)
                    output_pil.save("segmented_output.png")

                    with open("segmented_output.png", "rb") as file:
                        st.sidebar.download_button(
                            label="📥 Download Result",
                            data=file,
                            file_name="segmented_output.png",
                            mime="image/png"
                        )

                    # Show individual masks
                    with st.expander("🔍 View Individual Masks"):
                        num_masks_to_show = min(len(masks), 8)
                        cols = st.columns(4)
                        for idx in range(num_masks_to_show):
                            with cols[idx % 4]:
                                st.image(masks[idx], caption=f"Mask {idx+1}", 
                                       use_container_width=True, clamp=True)

                    # Save individual masks option
                    if st.sidebar.checkbox("💾 Save Individual Masks"):
                        for idx, mask in enumerate(masks):
                            mask_img = (mask * 255).astype(np.uint8)
                            cv2.imwrite(f"mask_{idx+1}.png", mask_img)
                        st.sidebar.success(f"Saved {len(masks)} individual masks")

                else:
                    st.warning("⚠️ No objects detected. Try adjusting settings or using a different segmentation mode.")

            except Exception as e:
                st.error(f"❌ Error during segmentation: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

else:
    st.info("👈 Choose an input method and load an image to get started!")

    # Instructions
    st.markdown("""
    ### How to Use:

    1. **Load Image**: Choose file uploader or enter a file path
    2. **Select Mode**:
       - **Automatic**: Segments all objects automatically
       - **Box Prompt**: Draw a box around the object you want to segment
       - **Point Prompt**: Click points on the object
    3. **For Box Mode**: Enter coordinates (x1, y1, x2, y2) or use number inputs
    4. **Click Segment**: Process the image
    5. **Download**: Save your segmented result

    #### Box Coordinate Guide:
    - (x1, y1) = Top-left corner of the box
    - (x2, y2) = Bottom-right corner of the box
    - Coordinates start from (0, 0) at top-left of image
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This tool uses **MobileSAM** for efficient image segmentation.
- 🚀 Fast GPU-accelerated inference
- 🎯 High-resolution mask output
- 🎨 Multiple segmentation modes
- 📦 Bounding box support
- 📍 Point prompt support
""")
