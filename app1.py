import streamlit as st
from ultralytics import SAM
import cv2
import numpy as np
from PIL import Image
import torch
import random
import os

# Page config
st.set_page_config(
    page_title="MobileSAM Segmentation Tool",
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
st.title("MobileSAM Image Segmentation Tool")
st.markdown("Upload an image or enter a file path to segment objects")

# Device info
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
st.sidebar.info(f"Running on: **{device_name}**")

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
            st.success(f"Image loaded from: {path_input}")
        else:
            st.error(f" File not found: {path_input}")

# Display original image
if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")

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
    show_preview = False

    if segmentation_mode == "Point Prompt":
        st.sidebar.markdown("#### Point Coordinates")
        st.sidebar.info("Enter points as [[x1,y1], [x2,y2], ...]")
        points_input = st.sidebar.text_area(
            "Points (JSON format):",
            value="[[300, 300]]",
            height=100,
            help="Example: [[100,150], [200,250], [300,350]]"
        )

        # Option to mark foreground or background
        point_type = st.sidebar.radio(
            "Point Type:",
            ["Foreground (object)", "Background (exclude)"]
        )

        # Preview points checkbox
        show_point_preview = st.sidebar.checkbox("Preview Points", value=True)

        # Visualize points on original image
        if show_point_preview:
            try:
                import json
                points = json.loads(points_input)
                img_with_points = np.array(image.copy())

                # Determine color based on point type
                if point_type == "Foreground (object)":
                    point_color = (0, 0, 255)  # Green for foreground
                    label_text = "FG"
                else:
                    point_color = (255, 0, 0)  # Red for background
                    label_text = "BG"

                # Draw each point
                for idx, point in enumerate(points):
                    x, y = int(point[0]), int(point[1])
                    # Draw circle
                    cv2.circle(img_with_points, (x, y), 10, point_color, -1)
                    cv2.circle(img_with_points, (x, y), 14, (255, 255, 255), 4)
                    # Draw label
                    cv2.putText(img_with_points, f"{label_text}{idx+1}", 
                               (x + 15, y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 2)
                    cv2.putText(img_with_points, f"{label_text}{idx+1}", 
                               (x + 15, y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, point_color, 1)

                with col1:
                    st.image(img_with_points, caption=f"Image with {len(points)} Point(s)", 
                           use_container_width=True)

                st.sidebar.success(f"📍 {len(points)} point(s) marked as {point_type.lower()}")
                show_preview = True

            except json.JSONDecodeError:
                st.sidebar.error(" Invalid JSON format")
            except Exception as e:
                st.sidebar.warning(f"Cannot preview: {str(e)}")

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
        st.sidebar.success(f"Box: [{x1}, {y1}, {x2}, {y2}]")

        # Visualize bbox on original image
        show_box_preview = st.sidebar.checkbox("Preview Bounding Box", value=True)
        if show_box_preview:
            img_with_box = np.array(image.copy())
            cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Add corner markers
            corner_size = 20
            cv2.line(img_with_box, (x1, y1), (x1 + corner_size, y1), (255, 0, 0), 3)
            cv2.line(img_with_box, (x1, y1), (x1, y1 + corner_size), (255, 0, 0), 3)
            cv2.line(img_with_box, (x2, y2), (x2 - corner_size, y2), (255, 0, 0), 3)
            cv2.line(img_with_box, (x2, y2), (x2, y2 - corner_size), (255, 0, 0), 3)
            # Add text labels
            cv2.putText(img_with_box, "Top-Left", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img_with_box, "Bottom-Right", (x2 - 100, y2 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            with col1:
                st.image(img_with_box, caption="Image with Bounding Box Preview", 
                       use_container_width=True)
            show_preview = True

    # Show original without preview if no preview mode enabled
    if not show_preview:
        with col1:
            st.image(image, use_container_width=True)

    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        use_retina = st.checkbox("High Resolution (Retina Masks)", value=True, 
                                 help="Better for complex edges")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.9, 0.05)

    # Segment button
    st.sidebar.markdown("---")
    if st.sidebar.button("Segment Image", type="primary", use_container_width=True):
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
                        st.error(" Invalid JSON format for points")
                        st.stop()
                    except Exception as e:
                        st.error(f" Error with point prompt: {str(e)}")
                        st.stop()

                elif segmentation_mode == "Box Prompt":
                    try:
                        # Parse the bounding box coordinates
                        bbox = [int(x.strip()) for x in box_input.split(',')]

                        if len(bbox) != 4:
                            st.error(" Bounding box must have exactly 4 values: x1, y1, x2, y2")
                            st.stop()

                        # Validate bbox coordinates
                        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                            st.error(" Invalid box: x2 must be > x1 and y2 must be > y1")
                            st.stop()

                        st.info(f"Using bounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")

                        # Use the bounding box for segmentation
                        results = model.predict(
                            image_path,
                            bboxes=[bbox],
                            device=device,
                            retina_masks=use_retina
                        )
                    except ValueError:
                        st.error(" Invalid box format. Use numbers only: x1,y1,x2,y2")
                        st.stop()
                    except Exception as e:
                        st.error(f" Error with box prompt: {str(e)}")
                        st.stop()

                # Process results
                if results and results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()

                    # Create visualization
                    img_array = np.array(image)
                    output = img_array.copy()

                    # Overlay masks with different colors
                    for i, mask in enumerate(masks):
                        color = np.array([0,255,random.random() * 255])
                        mask_resized = cv2.resize(
                            mask.astype(np.uint8),
                            (img_array.shape[1], img_array.shape[0])
                        )
                        colored_mask = np.zeros_like(img_array)
                        colored_mask[mask_resized > 0] = color
                        output = cv2.addWeighted(output, 1, colored_mask, 0.7, 0)

                    # Draw prompts on output for reference
                    if segmentation_mode == "Box Prompt":
                        bbox_coords = [int(x.strip()) for x in box_input.split(',')]
                        cv2.rectangle(output, 
                                    (bbox_coords[0], bbox_coords[1]), 
                                    (bbox_coords[2], bbox_coords[3]), 
                                    (0, 255, 0), 2)
                    elif segmentation_mode == "Point Prompt":
                        import json
                        points = json.loads(points_input)
                        for idx, point in enumerate(points):
                            x, y = int(point[0]), int(point[1])
                            if point_type == "Foreground (object)":
                                cv2.circle(output, (x, y), 8, (0, 255, 0), -1)
                            else:
                                cv2.circle(output, (x, y), 8, (255, 0, 0), -1)
                            cv2.circle(output, (x, y), 10, (255, 255, 255), 2)

                    # Display result
                    with col2:
                        st.markdown('<div style="padding: 18px;">', unsafe_allow_html=True)
                        st.subheader("Segmented Output")
                        st.image(output, use_container_width=True)

                    # Statistics
                    st.success(f"Segmented **{len(masks)}** object(s)")

                    # Download button
                    output_pil = Image.fromarray(output)
                    output_pil.save("segmented_output.png")

                    with open("segmented_output.png", "rb") as file:
                        st.sidebar.download_button(
                            label="Download Result",
                            data=file,
                            file_name="segmented_output.png",
                            mime="image/png"
                        )


                else:
                    st.warning(" No objects detected. Try adjusting settings or using a different segmentation mode.")

            except Exception as e:
                st.error(f" Error during segmentation: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

else:
    pass