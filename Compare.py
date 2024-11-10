import streamlit as st
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters, draw, segmentation, graph
from skimage.color import rgba2rgb
from skimage.filters import gaussian
from skimage.segmentation import active_contour, checkerboard_level_set, morphological_geodesic_active_contour, \
    random_walker
from sklearn.metrics import f1_score, recall_score
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Image Segmentation Comparison")


def load_image(uploaded_file):
    """Load and preprocess uploaded image."""
    img = Image.open(uploaded_file)
    img = np.array(img)
    if img.shape[-1] == 4:  # Handle RGBA
        img = rgba2rgb(img)
    gray = color.rgb2gray(img) if len(img.shape) == 3 else img
    return img, gray


def create_circular_mask(h, w, center=None, radius=None):
    """Create a circular mask."""
    if center is None:  # use the middle of the image
        center = (int(h / 2), int(w / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h - center[0], w - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

    mask = dist_from_center <= radius
    return mask


def active_contour_segmentation(img, img_gray):
    rows, cols = img.shape[:2]
    center_x, center_y = cols // 2, rows // 2
    radius = min(cols, rows) // 3
    s = np.linspace(0, 2 * np.pi, 400)
    r = center_y + radius * np.sin(s)
    c = center_x + radius * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img_gray, 3),
                           init, alpha=0.015, beta=10, gamma=0.001)
    mask = np.zeros_like(img_gray, dtype=bool)
    rr, cc = draw.polygon(snake[:, 0], snake[:, 1])
    # Clip coordinates to image boundaries
    rr = np.clip(rr, 0, mask.shape[0] - 1)
    cc = np.clip(cc, 0, mask.shape[1] - 1)
    mask[rr, cc] = True
    return np.uint8(mask * 255)  # Convert to uint8 for visualization


def level_set_segmentation(img_gray):
    init_ls = checkerboard_level_set(img_gray.shape, 10)
    mask = morphological_geodesic_active_contour(
        gaussian(img_gray, 3),
        num_iter=100,
        init_level_set=init_ls)
    return np.uint8(mask * 255)  # Convert to uint8 for visualization


def graph_cut_segmentation(img_gray):
    mask = img_gray > filters.threshold_otsu(img_gray)
    return np.uint8(mask * 255)  # Convert to uint8 for visualization


def random_walker_segmentation(img_gray):
    smooth_img = filters.gaussian(img_gray, 2)
    markers = np.zeros(smooth_img.shape, dtype=np.uint)
    markers[smooth_img < 0.6] = 1
    markers[smooth_img > 0.9] = 2
    mask = random_walker(smooth_img, markers, beta=10, mode='bf') == 2
    return np.uint8(mask * 255)  # Convert to uint8 for visualization


def seeded_region_growing(img_gray, seed_point=None):
    if seed_point is None:
        seed_point = (img_gray.shape[0] // 2, img_gray.shape[1] // 2)
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    mask[seed_point] = 255
    seed_queue = [seed_point]
    visited = set([seed_point])

    while seed_queue:
        x, y = seed_queue.pop(0)
        for i, j in [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]:
            nx, ny = x + i, y + j
            if ((nx, ny) not in visited and
                    0 <= nx < img_gray.shape[0] and
                    0 <= ny < img_gray.shape[1] and
                    abs(int(img_gray[nx, ny] * 255) - int(img_gray[x, y] * 255)) < 15):
                mask[nx, ny] = 255
                seed_queue.append((nx, ny))
                visited.add((nx, ny))
    return mask


def clustering_segmentation(img, k=3):
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img.shape)
    # Convert to grayscale and normalize
    gray_segmented = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
    return gray_segmented


def mean_shift_segmentation(img):
    shifted = cv2.pyrMeanShiftFiltering(img.astype(np.uint8), sp=21, sr=51)
    # Convert to grayscale
    gray_shifted = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)
    return gray_shifted


def watershed_segmentation(img_gray):
    denoised = gaussian(img_gray, 2)
    markers = np.zeros_like(img_gray, dtype=int)
    markers[denoised < 0.4] = 1
    markers[denoised > 0.7] = 2
    mask = segmentation.watershed(denoised, markers) == 2
    return np.uint8(mask * 255)  # Convert to uint8 for visualization


def main():
    st.title("Image Segmentation Method Comparison")

    # Sidebar for uploading image and selecting methods
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display original image
        img, img_gray = load_image(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(img, use_container_width=True)

        # Create ground truth with adjustable parameters
        with col2:
            st.subheader("Ground Truth (Generated)")
            radius_factor = st.slider("Ground Truth Circle Radius", 0.1, 0.5, 0.25, 0.05)
            ground_truth = create_circular_mask(
                img_gray.shape[0],
                img_gray.shape[1],
                radius=min(img_gray.shape) * radius_factor
            )
            ground_truth_display = np.uint8(ground_truth * 255)  # Convert for display
            st.image(ground_truth_display, use_container_width=True)

        # Define available methods
        methods = {
            'Active Contour': lambda: active_contour_segmentation(img, img_gray),
            'Level Set': lambda: level_set_segmentation(img_gray),
            'Graph Cut': lambda: graph_cut_segmentation(img_gray),
            'Random Walker': lambda: random_walker_segmentation(img_gray),
            'Seeded Region Growing': lambda: seeded_region_growing(img_gray),
            'Clustering': lambda: clustering_segmentation(img),
            'Mean Shift': lambda: mean_shift_segmentation(img),
            'Watershed': lambda: watershed_segmentation(img_gray)
        }

        # Method selection
        st.sidebar.header("Select Methods")
        selected_methods = st.sidebar.multiselect(
            "Choose segmentation methods to compare",
            list(methods.keys()),
            default=['Graph Cut', 'Clustering']
        )

        if st.sidebar.button("Run Selected Methods"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, method_name in enumerate(selected_methods):
                status_text.text(f"Running {method_name}...")
                start_time = time.time()

                try:
                    segmentation_result = methods[method_name]()
                    end_time = time.time()

                    # Calculate metrics using binary masks
                    y_true = ground_truth.flatten()
                    y_pred = (segmentation_result > 127).flatten()  # Convert to binary

                    metrics = {
                        'Method': method_name,
                        'Runtime': end_time - start_time,
                        'F1-Score': f1_score(y_true, y_pred),
                        'Recall': recall_score(y_true, y_pred)
                    }

                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"{method_name} Result")
                        st.image(segmentation_result, use_container_width=True)
                    with col2:
                        st.subheader(f"{method_name} Metrics")
                        st.write(f"Runtime: {metrics['Runtime']:.4f} seconds")
                        st.write(f"F1-Score: {metrics['F1-Score']:.4f}")
                        st.write(f"Recall: {metrics['Recall']:.4f}")

                    results.append(metrics)

                except Exception as e:
                    st.error(f"Error in {method_name}: {str(e)}")

                progress_bar.progress((i + 1) / len(selected_methods))

            status_text.text("Completed!")

            # Display comparison charts if there are results
            if results:
                st.subheader("Performance Comparison")
                df_results = pd.DataFrame(results)

                # Create tabs for different metrics
                tab1, tab2, tab3 = st.tabs(["Runtime", "F1-Score", "Recall"])

                with tab1:
                    fig, ax = plt.subplots()
                    ax.bar(df_results['Method'], df_results['Runtime'], color='skyblue')
                    ax.set_title("Runtime Comparison")
                    ax.set_xlabel("Method")
                    ax.set_ylabel("Runtime (seconds)")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                with tab2:
                    fig, ax = plt.subplots()
                    ax.bar(df_results['Method'], df_results['F1-Score'], color='lightgreen')
                    ax.set_title("F1-Score Comparison")
                    ax.set_xlabel("Method")
                    ax.set_ylabel("F1-Score")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                with tab3:
                    fig, ax = plt.subplots()
                    ax.bar(df_results['Method'], df_results['Recall'], color='lightcoral')
                    ax.set_title("Recall Comparison")
                    ax.set_xlabel("Method")
                    ax.set_ylabel("Recall")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

if __name__ == '__main__':
    main()