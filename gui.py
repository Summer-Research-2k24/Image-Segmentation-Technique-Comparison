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
from skimage.transform import resize
from sklearn.metrics import f1_score, recall_score
import io
from PIL import Image

# Set page config
st.set_page_config(page_title="Image Segmentation Explorer", layout="wide")


def normalize_image(img):
    """Normalize image to [0, 1] range."""
    if img.dtype == np.uint8:
        return img.astype(float) / 255.0
    elif img.max() > 1.0:
        return img.astype(float) / img.max()
    return img


# Utility Functions
def load_image(uploaded_file):
    """Load and preprocess uploaded image."""
    if uploaded_file is None:
        return None, None

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize RGB image
    img = normalize_image(img)

    # Convert to gray and ensure normalization
    gray = color.rgb2gray(img)
    gray = normalize_image(gray)

    return img, gray


def create_ground_truth(img_gray, center_ratio=0.5, radius_ratio=0.25):
    """Create circular ground truth mask."""
    ground_truth = np.zeros_like(img_gray, dtype=bool)
    center = (int(img_gray.shape[0] * center_ratio),
              int(img_gray.shape[1] * center_ratio))
    radius = int(min(img_gray.shape) * radius_ratio)
    rr, cc = draw.disk(center, radius)
    mask = (rr >= 0) & (rr < img_gray.shape[0]) & (cc >= 0) & (cc < img_gray.shape[1])
    rr, cc = rr[mask], cc[mask]
    ground_truth[rr, cc] = True
    return ground_truth


# Segmentation Methods (Energy-based)
def active_contour_segmentation(img, img_gray):
    rows, cols = img_gray.shape
    center_x, center_y = cols // 2, rows // 2
    radius = min(cols, rows) // 3
    s = np.linspace(0, 2 * np.pi, 400)
    r = center_y + radius * np.sin(s)
    c = center_x + radius * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img_gray, 3, preserve_range=True),
                           init, alpha=0.015, beta=10, gamma=0.001)
    mask = np.zeros_like(img_gray, dtype=bool)
    rr, cc = draw.polygon(snake[:, 0], snake[:, 1])
    mask[rr, cc] = True
    return mask


def level_set_segmentation(img_gray):
    init_ls = checkerboard_level_set(img_gray.shape, 10)
    return morphological_geodesic_active_contour(
        gaussian(img_gray, 3, preserve_range=True),
        num_iter=100,
        init_level_set=init_ls)


def graph_cut_segmentation(img_gray):
    return img_gray > filters.threshold_otsu(img_gray)


def random_walker_segmentation(img_gray):
    markers = np.zeros(img_gray.shape, dtype=np.uint)
    markers[img_gray < 0.6] = 1
    markers[img_gray > 0.9] = 2
    return random_walker(img_gray, markers, beta=10, mode='bf') - 1


# Segmentation Methods (Region-based)
def seeded_region_growing(img_gray):
    seed_point = (img_gray.shape[0] // 2, img_gray.shape[1] // 2)
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    mask[seed_point] = 1
    seed_queue = [seed_point]
    threshold = 0.15  # Normalized threshold

    while seed_queue:
        x, y = seed_queue.pop(0)
        for i, j in [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]:
            nx, ny = x + i, y + j
            if (0 <= nx < img_gray.shape[0] and 0 <= ny < img_gray.shape[1] and
                    mask[nx, ny] == 0 and
                    abs(img_gray[nx, ny] - img_gray[x, y]) < threshold):
                mask[nx, ny] = 1
                seed_queue.append((nx, ny))
    return mask.astype(bool)


def region_splitting_merging(img_gray):
    def split_region(image, x, y, size):
        if size < 1:
            return
        region = image[y:y + size, x:x + size]
        if np.std(region) < 0.02:
            segmented[y:y + size, x:x + size] = np.mean(region)
        else:
            half_size = size // 2
            for new_x, new_y in [(x, y), (x + half_size, y),
                                 (x, y + half_size), (x + half_size, y + half_size)]:
                split_region(image, new_x, new_y, half_size)

    segmented = np.zeros_like(img_gray)
    split_region(img_gray, 0, 0, img_gray.shape[0])
    return segmented > segmented.mean()


def clustering_segmentation(img, k=3):
    """
    Perform k-means clustering segmentation on the image.
    Works with both RGB and grayscale images.
    """
    # Check if the image is grayscale
    is_grayscale = len(img.shape) == 2

    if is_grayscale:
        # Reshape grayscale image to 2D array of pixels
        pixels = img.reshape((-1, 1))
    else:
        # Reshape RGB image to 2D array of pixels
        pixels = img.reshape((-1, 3))

    # Scale to [0, 255] for k-means
    pixels = (pixels * 255).astype(np.float32)

    # Define k-means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Apply k-means clustering
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8
    centers = np.uint8(centers)

    # Reconstruct segmented image
    if is_grayscale:
        # Reshape back to original grayscale image shape
        segmented = centers[labels.flatten()].reshape(img.shape)
        result = segmented.astype(float) / 255.0
    else:
        # Reshape back to original RGB image shape
        segmented = centers[labels.flatten()].reshape(img.shape)
        # Convert to grayscale and normalize
        result = color.rgb2gray(segmented / 255.0)

    # Ensure result is binary
    result = result > result.mean()

    return result


def mean_shift_segmentation(img):
    """
    Perform mean shift segmentation on the image.
    Ensures input is in the correct format for OpenCV (8-bit, 3-channel)
    """
    # Check if image is single channel (grayscale)
    if len(img.shape) == 2:
        # Convert grayscale to RGB
        img_rgb = np.stack((img,) * 3, axis=-1)
    else:
        img_rgb = img

    # Ensure image is in uint8 format and in range [0, 255]
    img_uint8 = (img_rgb * 255).clip(0, 255).astype(np.uint8)

    # Apply mean shift filtering
    shifted = cv2.pyrMeanShiftFiltering(img_uint8, sp=21, sr=51)

    # Convert result back to grayscale and normalize to [0, 1]
    if len(shifted.shape) == 3:
        result = color.rgb2gray(shifted)
    else:
        result = shifted.astype(float)

    return result / 255.0


def watershed_segmentation(img_gray):
    """
    Perform watershed segmentation on grayscale image.
    Returns binary mask indicating segmented regions.
    """
    # Apply gradient magnitude filter for edge detection
    gradient = filters.sobel(img_gray)

    # Normalize gradient to [0, 1] range
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())

    # Denoise the gradient image
    denoised = filters.gaussian(gradient, sigma=2, preserve_range=True)

    # Ensure denoised image is in [0, 1] range
    denoised = np.clip(denoised, 0, 1)

    # Create markers - identify likely foreground and background regions
    markers = np.zeros_like(img_gray, dtype=int)
    mean_val = denoised.mean()
    std_val = denoised.std()

    # Ensure thresholds are within [0, 1]
    low_threshold = np.clip(mean_val - std_val, 0, 1)
    high_threshold = np.clip(mean_val + std_val, 0, 1)

    markers[denoised < low_threshold] = 1  # Background
    markers[denoised > high_threshold] = 2  # Foreground

    # Apply watershed segmentation
    labels = segmentation.watershed(denoised, markers, watershed_line=True)

    # Convert to binary mask (foreground = True, background = False)
    binary_mask = (labels == 2)

    return binary_mask


# Evaluation Function
def evaluate_segmentation(segmented, ground_truth):
    if segmented.shape != ground_truth.shape:
        segmented = resize(segmented, ground_truth.shape, order=0,
                           preserve_range=True, anti_aliasing=False)

    if not np.array_equal(segmented, segmented.astype(bool)):
        segmented = segmented > segmented.mean()

    y_true = ground_truth.flatten().astype(bool)
    y_pred = segmented.flatten().astype(bool)

    return {
        'F1-score': f1_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred)
    }


# Streamlit App
def main():
    st.title("Image Segmentation Explorer")

    # Sidebar
    st.sidebar.header("Settings")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    # Method selection
    method_category = st.sidebar.radio(
        "Select Method Category",
        ["Energy-based Methods", "Region-based Methods"]
    )

    energy_methods = {
        "Active Contour": active_contour_segmentation,
        "Level Set": level_set_segmentation,
        "Graph Cut": graph_cut_segmentation,
        "Random Walker": random_walker_segmentation
    }

    region_methods = {
        "Seeded Region Growing": seeded_region_growing,
        "Region Splitting-Merging": region_splitting_merging,
        "Clustering": clustering_segmentation,
        "Mean Shift": mean_shift_segmentation,
        "Watershed": watershed_segmentation
    }

    methods = energy_methods if method_category == "Energy-based Methods" else region_methods
    selected_method = st.sidebar.selectbox("Select Method", list(methods.keys()))

    # Main content
    if uploaded_file is not None:
        img, img_gray = load_image(uploaded_file)

        if img is not None:
            # Create columns for display
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Original Image")
                st.image(img, use_container_width=True)

            # Generate ground truth
            ground_truth = create_ground_truth(img_gray)

            # Run segmentation
            with st.spinner(f'Running {selected_method} segmentation...'):
                start_time = time.time()
                try:
                    if selected_method == "Active Contour":
                        segmentation = methods[selected_method](img, img_gray)
                    else:
                        segmentation = methods[selected_method](img_gray)

                    end_time = time.time()
                    runtime = end_time - start_time

                    # Evaluate results
                    metrics = evaluate_segmentation(segmentation, ground_truth)

                    with col2:
                        st.subheader("Segmentation Result")
                        st.image(segmentation.astype(float), use_container_width=True)

                    with col3:
                        st.subheader("Ground Truth")
                        st.image(ground_truth.astype(float), use_container_width=True)

                    # Display metrics
                    st.subheader("Performance Metrics")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                    with metrics_col1:
                        st.metric("F1-score", f"{metrics['F1-score']:.4f}")

                    with metrics_col2:
                        st.metric("Recall", f"{metrics['Recall']:.4f}")

                    with metrics_col3:
                        st.metric("Runtime", f"{runtime:.4f} s")

                except Exception as e:
                    st.error(f"Error running segmentation: {str(e)}")
        else:
            st.error("Error loading image. Please try another file.")
    else:
        st.info("Please upload an image to begin.")

    # Add information about methods
    with st.expander("About the Methods"):
        st.markdown("""
        ### Energy-based Methods
        - **Active Contour**: Uses energy minimization to evolve a curve towards object boundaries
        - **Level Set**: Represents curves implicitly as the zero level set of a higher-dimensional function
        - **Graph Cut**: Treats image as a graph and finds optimal cuts to segment regions
        - **Random Walker**: Uses random walks to determine pixel labels based on seed points

        ### Region-based Methods
        - **Seeded Region Growing**: Grows regions from seed points based on pixel similarity
        - **Region Splitting-Merging**: Recursively splits and merges regions based on homogeneity
        - **Clustering**: Groups pixels into clusters based on color/intensity similarity
        - **Mean Shift**: Non-parametric clustering technique that finds modes in the feature space
        - **Watershed**: Treats image as a topographic surface and finds catchment basins
        """)


if __name__ == "__main__":
    main()
