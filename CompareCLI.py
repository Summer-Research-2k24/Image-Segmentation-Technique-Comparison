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


def load_image(path):
    """Load and preprocess image."""
    img = plt.imread(path)
    if img.shape[-1] == 4:  # Handle RGBA
        img = rgba2rgb(img)
    gray = color.rgb2gray(img) if len(img.shape) == 3 else img
    return img, gray


# Energy-based Methods
def active_contour_segmentation(img, img_gray):
    """Active Contour segmentation."""
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
    mask[rr, cc] = True
    return mask


def level_set_segmentation(img_gray):
    """Level Set segmentation."""
    init_ls = checkerboard_level_set(img_gray.shape, 10)
    return morphological_geodesic_active_contour(
        gaussian(img_gray, 3),
        num_iter=100,
        init_level_set=init_ls)


def graph_cut_segmentation(img_gray):
    """Graph Cut segmentation using Otsu's method."""
    return img_gray > filters.threshold_otsu(img_gray)


def random_walker_segmentation(img_gray):
    """Random Walker segmentation."""
    smooth_img = filters.gaussian(img_gray, 2)
    markers = np.zeros(smooth_img.shape, dtype=np.uint)
    markers[smooth_img < 0.6] = 1
    markers[smooth_img > 0.9] = 2
    return random_walker(smooth_img, markers, beta=10, mode='bf')


# Region-based Methods
def seeded_region_growing(img_gray, seed_point=(100, 100), threshold=15):
    """Seeded Region Growing segmentation."""
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    mask[seed_point] = 255
    seed_queue = [seed_point]
    while seed_queue:
        x, y = seed_queue.pop(0)
        for i, j in [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]:
            nx, ny = x + i, y + j
            if (0 <= nx < img_gray.shape[0] and 0 <= ny < img_gray.shape[1] and
                    mask[nx, ny] == 0 and
                    abs(int(img_gray[nx, ny] * 255) - int(img_gray[x, y] * 255)) < threshold):
                mask[nx, ny] = 255
                seed_queue.append((nx, ny))
    return mask


def region_splitting_merging(img_gray):
    """Region Splitting and Merging segmentation."""

    def split_region(image, x, y, size):
        if size < 1:
            return
        region = image[y:y + size, x:x + size]
        if np.std(region) < 0.02:  # Threshold for homogeneity
            segmented[y:y + size, x:x + size] = np.mean(region)
        else:
            half_size = size // 2
            for new_x, new_y in [(x, y), (x + half_size, y),
                                 (x, y + half_size), (x + half_size, y + half_size)]:
                split_region(image, new_x, new_y, half_size)

    segmented = np.zeros_like(img_gray)
    split_region(img_gray, 0, 0, img_gray.shape[0])
    return segmented


def clustering_segmentation(img, k=3):
    """K-means clustering segmentation."""
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img.shape)
    return color.rgb2gray(segmented)


def mean_shift_segmentation(img):
    """Mean Shift segmentation."""
    shifted = cv2.pyrMeanShiftFiltering(img.astype(np.uint8), sp=21, sr=51)
    return color.rgb2gray(shifted)


def watershed_segmentation(img_gray):
    """Watershed segmentation."""
    denoised = gaussian(img_gray, 2)
    markers = np.zeros_like(img_gray, dtype=int)
    markers[denoised < 0.4] = 1
    markers[denoised > 0.7] = 2
    return segmentation.watershed(denoised, markers)


def evaluate_segmentation(segmented, ground_truth):
    """Calculate evaluation metrics."""
    if segmented.shape != ground_truth.shape:
        segmented = resize(segmented, ground_truth.shape, order=0,
                           preserve_range=True, anti_aliasing=False)

    # Ensure binary masks
    if not np.array_equal(segmented, segmented.astype(bool)):
        segmented = segmented > segmented.mean()

    y_true = ground_truth.flatten().astype(bool)
    y_pred = segmented.flatten().astype(bool)

    return {
        'F1-score': f1_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred)
    }


def main():
    # Load image
    image_path = 'img.png'
    img, img_gray = load_image(image_path)

    # Create mock ground truth for evaluation
    ground_truth = np.zeros_like(img_gray, dtype=bool)
    center = (img_gray.shape[0] // 2, img_gray.shape[1] // 2)
    rr, cc = draw.disk(center, min(img_gray.shape) // 4)
    ground_truth[rr, cc] = True

    # Define all segmentation methods
    methods = {
        # Energy-based methods
        'Active Contour': lambda: active_contour_segmentation(img, img_gray),
        'Level Set': lambda: level_set_segmentation(img_gray),
        'Graph Cut': lambda: graph_cut_segmentation(img_gray),
        'Random Walker': lambda: random_walker_segmentation(img_gray),
        # Region-based methods
        'Seeded Region Growing': lambda: seeded_region_growing(img_gray),
        'Region Splitting-Merging': lambda: region_splitting_merging(img_gray),
        'Clustering': lambda: clustering_segmentation(img),
        'Mean Shift': lambda: mean_shift_segmentation(img),
        'Watershed': lambda: watershed_segmentation(img_gray)
    }

    results = []

    # Run all methods and collect results
    for name, method in methods.items():
        print(f"Running {name}...")
        start_time = time.time()
        try:
            segmentation = method()
            end_time = time.time()
            metrics = evaluate_segmentation(segmentation, ground_truth)
            results.append({
                'Method': name,
                'F1-score': metrics['F1-score'],
                'Recall': metrics['Recall'],
                'Runtime': end_time - start_time
            })

            # Visualize results
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(img)
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(segmentation, cmap='gray')
            plt.title(f'{name} Segmentation')
            plt.axis('off')

            plt.subplot(133)
            plt.imshow(ground_truth, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in {name}: {str(e)}")

    # Print performance matrix
    print("\nPerformance Matrix:")
    print("{:<25} {:<10} {:<10} {:<10}".format(
        "Method", "F1-score", "Recall", "Runtime"))
    print("-" * 55)
    for result in results:
        print("{:<25} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            result["Method"],
            result["F1-score"],
            result["Recall"],
            result["Runtime"]))

    # Find best method based on F1-score
    best_method = max(results, key=lambda x: x['F1-score'])
    print(f"\nBest performing method: {best_method['Method']}")
    print(f"F1-score: {best_method['F1-score']:.4f}")
    print(f"Recall: {best_method['Recall']:.4f}")
    print(f"Runtime: {best_method['Runtime']:.4f} seconds")

    # Plot comparison
    methods = [r['Method'] for r in results]
    f1_scores = [r['F1-score'] for r in results]
    recalls = [r['Recall'] for r in results]
    runtimes = [r['Runtime'] for r in results]

    plt.figure(figsize=(15, 6))
    x = np.arange(len(methods))
    width = 0.25

    plt.bar(x - width, f1_scores, width, label='F1-score')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, runtimes, width, label='Runtime (s)')

    plt.xlabel('Methods')
    plt.ylabel('Scores / Time')
    plt.title('Performance Comparison of Segmentation Methods')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
