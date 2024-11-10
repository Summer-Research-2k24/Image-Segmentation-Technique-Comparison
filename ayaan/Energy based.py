import time

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, filters
from skimage import draw
from skimage.color import rgba2rgb
from skimage.filters import gaussian
from skimage.segmentation import active_contour, checkerboard_level_set, morphological_geodesic_active_contour, \
    random_walker
from skimage.transform import resize
from sklearn.metrics import f1_score, recall_score


def load(path):
    img = plt.imread(path)
    return img


def preprocess_image(img):
    if img.shape[-1] == 4:  # Check if the image has 4 channels (RGBA)
        img = rgba2rgb(img)
    if len(img.shape) == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img
    return img, img_gray


def active_contour_segmentation(img, img_gray):
    rows, cols = img.shape[:2]
    center_x, center_y = cols // 2, rows // 2
    radius = min(cols, rows) // 3
    s = np.linspace(0, 2 * np.pi, 400)
    r = center_y + radius * np.sin(s)
    c = center_x + radius * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img_gray, sigma=3, preserve_range=False),
                           init, alpha=0.015, beta=10, gamma=0.001)

    # Create a binary mask from the snake
    mask = np.zeros_like(img_gray, dtype=bool)
    rr, cc = draw.polygon(snake[:, 0], snake[:, 1])
    mask[rr, cc] = True

    return mask


def level_set_segmentation(img_gray):
    init_level_set = checkerboard_level_set(img_gray.shape, 10)
    snake = morphological_geodesic_active_contour(
        gaussian(img_gray, sigma=3, preserve_range=False),
        num_iter=100,
        init_level_set=init_level_set)
    return snake


def graph_cut_segmentation(img_gray):
    # Note: The original code didn't include an actual Graph Cut implementation
    # This is a placeholder and should be replaced with an actual Graph Cut algorithm
    return img_gray > filters.threshold_otsu(img_gray)


def random_walker_segmentation(img_gray):
    smooth_img = filters.gaussian(img_gray, sigma=2)
    markers = np.zeros(smooth_img.shape, dtype=np.uint)
    markers[smooth_img < 0.6] = 1
    markers[smooth_img > 0.9] = 2
    labels = random_walker(smooth_img, markers, beta=10, mode='bf')
    return labels


def calculate_metrics(segmentation, ground_truth):
    # Resize segmentation to match ground_truth if shapes are different
    if segmentation.shape != ground_truth.shape:
        segmentation = resize(segmentation, ground_truth.shape, order=0, preserve_range=True, anti_aliasing=False)

    # Convert segmentation to binary mask if it's not already
    if not np.array_equal(segmentation, segmentation.astype(bool)):
        segmentation = segmentation > 0

    # Ensure both arrays are flattened and of boolean type
    y_true = ground_truth.flatten().astype(bool)
    y_pred = segmentation.flatten().astype(bool)

    f1 = f1_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return f1, rec


def create_mock_ground_truth(img_gray):
    # This creates a simple circular ground truth
    rows, cols = img_gray.shape
    center_row, center_col = rows // 2, cols // 2
    radius = min(rows, cols) // 4
    y, x = np.ogrid[:rows, :cols]
    mask_area = (y - center_row) ** 2 + (x - center_col) ** 2 <= radius ** 2
    return mask_area


def main():
    img = load('../img.png')
    img, img_gray = preprocess_image(img)

    # Placeholder for ground truth segmentation
    # Replace this with actual ground truth data
    ground_truth = create_mock_ground_truth(img_gray)

    methods = [
        ("Active Contour", active_contour_segmentation),
        ("Level Set", level_set_segmentation),
        ("Graph Cut", graph_cut_segmentation),
        ("Random Walker", random_walker_segmentation)
    ]

    results = []

    for name, method in methods:
        start_time = time.time()
        if name == "Active Contour":
            segmentation = method(img, img_gray)
        else:
            segmentation = method(img_gray)
        end_time = time.time()
        run_time = end_time - start_time

        f1, recall = calculate_metrics(segmentation, ground_truth)

        results.append({
            "Method": name,
            "F1-score": f1,
            "Recall": recall,
            "Run-time": run_time
        })

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        ax2.imshow(segmentation.astype(float), cmap='gray')
        ax2.set_title(f'{name} Segmentation')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

    # Print performance matrix
    print("\nPerformance Matrix:")
    print("{:<15} {:<10} {:<10} {:<10}".format("Method", "F1-score", "Recall", "Run-time"))
    print("-" * 45)
    for result in results:
        print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            result["Method"], result["F1-score"], result["Recall"], result["Run-time"]))

    # Determine the best method
    best_method = max(results, key=lambda x: x['F1-score'])
    print(f"\nBest performing method: {best_method['Method']}")
    print(f"F1-score: {best_method['F1-score']:.4f}")
    print(f"Recall: {best_method['Recall']:.4f}")
    print(f"Run-time: {best_method['Run-time']:.4f} seconds")

    # Plot performance comparison
    methods = [result['Method'] for result in results]
    f1_scores = [result['F1-score'] for result in results]
    recalls = [result['Recall'] for result in results]
    run_times = [result['Run-time'] for result in results]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    rects1 = ax1.bar(x - width, f1_scores, width, label='F1-score', color='b', alpha=0.7)
    rects2 = ax1.bar(x, recalls, width, label='Recall', color='g', alpha=0.7)
    rects3 = ax2.bar(x + width, run_times, width, label='Run-time', color='r', alpha=0.7)

    ax1.set_ylabel('Score')
    ax2.set_ylabel('Run-time (seconds)')
    ax1.set_title('Performance Comparison of Segmentation Methods')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, max(run_times) * 1.2)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
