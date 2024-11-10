import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, color, graph
from sklearn.metrics import f1_score, recall_score
import time
from scipy import stats


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to load image from path: {image_path}")

    if image.shape[2] == 4:  # Check if image has 4 channels (RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image, gray


def segment_image(image_path):
    image, gray = load_image(image_path)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    segmented_bw = np.zeros_like(gray)
    segmented_bw[markers > 1] = 255
    return segmented_bw


def seeded_region_growing(image_path, seed_point=(100, 100), threshold=15):
    _, gray = load_image(image_path)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[seed_point] = 255
    seed_queue = [seed_point]
    while seed_queue:
        x, y = seed_queue.pop(0)
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = x + i, y + j
                if 0 <= nx < gray.shape[0] and 0 <= ny < gray.shape[1]:
                    if mask[nx, ny] == 0 and abs(int(gray[nx, ny]) - int(gray[x, y])) < threshold:
                        mask[nx, ny] = 255
                        seed_queue.append((nx, ny))
    return mask


def region_splitting_and_merging(image_path):
    def split_region(image, x, y, size):
        if size < 1:
            return
        region = image[y:y + size, x:x + size]
        mean_intensity = np.mean(region)
        std_dev = np.std(region)
        if std_dev < 5:
            segmented[y:y + size, x:x + size] = mean_intensity
        else:
            half_size = size // 2
            split_region(image, x, y, half_size)
            split_region(image, x + half_size, y, half_size)
            split_region(image, x, y + half_size, half_size)
            split_region(image, x + half_size, y + half_size, half_size)

    _, gray = load_image(image_path)
    segmented = np.zeros_like(gray, dtype=np.float32)
    split_region(gray, 0, 0, gray.shape[0])
    return (segmented * 255).astype(np.uint8)


def clustering_based_segmentation(image_path, k=3):
    image, _ = load_image(image_path)
    image_reshaped = image.reshape((-1, 3))
    image_reshaped = np.float32(image_reshaped)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(image_reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    segmented_image = centers[labels].reshape(image.shape).astype(np.uint8)
    return cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)


def mean_shift_segmentation(image_path):
    image, _ = load_image(image_path)
    shifted = cv2.pyrMeanShiftFiltering(image, sp=21, sr=51)
    return cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)


def watershed_segmentation(image_path):
    return segment_image(image_path)


def graph_based_segmentation(image_path):
    image, _ = load_image(image_path)
    labels = segmentation.slic(image, compactness=30, n_segments=400)
    g = graph.rag_mean_color(image, labels)
    labels2 = graph.cut_threshold(labels, g, 0.25)
    segmented = color.label2rgb(labels2, image, kind='avg')
    return cv2.cvtColor((segmented * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def evaluate_performance(original, segmented):
    # Ensure both images have the same shape
    min_shape = min(original.shape, segmented.shape)
    original = original[:min_shape[0], :min_shape[1]]
    segmented = segmented[:min_shape[0], :min_shape[1]]

    # Normalize and binarize the images
    original_bin = (original > original.mean()).astype(int)
    segmented_bin = (segmented > segmented.mean()).astype(int)

    original_flat = original_bin.flatten()
    segmented_flat = segmented_bin.flatten()

    f1 = f1_score(original_flat, segmented_flat, average='weighted', zero_division=1)
    recall = recall_score(original_flat, segmented_flat, average='weighted', zero_division=1)
    return f1, recall


def main():
    image_path = r'../img.png'
    original_image, gray_image = load_image(image_path)

    segmentation_methods = {
        'Seeded Region Growing': seeded_region_growing,
        'Region Splitting and Merging': region_splitting_and_merging,
        'Clustering Based': clustering_based_segmentation,
        'Mean Shift': mean_shift_segmentation,
        'Watershed': watershed_segmentation,
        'Graph Based': graph_based_segmentation
    }

    results = {}

    plt.figure(figsize=(20, 15))
    plt.subplot(3, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    for idx, (name, method) in enumerate(segmentation_methods.items(), start=2):
        start_time = time.time()
        segmented = method(image_path)
        execution_time = time.time() - start_time

        f1, recall = evaluate_performance(gray_image, segmented)

        results[name] = {
            'F1 Score': f1,
            'Recall Score': recall,
            'Execution Time': execution_time
        }

        plt.subplot(3, 3, idx)
        plt.imshow(segmented, cmap='gray')
        plt.title(f'{name}\nF1: {f1:.4f}, Recall: {recall:.4f}\nTime: {execution_time:.4f}s')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Plotting performance comparison
    methods = list(results.keys())
    f1_scores = [results[m]['F1 Score'] for m in methods]
    recall_scores = [results[m]['Recall Score'] for m in methods]
    execution_times = [results[m]['Execution Time'] for m in methods]

    plt.figure(figsize=(15, 5))
    x = np.arange(len(methods))
    width = 0.25

    plt.bar(x - width, f1_scores, width, label='F1 Score')
    plt.bar(x, recall_scores, width, label='Recall Score')
    plt.bar(x + width, execution_times, width, label='Execution Time (s)')

    plt.xlabel('Segmentation Methods')
    plt.ylabel('Scores / Time')
    plt.title('Performance Comparison of Segmentation Methods')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Summary
    best_f1 = max(results, key=lambda x: results[x]['F1 Score'])
    best_recall = max(results, key=lambda x: results[x]['Recall Score'])
    best_time = min(results, key=lambda x: results[x]['Execution Time'])

    print(f"Best method based on F1 Score: {best_f1}")
    print(f"Best method based on Recall Score: {best_recall}")
    print(f"Best method based on Execution Time: {best_time}")


if __name__ == "__main__":
    main()