import argparse
import numpy as np
from PIL import Image
from sklearn import mixture, cluster
from scipy.spatial import distance
from scipy.ndimage import zoom
import skimage.measure as skim_measure


def show_all(palette: np.ndarray, image: np.ndarray, mapped_image_pil: Image) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gspec

    fig = plt.figure(figsize=(12, 6))
    gs = gspec.GridSpec(2, 2, width_ratios=[4, 4])

    # Original Image
    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(image)
    ax0.axis('off')
    ax0.set_title("Original Image")

    # Pooled and Mapped Image
    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(mapped_image_pil)
    ax1.axis('off')
    ax1.set_title("Pooled and Mapped Image")

    # Palette
    ax2 = plt.subplot(gs[1, :])
    ax2.imshow([palette], aspect='auto')
    ax2.axis('off')
    ax2.set_title("Palette")

    plt.tight_layout()
    plt.show()


def process_image(im_path: str, ds: int, clusters: int, method: str, pooling: str, save: bool):

    def gmm(pixels: np.ndarray, clusters: int) -> np.ndarray:
        model = mixture.GaussianMixture(n_components=clusters, covariance_type="full").fit(pixels)
        return model.means_.astype(np.uint8)

    def kmeans(pixels: np.ndarray, clusters: int) -> np.ndarray:
        model = cluster.KMeans(n_clusters=clusters).fit(pixels)
        return model.cluster_centers_.astype(np.uint8)

    clusterizer = {
        'gmm': gmm,
        'kmeans': kmeans,
    }

    pooling_methods = {
        'sum': np.sum, 
        'min': np.amin, 
        'max': np.amax, 
        'mean' : np.mean, 
        'median' : np.median
    }

    image = Image.open(im_path).convert('RGB')
    input_arr = np.array(image)
    pixels = input_arr.reshape((-1, 3))

    palette = clusterizer[method](pixels, clusters)

    pooled = skim_measure.block_reduce(input_arr, block_size=(ds, ds, 1), func=pooling_methods[pooling])
    pooled_flat = pooled.reshape((-1, 3))

    mapped_colors = np.array([palette[distance.cdist([color], palette).argmin()] for color in pooled_flat])

    mapped_image = mapped_colors.reshape(pooled.shape)
    upscaled_image = zoom(mapped_image, (ds, ds, 1), order=0)
    mapped_image_pil = Image.fromarray(upscaled_image.astype('uint8'), 'RGB')

    if save:
        mapped_image_pil.save(f'{im_path[:-4]}_mapped_{ds}_{method}_{clusters}_{pooling}.png')

    show_all(palette, image, mapped_image_pil)


def main():
    parser = argparse.ArgumentParser(description='Pixelize a image with python.')
    parser.add_argument('im_path', type=str, help='Path to the input image')
    parser.add_argument('--ds', type=int, default=4, help='Downsample factor')
    parser.add_argument('--clusters', type=int, default=16, help='Number of clusters used to extract the palette')
    parser.add_argument('--method', type=str, default='kmeans', choices=['gmm', 'kmeans'], help='Clustering method')
    parser.add_argument('--pooling', type=str, default='mean', choices=['sum', 'min', 'max', 'mean', 'median'], help='Function applied to pooling')
    parser.add_argument('--save', action='store_true', help='Save the image')

    args = parser.parse_args()
    process_image(args.im_path, args.ds, args.clusters, args.method, args.pooling, args.save)


if __name__ == '__main__':
    main()
