import argparse
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageSequence
from sklearn import mixture, cluster
from scipy.spatial import distance
from scipy.ndimage import zoom
import skimage.measure as skim_measure


@dataclass
class PyxelizeArgs:
    path: str
    ds: int
    clusters: int
    method: str
    pooling: str
    save: bool
    plot: bool


def show_all(palette: np.ndarray, image: np.ndarray, mapped_image_pil: Image) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gspec

    fig = plt.figure(figsize=(12, 6))
    gs = gspec.GridSpec(2, 2, width_ratios=[4, 4])

    # Original Image
    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(image)
    ax0.axis("off")
    ax0.set_title("Original Image")

    # Pooled and Mapped Image
    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(mapped_image_pil)
    ax1.axis("off")
    ax1.set_title("Pooled and Mapped Image")

    # Palette
    ax2 = plt.subplot(gs[1, :])
    ax2.imshow([palette], aspect="auto")
    ax2.axis("off")
    ax2.set_title("Palette")

    plt.tight_layout()
    plt.show()


def palette_extractor(pixels: np.ndarray, clusters: int, method: str) -> np.ndarray:
    def gmm(pixels: np.ndarray, clusters: int) -> np.ndarray:
        model = mixture.GaussianMixture(
            n_components=clusters, covariance_type="full"
        ).fit(pixels)
        return model.means_.astype(np.uint8)

    def kmeans(pixels: np.ndarray, clusters: int) -> np.ndarray:
        model = cluster.KMeans(n_clusters=clusters).fit(pixels)
        return model.cluster_centers_.astype(np.uint8)

    clusterizer = {
        "gmm": gmm,
        "kmeans": kmeans,
    }
    return clusterizer[method](pixels, clusters)


def single_palettte_multiframe(path: str, clusters: int, method: str) -> np.ndarray:
    # Get all pixels in gif (multiple frames)
    with Image.open(path) as frames:
        all_pixels = []
        for frame in ImageSequence.Iterator(frames):
            flat_image = np.array(frame.convert("RGB")).reshape((-1, 3))
            all_pixels.append(flat_image)
        all_pixels_concat = np.concatenate(all_pixels)
        pallete = palette_extractor(all_pixels_concat, clusters, method)
        return pallete


def process_image(image: Image, palette: np.ndarray, pa: PyxelizeArgs) -> Image:

    pooling_methods = {
        "sum": np.sum,
        "min": np.amin,
        "max": np.amax,
        "mean": np.mean,
        "median": np.median,
    }

    # Flatten image
    input_arr = np.array(image)
    pixels = input_arr.reshape((-1, 3))

    # Extract palette from image unless unique palette for gifs is being used
    if palette is None:
        palette = palette_extractor(pixels, pa.clusters, pa.method)

    # Pool with given fn
    pooled = skim_measure.block_reduce(
        input_arr, block_size=(pa.ds, pa.ds, 1), func=pooling_methods[pa.pooling]
    )

    # Reflat & map
    pooled_flat = pooled.reshape((-1, 3))
    mapped_colors = np.array(
        [palette[distance.cdist([color], palette).argmin()] for color in pooled_flat]
    )

    # Upscale
    mapped_image = mapped_colors.reshape(pooled.shape)
    upscaled_image = zoom(mapped_image, (pa.ds, pa.ds, 1), order=0)

    mapped_image_pil = Image.fromarray(upscaled_image.astype("uint8"), "RGB")
    if pa.plot:
        show_all(palette, image, mapped_image_pil)
    return mapped_image_pil


def extract_frames(pa: PyxelizeArgs):

    save_str = f"{pa.path[:-4]}_mapped_{pa.ds}_{pa.method}_{pa.clusters}_{pa.pooling}"

    # Extract unique palette from all images composing the gif
    palette = single_palettte_multiframe(pa.path, pa.clusters, pa.method)

    # Process each image with same palette
    with Image.open(pa.path) as frames:
        framebuffer = []
        for frame in ImageSequence.Iterator(frames):
            mapped = process_image(frame.convert("RGB"), palette, pa)
            framebuffer.append(mapped)

        framebuffer[0].save(
            f"{save_str}.gif", save_all=True, append_images=framebuffer[1:], loop=0
        )


def extract_image(pa: PyxelizeArgs):
    mapped = process_image(Image.open(pa.path).convert("RGB"), None, pa)
    if pa.save:
        mapped.save(f"{pa.path[:-4]}_mapped_{pa.ds}_{pa.method}_{pa.clusters}_{pa.pooling}.png")


def main():

    parser = argparse.ArgumentParser(description="Pixelize a image with python.")
    parser.add_argument("path", type=str, help="Path to the input item (image or gif)")
    parser.add_argument("--ds", type=int, default=4, help="Downsample factor")
    parser.add_argument("--clusters", type=int, default=16, help="Number of clusters used to extract the palette")
    parser.add_argument("--method", type=str, default="kmeans", choices=["gmm", "kmeans"], help="Clustering method")
    parser.add_argument("--pooling", type=str, default="mean", choices=["sum", "min", "max", "mean", "median"], help="Function applied to pooling")
    parser.add_argument("--save", action="store_true", help="Save the image")
    parser.add_argument("--plot", action="store_true", help="Plot the image")
    args = parser.parse_args()

    input_type = {
        "jpg": extract_image, 
        "gif": extract_frames
    }
    pyx_args = PyxelizeArgs(args.path, args.ds, args.clusters, args.method, args.pooling, args.save, args.plot)
    input_type[args.path[-3:]](pyx_args)


if __name__ == "__main__":
    main()
