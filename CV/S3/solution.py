# pylint: disable=no-member
import cv2
import numpy as np
import matplotlib.pyplot as plt


def box(n: int) -> np.ndarray:
    if n % 2 == 0:
        raise ValueError("Kernel size must be odd")

    kernel = np.ones((n, n), np.float32) / (n * n)
    return kernel.astype(np.float32)


def bartlett(n: int) -> np.ndarray:
    if n % 2 == 0:
        raise ValueError("Kernel size must be odd")

    l = [x for x in range(1, (n + 1) // 2)]
    l += [x for x in range((n + 1) // 2, 0, -1)]

    kernel = [[l[i] * l[j] for j in range(n)] for i in range(n)]
    total_sum = np.sum(kernel)

    kernel = np.array(kernel) / total_sum

    return kernel.astype(np.float32)


def gaussian(n: int) -> np.ndarray:
    if n % 2 == 0:
        raise ValueError("Kernel size must be odd")

    coef = [1]
    for i in range(1, n):
        new_row = [1]
        for j in range(1, i):
            new_row.append(coef[j - 1] + coef[j])
        new_row.append(1)
        coef = new_row

    kernel = [[coef[i] * coef[j] for j in range(n)] for i in range(n)]
    total_sum = np.sum(kernel)

    kernel = np.array(kernel) / total_sum

    return kernel.astype(np.float32)


def laplacian(n: int) -> np.ndarray:
    if n % 2 == 0:
        raise ValueError("Kernel size must be odd")

    if n == 3:
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    elif n == 5:
        kernel = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 2, 1, 0],
                [1, 2, -17, 2, 1],
                [0, 1, 2, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            np.float32,
        )
    else:
        raise ValueError("Kernel size must be 3 or 5")

    return kernel.astype(np.float32)


def convolution(img: np.ndarray, kernel: np.ndarray, padding: int):
    width, height = img.shape[:2]
    isGray = len(img.shape) == 2

    if isGray:
        out = np.zeros((width, height), np.float32)
    else:
        out = np.zeros((width, height, 3), np.float32)

    for i in range(padding, width - padding):
        for j in range(padding, height - padding):
            if isGray:
                region = img[
                    i - padding : i + padding + 1, j - padding : j + padding + 1
                ]
                filtered_value = np.sum(region * kernel)
                out[i, j] = filtered_value
            else:
                for k in range(3):
                    region = img[
                        i - padding : i + padding + 1, j - padding : j + padding + 1, k
                    ]
                    filtered_value = np.sum(region * kernel)
                    out[i, j, k] = filtered_value

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def filter(
    img: np.ndarray, type_kernel: str, kernel_size: int, padding_strategy: str
) -> np.ndarray:

    width, height = img.shape[:2]
    padding = kernel_size // 2

    # Paddings
    if padding_strategy == "constant":
        img = np.pad(
            img,
            pad_width=((padding, padding), (padding, padding), (0, 0)),
            mode="constant",
        )
    elif padding_strategy == "edge":
        img = np.pad(
            img, pad_width=((padding, padding), (padding, padding), (0, 0)), mode="edge"
        )
    elif padding_strategy == "reflex":
        img = np.pad(
            img,
            pad_width=((padding, padding), (padding, padding), (0, 0)),
            mode="reflect",
        )

    # Kernels
    if type_kernel == "box":
        kernel = box(kernel_size)
    elif type_kernel == "bartlett":
        kernel = bartlett(kernel_size)
    elif type_kernel == "gaussian":
        kernel = gaussian(kernel_size)
    elif type_kernel == "laplacian":
        kernel = laplacian(kernel_size)

    img = convolution(img, kernel, padding)

    # Ajustar el recorte para asegurar que las dimensiones coincidan con la original
    img = img[padding : padding + width, padding : padding + height]
    return img.astype(np.uint8)


img = cv2.imread("./liam3.jpeg")
img = filter(img, "gaussian", 17, "edge")

img2 = cv2.imread("./liam3.jpeg")
img2 = filter(img2, "gaussian", 7, "edge")

# Resta de las dos imagenes gaussianas (DoG: Difference of Gaussians).
# Se usa int16 para evitar desbordamiento con valores negativos.
diff = np.int16(img) - np.int16(img2)

# Normaliza al rango [0, 255] para poder visualizarlo.
diff_display = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
diff_rgb = cv2.cvtColor(diff_display, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

axes[0].imshow(img_rgb)
axes[0].set_title("Gaussiano 17")
axes[0].axis("off")

axes[1].imshow(img2_rgb)
axes[1].set_title("Gaussiano 7")
axes[1].axis("off")

axes[2].imshow(diff_rgb)
axes[2].set_title("Diferencia (17 - 7)")
axes[2].axis("off")

plt.show()


# for i in range(3, 26, 2):
#   img = cv2.imread('output/images/lenna_gray.png')
#   filtered_img = filter(img, 'gaussian', i, 'edge')
#   cv2.imwrite(f'output/images/gaussian_gray_{i}.png', filtered_img)
#   print(f'Image processed with kernel size {i} and saved as gaussian_gray_{i}.png')

# img = cv2.imread('../lenna.png')
# filtered_img = filter(img, 'laplacian', 5, 'edge')
# cv2.imshow('Laplacian Filter', filtered_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
