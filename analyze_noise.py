from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

plt.rcParams["image.cmap"] = "gray"
from utils import get_data


image_noise_path = Path("images") / "02_25_noisy.png"
test_image = Image.open(image_noise_path)
test_image_array = np.array(test_image) / 255.0
_, _, image_ct = get_data(Path("data") / "test_dataset", 1)
image_ct_roi = image_ct[0][100:400, 100:400]

plt.subplot(121)
plt.title("DnCNN model test image")
plt.imshow(test_image_array)
plt.xlabel(f" $\sigma$={test_image_array.std():.4f}")
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(
    top=0.98, bottom=0.152, left=0.031, right=0.97, hspace=0.122, wspace=0.063
)

plt.suptitle("Noise levels")
plt.subplot(122)
plt.title("CT Scan image")
plt.imshow(image_ct_roi)
plt.xlabel(f" $\sigma$={image_ct_roi.std():.4f}")
plt.xticks([])
plt.yticks([])

plt.show()
