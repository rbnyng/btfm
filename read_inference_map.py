import numpy as np
from matplotlib import pyplot as plt

# npy path
npy_path = "/maps/zf281/btfm-training/classification_map.npy"
data = np.load(npy_path)
plt.imshow(data)
# save as png
# plt.imsave("classification_map.png", data)