from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from background import estimate_background
# 读取 FITS 文件
fits_file = 'image/20240902165831475_0100_0260_0300.FITS'
hdul = fits.open(fits_file)
image = hdul[0].data  # 获取图像数据

# 确保图像是二维的
if len(image.shape) == 2:
    background_map, rms_map = estimate_background(image)
else:
    print("图像数据必须是二维的")

# 显示背景图和噪声图
fig, axs = plt.subplots(1, 2 , figsize=(12, 6))

# 显示背景图
axs[0].imshow(background_map, cmap='gray')
axs[0].set_title('Background Map')
axs[0].axis('off')

# 显示噪声图
axs[1].imshow(rms_map, cmap='gray')
axs[1].set_title('RMS Map')
axs[1].axis('off')

# axs[2].imshow(image, cmap='gray')
# axs[2].set_title('Original Image')
# axs[2].axis('off')

plt.tight_layout()
plt.show()
