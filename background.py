import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline


def estimate_background(image, mesh_size=64, filter_size=3):
    """
    估计图像的背景和背景噪声。

    Parameters:
    - image: 输入图像 (2D numpy array)
    - mesh_size: 每个网格的大小（单位：像素）
    - filter_size: 滤波器的尺寸，用于平滑背景图

    Returns:
    - background_map: 平滑的背景图
    - rms_map: 背景噪声图
    """
    # 图像尺寸
    height, width = image.shape

    # 网格分块
    grid_y, grid_x = height // mesh_size, width // mesh_size

    # 初始化背景和RMS矩阵
    background = np.zeros((grid_y, grid_x))
    rms = np.zeros((grid_y, grid_x))

    # 遍历每个网格
    for i in range(grid_y):
        for j in range(grid_x):
            # 提取网格内的像素
            sub_image = image[i * mesh_size:(i + 1) * mesh_size,
                        j * mesh_size:(j + 1) * mesh_size]

            # 计算直方图的中位数和标准差
            median = np.median(sub_image)
            mean = np.mean(sub_image)
            sigma = np.std(sub_image)

            # 剪裁法估计背景
            clipped = sub_image[(sub_image > median - 3 * sigma) &
                                (sub_image < median + 3 * sigma)]
            if len(clipped) > 0:
                mean_clipped = np.mean(clipped)
                mode = 2.5 * np.median(clipped) - 1.5 * mean_clipped
            else:
                mode = median

            background[i, j] = mode
            rms[i, j] = np.std(clipped) if len(clipped) > 0 else sigma

    # 使用双三次样条插值生成全图背景
    y = np.linspace(0, grid_y - 1, grid_y) * mesh_size + mesh_size // 2
    x = np.linspace(0, grid_x - 1, grid_x) * mesh_size + mesh_size // 2
    spline = RectBivariateSpline(y, x, background)
    yy, xx = np.mgrid[0:height, 0:width]
    background_map = spline.ev(yy, xx)

    # 平滑背景图
    background_map = gaussian_filter(background_map, filter_size)

    # 同样处理RMS图
    rms_spline = RectBivariateSpline(y, x, rms)
    rms_map = rms_spline.ev(yy, xx)
    rms_map = gaussian_filter(rms_map, filter_size)

    return background_map, rms_map

# 示例用法
# 假设 `image` 是一张二维图像
# background_map, rms_map = estimate_background(image)
