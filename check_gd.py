


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# 加载图像
nii_path = 'datasets/Competition_Dataset/Train/Case3001/Case3001_gd.nii.gz'
nii_img = nib.load(nii_path)
header = nii_img.header
print(header)
# 获取 numpy 数组
img = nii_img.get_fdata()  # shape: [H, W, D]，一般是 (256, 256, 50)

print("维度:", header.get_data_shape())
print("体素尺寸:", header.get_zooms())

# 显示中间层切片
slice_idx = img.shape[2] // 2
plt.imshow(img[:, :, slice_idx], cmap='CMRmap')
plt.colorbar()
plt.show()