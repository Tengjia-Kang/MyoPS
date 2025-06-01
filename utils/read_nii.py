import nibabel as nib
import numpy as np

# 加载图像
# nii_path = '../datasets/Competition_Dataset/Test/Test/Case4001/Case4001_C0.nii.gz'
nii_path = '../datasets/MyoPS 2020 Dataset/MyoPS 2020 Dataset/test20/test20/myops_test_201_C0.nii.gz'
nii_img = nib.load(nii_path)

# 获取 numpy 数组
img_data = nii_img.get_fdata()  # shape: [H, W, D]，一般是 (256, 256, 50)
print(img_data.shape)

import matplotlib.pyplot as plt

# 获取中间切片
middle_slice = img_data[:, :, 4]

# cmap 将数值映射到颜色
plt.imshow(middle_slice, cmap='gray')
# plt.imshow(img_data, cmap='CMRmap')
plt.title('slice')
plt.axis('off')
plt.show()
