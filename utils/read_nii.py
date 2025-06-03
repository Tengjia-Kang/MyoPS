import nibabel as nib
import numpy as np

# 加载图像
# nii_path = 'datasets/Competition_Dataset/Train/Case3001/Case3001_gd.nii.gz'
nii_path = '../datasets/Competition_Dataset/Train/Case3001/Case3001_C0.nii.gz'
# nii_path = '../test/test_1/Case4022_result.nii.gz'
nii_path = 'inference_results/checkpoint_epoch130/Case3005_T2_seg.nii.gz'
nii_img = nib.load(nii_path)


# 获取 numpy 数组
img_data = nii_img.get_fdata()  # shape: [H, W, D]，一般是 (256, 256, 50)
print(img_data.shape)

import matplotlib.pyplot as plt

# 获取中间切片
middle_slice = img_data[:, :, 1]

# cmap 将数值映射到颜色
# plt.imshow(middle_slice, cmap='gray')
plt.imshow(middle_slice, cmap='CMRmap')
plt.title('slice')
plt.axis('off')
plt.show()
