import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def normalize_image(image):
    """
    将图像标准化到0-1范围
    """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val != 0:
        return (image - min_val) / (max_val - min_val)
    return image

def save_slice_as_png(slice_data, output_path):
    """
    保存单个切片为PNG图像
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def convert_nii_to_png(input_path, output_dir):
    """
    将单个.nii.gz文件转换为PNG图像序列
    """
    # 加载nii文件
    nii_img = nib.load(input_path)
    img_data = nii_img.get_fdata()
    
    # 获取文件名（不包含扩展名）
    base_name = Path(input_path).stem
    if base_name.endswith('.nii'):  # 处理.nii.gz的情况
        base_name = base_name[:-4]
    
    # 创建输出目录
    output_subdir = os.path.join(output_dir, base_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    # 确定图像维度顺序
    dims = img_data.shape
    if len(dims) > 3:
        img_data = img_data[:, :, :, 0]  # 如果是4D图像，只取第一个时间点
    
    # 遍历每个切片
    total_slices = img_data.shape[2]  # 假设第三维是切片维度
    for z in range(total_slices):
        # 获取当前切片
        slice_data = img_data[:, :, z]
        
        # 标准化到0-1范围
        slice_data = normalize_image(slice_data)
        
        # 保存为PNG
        output_path = os.path.join(output_subdir, f'slice_{z:03d}.png')
        save_slice_as_png(slice_data, output_path)
        
        # 打印进度
        if (z + 1) % 10 == 0:
            print(f'Processing {base_name}: {z+1}/{total_slices} slices completed')

def batch_convert_folder(input_folder, output_folder):
    """
    批量转换文件夹中的所有.nii.gz文件
    """
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有.nii.gz文件
    nii_files = list(Path(input_folder).rglob('*.nii.gz'))
    total_files = len(nii_files)
    
    print(f'Found {total_files} .nii.gz files')
    
    # 处理每个文件
    for i, nii_file in enumerate(nii_files, 1):
        print(f'\nProcessing file {i}/{total_files}: {nii_file.name}')
        try:
            convert_nii_to_png(str(nii_file), output_folder)
            print(f'Successfully converted {nii_file.name}')
        except Exception as e:
            print(f'Error processing {nii_file.name}: {str(e)}')

def main():
    parser = argparse.ArgumentParser(description='Convert .nii.gz files to PNG images')
    parser.add_argument('input_path', help='Input .nii.gz file or folder containing .nii.gz files')
    parser.add_argument('output_dir', help='Output directory for PNG images')
    parser.add_argument('--single', action='store_true', help='Process single file instead of folder')
    
    args = parser.parse_args()
    
    if args.single:
        # 处理单个文件
        if not os.path.isfile(args.input_path):
            print(f'Error: {args.input_path} is not a file')
            return
        convert_nii_to_png(args.input_path, args.output_dir)
    else:
        # 处理整个文件夹
        if not os.path.isdir(args.input_path):
            print(f'Error: {args.input_path} is not a directory')
            return
        batch_convert_folder(args.input_path, args.output_dir)

if __name__ == '__main__':
    main() 