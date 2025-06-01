import os
import random
import nibabel as nib
import re
def get_slice_count(nii_path):
    """读取 nii 文件并获取切片数量"""
    img = nib.load(nii_path)
    return img.shape[2]  # 假设 z 轴为切片方向

def write_txt(file_list, save_path, base_dir):
    with open(save_path, 'w') as f:
        for case in file_list:
            case_suffix = case + '_C0.nii.gz'
            nii_path = os.path.join(base_dir,case, case_suffix)
            # nii_path = os.path.join(base_dir, case_suffix)
            if not os.path.exists(nii_path):
                continue
            num_slices = get_slice_count(nii_path)
            folder_path = os.path.dirname(nii_path)
            for i in range(num_slices):
                # 格式：image_prefix label_prefix slice_index
                line = f"{folder_path}/{case} {folder_path}/{case} {i}\n"
                f.write(line)

def generate_split_txt(data_dir, output_dir):
    cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    cases.sort()
    random.shuffle(cases)
    os.makedirs(output_dir, exist_ok=True)
    write_txt(cases, os.path.join(output_dir, 'train.txt'), data_dir)



def generate_split_txt1(data_dir, output_dir):

    cases = [d for d in os.listdir(data_dir) if (os.path.join(data_dir, d)).endswith(('.nii', '.nii.gz'))]
    cases.sort()
    # 提取单一序号
    serial_numbers = []
    for case in cases:
        # 使用正则表达式匹配文件名中的序号（假设格式为 _training_数字_）
        match = re.search(r'_training_(\d+)_', case)
        if match:
            serial_numbers.append(match.group(1))  # 提取匹配的数字部分
    serial_numbers = list(set(serial_numbers))

    case_prefix = case[:len('myops_training_')]
    cases = list( (case_prefix) + index for index in serial_numbers)
    # random.shuffle(cases)
    os.makedirs(output_dir, exist_ok=True)

    write_txt(cases, os.path.join(output_dir, 'train.txt'), data_dir)


# ====== 用法示例 ======
if __name__ == '__main__':
    generate_split_txt(
        data_dir='datasets/Competition_Dataset/Train',
        output_dir='datasets/Competition_Dataset/Train'
    )

    # generate_split_txt1(
    #     data_dir='datasets/MyoPS_2020_Dataset/train25',
    #     output_dir='datasets/MyoPS_2020_Dataset/train25'
    # )
