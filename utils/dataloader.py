import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from utils.tools import Normalization, ImageTransform, LabelTransform
    
#
# class CrossModalDataLoader(Dataset):
#
#     def __init__(self, path, file_name, dim, max_iters = None, stage = 'Train'):
#
#         self.path = path
#         self.crop_size = dim
#         self.stage = stage
#
#         self.Img = [item.strip().split() for item in open(self.path + file_name)]
#
#         if max_iters != None:
#             self.Img = self.Img * int(np.ceil(float(max_iters) / len(self.Img)))
#
#         self.files = []
#
#         for item in self.Img:
#
#             img_path, gt_path, imgidx = item
#
#             C0_path = img_path + '_C0.nii.gz'
#             LGE_path = img_path + '_LGE.nii.gz'
#             T2_path = img_path + '_T2.nii.gz'
#             T1m_path = img_path + '_T1m.nii.gz'
#             T2starm_path = img_path + '_T2starm.nii.gz'
#             label_path = gt_path + '_gd.nii.gz'
#
#             # C0_file = os.path.join(self.path, C0_path)
#             # LGE_file = os.path.join(self.path, LGE_path)
#             # T2_file = os.path.join(self.path, T2_path)
#             # T1m_file = os.path.join(self.path, T1m_path)
#             # T2starm_file = os.path.join(self.path, T2starm_path)
#             # label_file = os.path.join(self.path, label_path)
#
#             C0_file = os.path.join(C0_path)
#             LGE_file = os.path.join(LGE_path)
#             T2_file = os.path.join(T2_path)
#             T1m_file = os.path.join(T1m_path)
#             T2starm_file = os.path.join(T2starm_path)
#             label_file = os.path.join(label_path)
#             self.files.append({
#                 "C0": C0_file,
#                 "LGE": LGE_file,
#                 "T2": T2_file,
#                 "T1m": T1m_file,
#                 "T2starm": T2starm_file,
#                 "label": label_file,
#                 "index": int(imgidx)
#             })
#
#         self.normalize = Normalization()
#         self.image_transform = ImageTransform(self.crop_size, self.stage)
#         self.label_transform = LabelTransform(self.stage)
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, index):
#
#         file_path = self.files[index]
#
#         # get raw data
#         C0_raw = nib.load(file_path["C0"])
#         LGE_raw = nib.load(file_path["LGE"])
#         T2_raw = nib.load(file_path["T2"])
#         T1m_raw = nib.load(file_path["T1m"])
#         T2starm_raw = nib.load(file_path["T2starm"])
#         gd_raw = nib.load(file_path["label"])
#         imgidx = file_path["index"]
#
#         # get data [x,y,z] & normalize
#         C0_img = self.normalize(C0_raw.get_fdata(),'Truncate')
#         LGE_img = self.normalize(LGE_raw.get_fdata(),'Truncate')
#         T2_img = self.normalize(T2_raw.get_fdata(),'Truncate')
#         T1m_img = T1m_raw.get_fdata()
#         if int(T1m_img.max()) != 0:
#             T1m_img = self.normalize(T1m_img,'Truncate')
#         T2starm_img = self.normalize(T2starm_raw.get_fdata(),'Truncate')
#         gd_img = gd_raw.get_fdata()
#
#         # cut slice [x,y,1] -> [x,y,5]
#         C0_slice = C0_img[:,:,imgidx:imgidx+1].astype(np.float32)
#         LGE_slice = LGE_img[:,:,imgidx:imgidx+1].astype(np.float32)
#         T2_slice = T2_img[:,:,imgidx:imgidx+1].astype(np.float32)
#         T1m_slice = T1m_img[:,:,imgidx:imgidx+1].astype(np.float32)
#         T2starm_slice = T2starm_img[:,:,imgidx:imgidx+1].astype(np.float32)
#         label_slice = gd_img[:,:,imgidx:imgidx+1].astype(np.float32)
#         image = np.concatenate([C0_slice,LGE_slice,T2_slice,T1m_slice,T2starm_slice,label_slice], axis=2)
#         img_C0, img_LGE, img_T2, img_T1m, img_T2starm, label = torch.chunk(self.image_transform(image), chunks=6, dim=0)
#
#         img_C0 = self.normalize(img_C0, 'Zero_Mean_Unit_Std')
#         img_LGE = self.normalize(img_LGE, 'Zero_Mean_Unit_Std')
#         img_T2 = self.normalize(img_T2, 'Zero_Mean_Unit_Std')
#         img_T1m = self.normalize(img_T1m, 'Zero_Mean_Unit_Std')
#         img_T2starm = self.normalize(img_T2starm, 'Zero_Mean_Unit_Std')
#
#         # label transform [class,H,W]
#         label_cardiac, label_scar, label_edema, label_pathology = self.label_transform(label)
#
#         return img_C0, img_LGE, img_T2, img_T1m, img_T2starm, label_cardiac, label_scar, label_edema, label_pathology


class CrossModalDataLoader(Dataset):

    def __init__(self, path, file_name, dim, max_iters=None, stage='Train', modalities=None):
        self.path = path
        self.crop_size = dim
        self.stage = stage
        self.modalities = modalities if modalities is not None else ['C0', 'LGE', 'T2', 'T1m', 'T2starm']

        # 读取训练集列表
        self.Img = [item.strip().split() for item in open(self.path + file_name)]
        if max_iters is not None:
            # 如果max_iters不为None，则将训练集列表重复max_iters/len(self.Img)次
            self.Img = self.Img * int(np.ceil(float(max_iters) / len(self.Img)))

        self.files = []
        for item in self.Img:
            img_path, gt_path, imgidx = item # 图片路径，标签路径，图片索引(切片索引)
            file_dict = {'index': int(imgidx), 'label': os.path.join(gt_path + '_gd.nii.gz')} # 标签路径
            if 'C0' in self.modalities:
                file_dict['C0'] = os.path.join(img_path + '_C0.nii.gz')
            if 'LGE' in self.modalities:
                file_dict['LGE'] = os.path.join(img_path + '_LGE.nii.gz')
            if 'T2' in self.modalities:
                file_dict['T2'] = os.path.join(img_path + '_T2.nii.gz')
            if 'T1m' in self.modalities:
                file_dict['T1m'] = os.path.join(img_path + '_T1m.nii.gz')
            if 'T2starm' in self.modalities:
                file_dict['T2starm'] = os.path.join(img_path + '_T2starm.nii.gz')
            self.files.append(file_dict)

        self.normalize = Normalization()
        self.image_transform = ImageTransform(self.crop_size, self.stage)
        self.label_transform = LabelTransform(self.stage)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        img_slices = []
        # 读取index切片 各个模态的图像 [(H,W,1), (H,W,1), (H,W,1)]
        for modality in self.modalities:
            raw_data = nib.load(file_path[modality]).get_fdata()
            
            if modality == 'T1m' and int(raw_data.max()) != 0:
                raw_data = self.normalize(raw_data, 'Truncate')
            elif modality != 'T1m':
                raw_data = self.normalize(raw_data, 'Truncate')
            slice_data = raw_data[:, :, file_path['index']:file_path['index']+1].astype(np.float32)
            img_slices.append(slice_data)

        # 拼接图像 & 标签
        label_raw = nib.load(file_path['label']).get_fdata()
        label_slice = label_raw[:, :, file_path['index']:file_path['index']+1].astype(np.float32)
        image = np.concatenate(img_slices + [label_slice], axis=2)

        # transform
        transformed = self.image_transform(image)

        # 沿批次维度（dim=0）将 transformed 张量分块，每个块对应一个模态，变成tuple
        imgs = torch.chunk(transformed, chunks=len(self.modalities) + 1, dim=0)
        # 数据增强后的不同模态的图片
        img_tensors = [self.normalize(img, 'Zero_Mean_Unit_Std') for img in imgs[:-1]]

        label = self.label_transform(imgs[-1])
        return (*img_tensors, *label)

class MultiModalCardiacDataset(Dataset):
    def __init__(self, data_root, mode='train', transform=None):
        """
        多模态心脏分割数据集
        
        Args:
            data_root: 数据根目录
            mode: 'train' 或 'test'
            transform: 数据增强转换
        """
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.normalize = Normalization()
        
        # 获取所有case
        self.cases = []
        for case in os.listdir(data_root):
            if case.startswith('Case'):
                case_dir = os.path.join(data_root, case)
                if os.path.isdir(case_dir):
                    self.cases.append(case)
        self.cases.sort()
        
    def __len__(self):
        return len(self.cases)
    
    def load_nii(self, file_path):
        """加载nii.gz文件"""
        img = nib.load(file_path)
        return img.get_fdata()
    
    def process_label(self, label):
        """处理标签为三种模式"""
        cardiac_label = np.zeros_like(label)
        scar_label = np.zeros_like(label)
        edema_label = np.zeros_like(label)
        
        # Cardiac模式
        cardiac_label[label == 200] = 1  # 心肌
        cardiac_label[label == 500] = 2  # 心腔
        
        # Scar模式
        scar_label[label == 200] = 1     # 心肌
        scar_label[label == 1220] = 1    # 心肌
        scar_label[label == 600] = 2     # 疤痕
        
        # Edema模式
        edema_label[label == 200] = 1    # 心肌水肿
        edema_label[label == 1220] = 2   # 水肿
        edema_label[label == 2221] = 2   # 远端水肿
        
        return cardiac_label, scar_label, edema_label
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        case_dir = os.path.join(self.data_root, case)
        
        # 加载三个模态的图像
        C0_path = os.path.join(case_dir, f'{case}_C0.nii.gz')
        LGE_path = os.path.join(case_dir, f'{case}_LGE.nii.gz')
        T2_path = os.path.join(case_dir, f'{case}_T2.nii.gz')
        
        C0_img = self.load_nii(C0_path)
        LGE_img = self.load_nii(LGE_path)
        T2_img = self.load_nii(T2_path)
        
        # 标准化
        C0_img = self.normalize(C0_img, 'Zero_Mean_Unit_Std')
        LGE_img = self.normalize(LGE_img, 'Zero_Mean_Unit_Std')
        T2_img = self.normalize(T2_img, 'Zero_Mean_Unit_Std')
        
        # 如果是训练模式，加载标签
        if self.mode == 'train':
            label_path = os.path.join(case_dir, f'{case}_gd.nii.gz')
            label = self.load_nii(label_path)
            cardiac_label, scar_label, edema_label = self.process_label(label)
            
            # 转换为tensor
            C0_img = torch.from_numpy(C0_img).float()
            LGE_img = torch.from_numpy(LGE_img).float()
            T2_img = torch.from_numpy(T2_img).float()
            cardiac_label = torch.from_numpy(cardiac_label).long()
            scar_label = torch.from_numpy(scar_label).long()
            edema_label = torch.from_numpy(edema_label).long()
            
            # 应用数据增强
            if self.transform:
                data = {
                    'C0': C0_img,
                    'LGE': LGE_img,
                    'T2': T2_img,
                    'cardiac': cardiac_label,
                    'scar': scar_label,
                    'edema': edema_label
                }
                data = self.transform(data)
                C0_img = data['C0']
                LGE_img = data['LGE']
                T2_img = data['T2']
                cardiac_label = data['cardiac']
                scar_label = data['scar']
                edema_label = data['edema']
            
            return {
                'C0': C0_img,
                'LGE': LGE_img,
                'T2': T2_img,
                'cardiac_label': cardiac_label,
                'scar_label': scar_label,
                'edema_label': edema_label,
                'case_name': case
            }
        
        else:  # test模式
            # 转换为tensor
            C0_img = torch.from_numpy(C0_img).float()
            LGE_img = torch.from_numpy(LGE_img).float()
            T2_img = torch.from_numpy(T2_img).float()
            
            return {
                'C0': C0_img,
                'LGE': LGE_img,
                'T2': T2_img,
                'case_name': case
            }
