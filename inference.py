import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from network.model import MyoPSNet
from utils.tools import Normalization, ImageTransform, ResultTransform
from process import LargestConnectedComponents
from utils.config import config
import matplotlib.pyplot as plt
from utils.post_process import process_3d_predictions
from utils.dataloader import MultiModalCardiacDataset
from torch.utils.data import DataLoader

def mask_to_rgb(mask):
    """
    将分割掩码转换为RGB图像
    mask: [C, H, W]，C = num_classes（通常为3）
    返回: [H, W, 3]，RGB格式
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu()
        mask = torch.argmax(mask, dim=0).numpy()  # [H, W]
    elif isinstance(mask, np.ndarray):
        mask = np.argmax(mask, axis=0)  # [H, W]
    
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # 类别颜色映射
    colors = {
        0: (0, 0, 0),      # 背景 - 黑色
        1: (255, 0, 0),    # 类别1 - 红色
        2: (0, 255, 0),    # 类别2 - 绿色
    }

    for cls_id, color in colors.items():
        rgb[mask == cls_id] = color

    return rgb

def save_visualization(img_data, seg_data, save_path, slice_idx):
    """
    保存原始图像和分割结果的可视化图
    """
    plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(131)
    plt.imshow(img_data, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # 显示分割结果
    plt.subplot(132)
    rgb_mask = mask_to_rgb(seg_data)
    plt.imshow(rgb_mask)
    plt.title('Segmentation')
    plt.axis('off')
    
    # 显示叠加结果
    plt.subplot(133)
    plt.imshow(img_data, cmap='gray')
    plt.imshow(rgb_mask, alpha=0.3)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class CardiacInference:
    def __init__(self, model_path, config):
        self.config = config
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化工具
        self.normalize = Normalization()
        self.keepLCC = LargestConnectedComponents()
        self.image_transform = ImageTransform(config.dim, 'Test')
        self.result_transform = ResultTransform(ToOriginal=True)
        
        # 加载模型
        self.model = self._load_model()
        
        # 标签映射
        self.label_mapping = {
            'cardiac': {1: 200, 2: 500},  # Myocardium, LV
            'scar': {1: 600},             # Scar
            'edema': {1: 1220, 2: 2221}   # Myocardial Edema, Edema
        }
        
        # 添加可视化输出目录
        self.vis_dir = os.path.join('inference_results', 'visualization')
        os.makedirs(self.vis_dir, exist_ok=True)
        
    def _load_model(self):
        """加载模型"""
        print(f"Loading model from {self.model_path}")
        model = MyoPSNet(in_chs=(3, 2, 2), out_chs=(3, 3, 3)).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print("Model loaded successfully")
        return model
    
    def resize_image(self, image, target_size):
        """调整图像尺寸到目标大小"""
        if image.shape != target_size:
            # 转换为tensor并添加维度
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
            # 使用双线性插值进行resize
            resized = F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)
            return resized.squeeze().numpy()
        return image

    def resize_back(self, pred, original_size):
        """将预测结果调整回原始尺寸"""
        if pred.shape != original_size:
            # 转换为tensor并添加维度
            pred_tensor = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
            # 使用最近邻插值以保持标签值
            resized = F.interpolate(pred_tensor, size=original_size, mode='nearest')
            return resized.squeeze().numpy()
        return pred

    def preprocess_image(self, image_path):
        """预处理单个图像"""
        # 加载并标准化图像
        image = nib.load(image_path)
        image_data = self.normalize(image.get_fdata(), 'Truncate').astype(np.float32)
        print(f"\nPreprocessed image stats for {image_path}:")
        print(f"min={image_data.min():.3f}, max={image_data.max():.3f}, mean={image_data.mean():.3f}")
        return image_data, image.affine

    def process_slice(self, C0_slice, LGE_slice, T2_slice, original_size):
        """处理单个切片"""
        # 调整图像尺寸到模型输入大小
        target_size = (self.config.dim, self.config.dim)
        C0_resized = self.resize_image(C0_slice, target_size)
        LGE_resized = self.resize_image(LGE_slice, target_size)
        T2_resized = self.resize_image(T2_slice, target_size)
        
        # 标准化到[-1,1]范围
        C0_normalized = self.normalize(C0_resized, 'Zero_Mean_Unit_Std')
        LGE_normalized = self.normalize(LGE_resized, 'Zero_Mean_Unit_Std')
        T2_normalized = self.normalize(T2_resized, 'Zero_Mean_Unit_Std')
        
        # 转换为tensor并添加维度
        C0_input = torch.from_numpy(C0_normalized).unsqueeze(0).unsqueeze(0).float().to(self.device)
        LGE_input = torch.from_numpy(LGE_normalized).unsqueeze(0).unsqueeze(0).float().to(self.device)
        T2_input = torch.from_numpy(T2_normalized).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # 模型推理
        with torch.no_grad():
            seg_C0, seg_LGE, seg_T2 = self.model(C0_input, LGE_input, T2_input)
            
        # 获取分割结果
        pred_C0 = seg_C0.argmax(dim=1).squeeze().cpu().numpy()
        pred_LGE = seg_LGE.argmax(dim=1).squeeze().cpu().numpy()
        pred_T2 = seg_T2.argmax(dim=1).squeeze().cpu().numpy()
        
        # 将预测结果调整回原始尺寸
        pred_C0 = self.resize_back(pred_C0, original_size)
        pred_LGE = self.resize_back(pred_LGE, original_size)
        pred_T2 = self.resize_back(pred_T2, original_size)
        
        return pred_C0, pred_LGE, pred_T2

    def map_to_original_labels(self, pred_C0, pred_LGE, pred_T2):
        """将预测结果映射回原始标签值"""
        final_label = np.zeros_like(pred_C0, dtype=np.uint16)
        
        # 按优先级顺序映射（从低到高）
        # Cardiac
        for label, value in self.label_mapping['cardiac'].items():
            final_label[pred_C0 == label] = value
            
        # Scar
        for label, value in self.label_mapping['scar'].items():
            final_label[pred_LGE == label] = value
            
        # Edema (highest priority)
        for label, value in self.label_mapping['edema'].items():
            final_label[pred_T2 == label] = value
            
        return final_label

    def save_results(self, result_dict, save_dir, case_name, affine):
        """保存结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存分模态结果
        for modality, data in result_dict['modalities'].items():
            save_path = os.path.join(save_dir, f'{case_name}_{modality}_seg.nii.gz')
            nib.save(nib.Nifti1Image(data.transpose(1, 2, 0), affine), save_path)
        
        # 保存映射后的结果
        save_path = os.path.join(save_dir, f'{case_name}_mapped.nii.gz')
        nib.save(nib.Nifti1Image(result_dict['mapped'].transpose(1, 2, 0), affine), save_path)

    def run_inference(self, test_data_root, output_dir):
        """运行推理"""
        # 创建数据加载器
        test_dataset = MultiModalCardiacDataset(
            data_root=test_data_root,
            mode='test'
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 遍历所有测试数据
        for batch in tqdm(test_loader, desc="Processing cases"):
            C0 = batch['C0'].to(self.device)
            LGE = batch['LGE'].to(self.device)
            T2 = batch['T2'].to(self.device)
            case_name = batch['case_name'][0]
            
            # 获取图像尺寸
            D, H, W = C0.shape[2:]  # [1, 1, D, H, W]
            
            # 创建结果数组
            cardiac_pred = np.zeros((D, H, W), dtype=np.uint8)
            scar_pred = np.zeros((D, H, W), dtype=np.uint8)
            edema_pred = np.zeros((D, H, W), dtype=np.uint8)
            
            # 逐层处理
            with torch.no_grad():
                for d in range(D):
                    # 获取当前层
                    C0_slice = C0[..., d:d+1]  # 保持4D格式 [1, 1, 1, H, W]
                    LGE_slice = LGE[..., d:d+1]
                    T2_slice = T2[..., d:d+1]
                    
                    # 模型推理
                    seg_cardiac, seg_scar, seg_edema = self.model(C0_slice, LGE_slice, T2_slice)
                    
                    # 获取预测结果
                    cardiac_pred[d] = seg_cardiac.argmax(dim=1).squeeze().cpu().numpy()
                    scar_pred[d] = seg_scar.argmax(dim=1).squeeze().cpu().numpy()
                    edema_pred[d] = seg_edema.argmax(dim=1).squeeze().cpu().numpy()
            
            # 后处理：将三种模式的预测结果合并为最终格式
            final_pred = process_3d_predictions(cardiac_pred, scar_pred, edema_pred)
            
            # 保存结果
            affine = np.eye(4)  # 如果需要，可以从原始数据中获取正确的affine矩阵
            output_path = os.path.join(output_dir, f'{case_name}.nii.gz')
            nib.save(nib.Nifti1Image(final_pred, affine), output_path)
            
            print(f"Processed {case_name}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test cases')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    args = parser.parse_args()
    
    # 运行推理
    inferencer = CardiacInference(args.model_path, args)
    inferencer.run_inference(args.test_dir, args.output_dir)

if __name__ == '__main__':
    main() 