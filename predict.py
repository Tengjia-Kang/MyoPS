import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore")
import tqdm
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from utils.config import config
from network.model import MyoPSNet
from process import LargestConnectedComponents
from utils.tools import Normalization, ImageTransform, ResultTransform



def predict(args, model_path, epoch, slice_index):

    # Load model
    model = MyoPSNet(in_chs=(3, 2, 2), out_chs=(3, 3, 3))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()


    # Output folder
    save_dir = f'test/test_{epoch}'
    os.makedirs(save_dir, exist_ok=True)

    normalize = Normalization()
    keepLCC = LargestConnectedComponents()
    image_transform = ImageTransform(args.dim, 'Test')

    result_transform = ResultTransform(ToOriginal=True)

    # 遍历测试文件夹
    test_root = args.test_path
    # test_root = os.path.join(args.path, 'Competition_Dataset', 'Test')
    cases = sorted(os.listdir(test_root))

    for case in cases:
        case_path = os.path.join(test_root, case)
        C0_path = os.path.join(case_path, f'{case}_C0.nii.gz')
        LGE_path = os.path.join(case_path, f'{case}_LGE.nii.gz')
        T2_path = os.path.join(case_path, f'{case}_T2.nii.gz')


        modalities = ['C0', 'LGE', 'T2']
        file_path = {
            'C0' : C0_path,
            'LGE' : LGE_path,
            'T2' : T2_path,
            'index' : slice_index
        }
        img_slices = []

        for modality in modalities:
            raw_data = nib.load(file_path[modality]).get_fdata()

            slice_data = raw_data[:, :, file_path['index']:file_path['index'] + 1].astype(np.float32)
            img_slices.append(slice_data)

        image = np.concatenate(img_slices, axis=2)
        transformed = image_transform(image)
        imgs = torch.chunk(transformed, chunks=len(modalities) + 1, dim=0)
        # 数据增强后的不同模态的图
        img_tensors = [normalize(img, 'Zero_Mean_Unit_Std') for img in imgs[:]]

        img_C0, img_LGE, img_T2 = img_tensors[0].unsqueeze(0), img_tensors[1].unsqueeze(0), img_tensors[2].unsqueeze(0)

        seg_C0, seg_LGE, seg_T2 = model(img_C0, img_LGE, img_T2)
        seg = {'C0': seg_C0, 'LGE': seg_LGE, 'T2': seg_T2}

        # 后处理：保留最大连通区域
        # seg_LGE = keepLCC(seg_LGE.squeeze(), 'scar')  # Scar模式：背景 -> 0，疤痕 -> 1
        # seg_T2 = keepLCC(seg_T2.squeeze(), 'edema')  # Edema模式：背景 -> 0，水肿 -> 1
        # seg_C0 = keepLCC(seg_C0.squeeze(), 'cardiac')  # Cardiac模式：背景 -> 0，心肌/LV -> 1/2

        seg_C0 = seg['C0'].detach().cpu()
        seg_C0 = torch.argmax(seg_C0, dim=0).numpy()  # [H, W]

        # pred_C0 = seg_C0.argmax(dim=1).cpu().numpy()
        # pred_T2 = seg_T2.argmax(dim=1).cpu().numpy()
        # pred_LGE = seg_LGE.argmax(dim=1).cpu().numpy()
        # 映射预测结果到原始标签值
        final_label = np.zeros_like(pred_C0, dtype=np.uint8)

        # 处理心肌和LV（seg_C0：1->200，2->500）
        final_label[pred_C0 == 1] = 200  # Myocardium
        final_label[pred_C0 == 2] = 500  # LV

        # 处理疤痕（seg_LGE：1->600）
        final_label[pred_LGE == 1] = 600  # Scar

        # 处理心肌水肿和水肿（seg_T2：1->1220，2->2221）
        final_label[pred_T2 == 1] = 1220  # Myocardial Edema
        final_label[pred_T2 == 2] = 2221  # Edema

        # 反向映射（确保优先级，例如Edema覆盖其他）
        # 注意：根据实际需求调整覆盖顺序
        final_label[pred_T2 == 2] = 2221  # Edema优先级最高

        # 保存结果
        result_volume = final_label.transpose(2, 0, 1)  # 调整维度顺序
        seg_nii = nib.Nifti1Image(result_volume, nib.load(C0_path).affine)
        nib.save(seg_nii, os.path.join(save_dir, f'{case}_pred.nii.gz'))
        print(f"Saved {case} prediction")
        # 后处理
        # seg_LGE = keepLCC(seg_LGE, 'scar')
        # seg_T2 = keepLCC(seg_T2, 'edema')
        # seg_C0 = keepLCC(seg_C0, 'cardiac')
        # seg_pathology = result_transform(seg_LGE, seg_C0, seg_T2)

        # 反映射回原图大小
        # center_x, center_y = dim_x // 2, dim_y // 2
        # result[:, center_x - args.dim//2: center_x + args.dim//2,
        #           center_y - args.dim//2: center_y + args.dim//2].copy_(seg_pathology)

        # result = result.numpy().transpose(1, 2, 0)
        # seg_map = nib.Nifti1Image(result, img_affine)
        # save_path = os.path.join(save_dir, case + '_result.nii.gz')
        # nib.save(seg_map, save_path)
        #
        # print(f"{case} -> Done.")
#
#
#         # 保存结果
#         result_volume = final_label.transpose(2, 0, 1)  # (args.dim, args.dim, dim_z)
#         seg_nii = nib.Nifti1Image(result_volume, nib.load(C0_path).affine)
#         nib.save(seg_nii, os.path.join(save_dir, f'{case}_pred.nii.gz'))
#         print(f"Saved {case} prediction")



# def predict(args, model_path, epoch):
#     # Load model
#     model = MyoPSNet(in_chs=(3, 2, 2), out_chs=(3, 3, 3)).cuda()
#     model.load_state_dict(torch.load(model_path, map_location='cpu'))
#     model.eval()
#
#     # Output folder
#     save_dir = f'test/test_{epoch}'
#     os.makedirs(save_dir, exist_ok=True)
#
#     normalize = Normalization()
#     image_transform = ImageTransform(args.dim, 'Test')
#     result_transform = ResultTransform(ToOriginal=True)
#
#     # 遍历测试案例
#     test_root = args.test_path
#     cases = sorted(os.listdir(test_root))
#
#     for case in cases:
#         case_path = os.path.join(test_root, case)
#         C0_path = os.path.join(case_path, f'{case}_C0.nii.gz')
#         LGE_path = os.path.join(case_path, f'{case}_LGE.nii.gz')
#         T2_path = os.path.join(case_path, f'{case}_T2.nii.gz')
#
#         # 加载原始数据
#         C0_img = normalize(nib.load(C0_path).get_fdata(), 'Truncate').astype(np.float32)
#         LGE_img = normalize(nib.load(LGE_path).get_fdata(), 'Truncate').astype(np.float32)
#         T2_img = normalize(nib.load(T2_path).get_fdata(), 'Truncate').astype(np.float32)
#
#         # 获取原始尺寸
#         dim_x, dim_y, dim_z = C0_img.shape
#         result = np.zeros((dim_z, dim_x, dim_y), dtype=np.uint8)
#
#         # 预处理拼接（调整维度顺序）
#         oridata = np.concatenate([C0_img, LGE_img, T2_img], axis=2)  # (dim_x, dim_y, 3)
#         processed = image_transform(oridata)  # 应输出 (3, args.dim, args.dim)
#
#         # 分割模态
#         slices = torch.chunk(processed, chunks=3, dim=0)  # 每个模态形状 (1, args.dim, args.dim)
#         test_C0 = slices[0].unsqueeze(0).float().cuda()  # (1, 1, args.dim, args.dim)
#         test_LGE = slices[1].unsqueeze(0).float().cuda()
#         test_T2 = slices[2].unsqueeze(0).float().cuda()
#
#         with torch.no_grad():
#             seg_C0, seg_LGE, seg_T2 = model(test_C0, test_LGE, test_T2)
#
#         # 后处理
#         pred_C0 = seg_C0.argmax(dim=1).squeeze().cpu().numpy()  # (args.dim, args.dim)
#         pred_LGE = seg_LGE.argmax(dim=1).squeeze().cpu().numpy()
#         pred_T2 = seg_T2.argmax(dim=1).squeeze().cpu().numpy()
#
#         # 映射标签（示例逻辑，需根据实际调整）
#         final_label = np.zeros_like(pred_C0)
#         final_label[pred_LGE == 2] = 600  # Scar
#         final_label[pred_T2 == 2] = 2221  # Edema
#         final_label[pred_T2 == 1] = 1220  # Myocardial Edema
#         final_label[pred_C0 == 1] = 200   # Myocardium
#         final_label[pred_C0 == 2] = 500   # LV
#
#         # 保存结果
#         result_volume = final_label.transpose(2, 0, 1)  # (args.dim, args.dim, dim_z)
#         seg_nii = nib.Nifti1Image(result_volume, nib.load(C0_path).affine)
#         nib.save(seg_nii, os.path.join(save_dir, f'{case}_pred.nii.gz'))
#         print(f"Saved {case} prediction")

# 保留原有的predict_multiple函数




# def predict(args, model_path, epoch):
#
#     # model definition
#     model = MyoPSNet(in_chs=(3, 2, 2), out_chs=(3,3,3,3))
#     model.load_state_dict(torch.load(model_path, map_location='cpu'))
#     model.eval()
#
#     if not os.path.exists('test/test_' + str(epoch)):
#         os.makedirs('test/test_' + str(epoch))
#
#     test_img = pd.read_csv(args.path + 'Zhongshan/test.csv')
#
#     normalize = Normalization()
#     keepLCC = LargestConnectedComponents()
#     image_transform = ImageTransform(args.dim, 'Test')
#     result_transform = ResultTransform(ToOriginal=True)
#
#     for i in range(int(len(test_img))):
#
#         prefix_data = os.path.join(args.path + 'Zhongshan/' + test_img.iloc[i]["stage"], test_img.iloc[i]["file_name"])
#         dim_x, dim_y, dim_z = test_img.iloc[i]["dx"], test_img.iloc[i]["dy"], test_img.iloc[i]["dz"]
#         result = torch.zeros([dim_z, dim_x, dim_y])
#
#         # get data [x,y,z]
#         C0_raw = nib.load(prefix_data + '_C0.nii.gz')
#         LGE_raw = nib.load(prefix_data + '_LGE.nii.gz')
#         T2_raw = nib.load(prefix_data + '_T2.nii.gz')
#         T1m_raw = nib.load(prefix_data + '_T1m.nii.gz')
#         T2starm_raw = nib.load(prefix_data + '_T2starm.nii.gz')
#         img_affine = C0_raw.affine
#
#         C0_img = normalize(C0_raw.get_fdata(), 'Truncate').astype(np.float32)
#         LGE_img = normalize(LGE_raw.get_fdata(), 'Truncate').astype(np.float32)
#         T2_img = normalize(T2_raw.get_fdata(), 'Truncate').astype(np.float32)
#         T1m_img = normalize(T1m_raw.get_fdata(), 'Truncate').astype(np.float32)
#         T2starm_img = normalize(T2starm_raw.get_fdata(), 'Truncate').astype(np.float32)
#
#         oridata = np.concatenate([C0_img, LGE_img, T2_img, T1m_img, T2starm_img], axis=2)
#         img_C0, img_LGE, img_T2, img_T1m, img_T2starm = torch.chunk(image_transform(oridata), chunks=5, dim=0)
#
#         test_C0 = torch.FloatTensor(1, 1, args.dim, args.dim)
#         test_LGE = torch.FloatTensor(1, 1, args.dim, args.dim)
#         test_T2 = torch.FloatTensor(1, 1, args.dim, args.dim)
#         test_T1m = torch.FloatTensor(1, 1, args.dim, args.dim)
#         test_T2starm = torch.FloatTensor(1, 1, args.dim, args.dim)
#
#         seg_LGE = torch.FloatTensor(dim_z, args.dim, args.dim)
#         seg_T2 = torch.FloatTensor(dim_z, args.dim, args.dim)
#         seg_mapping = torch.FloatTensor(dim_z, args.dim, args.dim)
#
#         for j in range(dim_z):
#
#             img_C0_slice = normalize(img_C0[j:j+1,...], 'Zero_Mean_Unit_Std')
#             img_LGE_slice = normalize(img_LGE[j:j+1,...], 'Zero_Mean_Unit_Std')
#             img_T2_slice = normalize(img_T2[j:j+1,...], 'Zero_Mean_Unit_Std')
#             img_T1m_slice = normalize(img_T1m[j:j+1,...], 'Zero_Mean_Unit_Std')
#             img_T2starm_slice = normalize(img_T2starm[j:j+1,...], 'Zero_Mean_Unit_Std')
#
#             test_C0.copy_(img_C0_slice.unsqueeze(0))
#             test_LGE.copy_(img_LGE_slice.unsqueeze(0))
#             test_T2.copy_(img_T2_slice.unsqueeze(0))
#             test_T1m.copy_(img_T1m_slice.unsqueeze(0))
#             test_T2starm.copy_(img_T2starm_slice.unsqueeze(0))
#
#             _, res_LGE, res_T2, res_mapping = model(test_C0, test_LGE, test_T2, test_T1m, test_T2starm)
#
#             seg_LGE[j:j+1,:,:].copy_(torch.argmax(res_LGE, dim=1))
#             seg_T2[j:j+1,:,:].copy_(torch.argmax(res_T2, dim=1))
#             seg_mapping[j:j+1,:,:].copy_(torch.argmax(res_mapping, dim=1))
#
#         # post process
#         seg_LGE = keepLCC(seg_LGE, 'scar')
#         seg_T2 = keepLCC(seg_T2, 'edema')
#         seg_mapping = keepLCC(seg_mapping, 'scar')
#         seg_pathology = result_transform(seg_LGE, seg_mapping, seg_T2)
#
#         result[:, dim_x//2-args.dim//2:dim_x//2+args.dim//2, dim_y//2-args.dim//2:dim_y//2+args.dim//2].copy_(seg_pathology)
#         result = result.numpy().transpose(1,2,0)
#         seg_map = nib.Nifti1Image(result, img_affine)
#         nib.save(seg_map, 'test/test_' + str(epoch) + '/' + test_img.iloc[i]["file_name"] + '_result.nii.gz')
#         print(test_img.iloc[i]["file_name"] + "_Successfully saved!")


def predict_multiple(args):

    if not os.path.exists('test'):
        os.makedirs('test')

    for root, folder, files in os.walk(args.load_path):
        for file in files:
            model_path = os.path.join(args.load_path, file)
            dice = float(file.split('[')[0])
            epoch = int((file.split('[')[1]).split(']')[0])
            if args.predict_mode == 'single':
                if dice == args.threshold:
                    print('--- Start predicting epoch ' + str(epoch) + ' ---') 
                    predict(args, model_path, epoch)  
                    print('--- Test done for epoch ' + str(epoch) + ' ---')
            if args.predict_mode == 'multiple':
                if dice > args.threshold:
                    print('--- Start predicting epoch ' + str(epoch) + ' ---') 
                    predict(args, model_path, epoch)  
                    print('--- Test done for epoch ' + str(epoch) + ' ---')
    

if __name__ == '__main__':
    args = config()
    predict(args, 'checkpoints/best_model.pth', 2, slice_index=0)
    # predict_multiple(args)    predict(args, 'checkpoints/best_model.pth', 2, slice_index=0)

