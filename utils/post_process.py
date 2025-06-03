import numpy as np

def combine_predictions(cardiac_pred, scar_pred, edema_pred):
    """
    将三种模式的预测结果合并为最终的标签格式
    
    Args:
        cardiac_pred: shape (H, W), 值为 0(背景), 1(心肌), 2(心腔)
        scar_pred: shape (H, W), 值为 0(背景), 1(心肌), 2(疤痕)
        edema_pred: shape (H, W), 值为 0(背景), 1(心肌水肿), 2(水肿)
    
    Returns:
        combined: shape (H, W), 值为最终标签格式 (200, 500, 600, 1220, 2221)
    """
    H, W = cardiac_pred.shape
    combined = np.zeros((H, W), dtype=np.uint16)
    
    # 1. 首先处理cardiac的预测结果
    # 心肌 (200)
    combined[cardiac_pred == 1] = 200
    # 心腔 (500)
    combined[cardiac_pred == 2] = 500
    
    # 2. 处理scar的预测结果
    # 疤痕区域 (600) - 在心肌区域内的疤痕
    scar_region = (scar_pred == 2) & (combined == 200)
    combined[scar_region] = 600
    
    # 3. 处理edema的预测结果
    # 心肌水肿 (1220) - 在心肌区域内的水肿
    myo_edema = (edema_pred == 1) & (combined == 200)
    combined[myo_edema] = 1220
    
    # 远端水肿 (2221) - 不在心肌区域内的水肿
    remote_edema = (edema_pred == 2) & (combined == 0)
    combined[remote_edema] = 2221
    
    return combined

def process_3d_predictions(cardiac_pred_3d, scar_pred_3d, edema_pred_3d):
    """
    处理3D预测结果
    
    Args:
        cardiac_pred_3d: shape (D, H, W)
        scar_pred_3d: shape (D, H, W)
        edema_pred_3d: shape (D, H, W)
    
    Returns:
        combined_3d: shape (D, H, W)
    """
    D, H, W = cardiac_pred_3d.shape
    combined_3d = np.zeros((D, H, W), dtype=np.uint16)
    
    for d in range(D):
        combined_3d[d] = combine_predictions(
            cardiac_pred_3d[d],
            scar_pred_3d[d],
            edema_pred_3d[d]
        )
    
    return combined_3d 