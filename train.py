import os
import time

import torch
from itertools import cycle
import torch.optim as optim
from criterion.loss import MyoPSLoss
from utils.tools import weights_init
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from network.model import MyoPSNet
# from validation import Validation2d
from utils.dataloader import CrossModalDataLoader, MultiModalCardiacDataset
import torch
import torchvision.utils as vutils
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"]='0'


from utils.config import config


import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def mask_to_rgb(mask):
    """
    mask: [C, H, W]，C = num_classes（通常为3）
    返回: [3, H, W]，RGB格式
    """
    mask = mask.detach().cpu()
    mask = torch.argmax(mask, dim=0).numpy()  # [H, W]
    H, W = mask.shape
    rgb = np.zeros((3, H, W), dtype=np.uint8)

    # 类别颜色映射
    colors = {
        0: (0, 0, 0),         # 背景 - 黑色
        1: (255, 0, 0),       # 类别1 - 红色
        2: (0, 255, 0),       # 类别2 - 绿色
        3: (0, 0, 255),       # 类别3 - 蓝色（如有第四类）
    }

    for cls_id, color in colors.items():
        rgb[0][mask == cls_id] = color[0]
        rgb[1][mask == cls_id] = color[1]
        rgb[2][mask == cls_id] = color[2]

    return rgb / 255.0  # 转为 0~1 float


def tensor_to_rgb(tensor):
    # 确保张量在 CPU 上且是 numpy 格式
    tensor = tensor.detach().cpu()
    
    # 检查输入张量的维度
    if tensor.dim() == 2:  # 如果是 2D 张量 (H, W)
        # 扩展为 (H, W, 1)
        return tensor.unsqueeze(-1).numpy()
    elif tensor.dim() == 3:  # 如果是 3D 张量 (C, H, W)
        # 调整维度顺序为 (H, W, C)
        return tensor.permute(1, 2, 0).numpy()
    elif tensor.dim() == 4:  # 如果是 4D 张量 (B, C, H, W)
        # 去除 batch 维度并调整顺序
        return tensor.squeeze(0).permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Unexpected tensor dimensions: {tensor.dim()}")

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # pred: [B, C, H, W]
        # target: [B, H, W]
        pred = torch.softmax(pred, dim=1)
        target = nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None):
        super(MultiTaskLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.weights = weights if weights is not None else {'cardiac': 1.0, 'scar': 1.0, 'edema': 1.0}
        
    def forward(self, predictions, targets):
        losses = {}
        total_loss = 0
        
        for task in ['cardiac', 'scar', 'edema']:
            pred = predictions[task]
            target = targets[f'{task}_label']
            
            # 计算Dice Loss和CrossEntropy Loss
            dice = self.dice_loss(pred, target)
            ce = self.ce_loss(pred, target)
            
            # 组合损失
            task_loss = dice + ce
            losses[f'{task}_loss'] = task_loss
            total_loss += self.weights[task] * task_loss
            
        losses['total'] = total_loss
        return losses

def train_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    task_losses = {'cardiac': 0, 'scar': 0, 'edema': 0}
    
    for batch_idx, batch in enumerate(loader):
        # 获取数据
        C0 = batch['C0'].to(device)
        LGE = batch['LGE'].to(device)
        T2 = batch['T2'].to(device)
        
        cardiac_label = batch['cardiac_label'].to(device)
        scar_label = batch['scar_label'].to(device)
        edema_label = batch['edema_label'].to(device)
        
        # 前向传播
        cardiac_pred, scar_pred, edema_pred = model(C0, LGE, T2)
        
        # 计算损失
        losses = criterion(
            {'cardiac': cardiac_pred, 'scar': scar_pred, 'edema': edema_pred},
            {'cardiac_label': cardiac_label, 'scar_label': scar_label, 'edema_label': edema_label}
        )
        
        # 反向传播
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # 更新统计信息
        total_loss += losses['total'].item()
        task_losses['cardiac'] += losses['cardiac_loss'].item()
        task_losses['scar'] += losses['scar_loss'].item()
        task_losses['edema'] += losses['edema_loss'].item()
        
        # 记录训练信息
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(loader)}, '
                  f'Loss: {losses["total"].item():.4f}, '
                  f'Cardiac: {losses["cardiac_loss"].item():.4f}, '
                  f'Scar: {losses["scar_loss"].item():.4f}, '
                  f'Edema: {losses["edema_loss"].item():.4f}')
            
            # 记录到TensorBoard
            step = epoch * len(loader) + batch_idx
            writer.add_scalar('Train/Total_Loss', losses['total'].item(), step)
            writer.add_scalar('Train/Cardiac_Loss', losses['cardiac_loss'].item(), step)
            writer.add_scalar('Train/Scar_Loss', losses['scar_loss'].item(), step)
            writer.add_scalar('Train/Edema_Loss', losses['edema_loss'].item(), step)
    
    # 计算平均损失
    avg_loss = total_loss / len(loader)
    avg_task_losses = {k: v / len(loader) for k, v in task_losses.items()}
    
    return avg_loss, avg_task_losses

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    task_losses = {'cardiac': 0, 'scar': 0, 'edema': 0}
    
    with torch.no_grad():
        for batch in loader:
            C0 = batch['C0'].to(device)
            LGE = batch['LGE'].to(device)
            T2 = batch['T2'].to(device)
            
            cardiac_label = batch['cardiac_label'].to(device)
            scar_label = batch['scar_label'].to(device)
            edema_label = batch['edema_label'].to(device)
            
            cardiac_pred, scar_pred, edema_pred = model(C0, LGE, T2)
            
            losses = criterion(
                {'cardiac': cardiac_pred, 'scar': scar_pred, 'edema': edema_pred},
                {'cardiac_label': cardiac_label, 'scar_label': scar_label, 'edema_label': edema_label}
            )
            
            total_loss += losses['total'].item()
            task_losses['cardiac'] += losses['cardiac_loss'].item()
            task_losses['scar'] += losses['scar_loss'].item()
            task_losses['edema'] += losses['edema_loss'].item()
    
    avg_loss = total_loss / len(loader)
    avg_task_losses = {k: v / len(loader) for k, v in task_losses.items()}
    
    return avg_loss, avg_task_losses

def TrainProcess(args):

    model = MyoPSNet(in_chs=(3, 2, 2), out_chs=(3, 3, 3)).cuda()
    model.apply(weights_init)

    mlsc_loss = MyoPSLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 初始化训练状态
    start_epoch = 0
    best_loss = float('inf')
    global_step = 0

    # 断点续训逻辑
    if args.resume and os.path.exists(args.checkpoint_path):
        print(f"=> Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器和学习率调度器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练状态
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_loss = checkpoint['best_loss']
        
        print(f"=> Resumed from epoch {start_epoch}, best loss: {best_loss:.6f}")
    else:
        print("=> Starting new training run")

    # Train_Image = CrossModalDataLoader(path=args.path, file_name='train.txt',
    #                                    dim=args.dim, max_iters=100 * args.batch_size,
    #                                    stage='Train', modalities=args.modalities)

    Train_Image = MultiModalCardiacDataset(data_root=args.path, mode='train', transform=None)
    
    Train_loader = cycle(DataLoader(Train_Image, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True))
    writer = SummaryWriter(log_dir=os.path.join('runs', time.strftime("%Y%m%d-%H%M%S")))
    
    IterCount = int(len(Train_Image) / args.batch_size)

    # 记录模型结构
    dummy_input_C0 = torch.randn(1, 1, args.dim, args.dim).cuda()
    dummy_input_LGE = torch.randn(1, 1, args.dim, args.dim).cuda()
    dummy_input_T2 = torch.randn(1, 1, args.dim, args.dim).cuda()
    writer.add_graph(model, (dummy_input_C0, dummy_input_LGE, dummy_input_T2))

    # 记录训练配置
    writer.add_text('Training Config', f'Batch Size: {args.batch_size}\n'
                                     f'Learning Rate: {args.lr}\n'
                                     f'Input Dimension: {args.dim}\n'
                                     f'Modalities: {args.modalities}', 0)

    for epoch in range(start_epoch, args.end_epoch):
        model.train()
        running_loss = 0.0
        running_loss_seg = 0.0
        running_loss_inv = 0.0
        running_loss_inc = 0.0
        
        print(f"Epoch [{epoch+1}/{args.end_epoch}]")

        for iteration in range(IterCount):
            # Load data
            img_C0, img_LGE, img_T2, label_cardiac, label_scar, label_edema, _ = next(Train_loader)

            img_C0 = img_C0.cuda(non_blocking=True)
            img_LGE = img_LGE.cuda(non_blocking=True)
            img_T2 = img_T2.cuda(non_blocking=True)
            label_cardiac = label_cardiac.cuda(non_blocking=True)
            label_scar = label_scar.cuda(non_blocking=True)
            label_edema = label_edema.cuda(non_blocking=True)

            # Forward
            seg_C0, seg_LGE, seg_T2 = model(img_C0, img_LGE, img_T2)
            seg = {'C0': seg_C0, 'LGE': seg_LGE, 'T2': seg_T2}
            label = {'cardiac': label_cardiac, 'scar': label_scar, 'edema': label_edema}

            loss_seg, loss_invariant, loss_inclusive, loss = mlsc_loss(seg, label)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新统计信息
            running_loss += loss.item()
            running_loss_seg += loss_seg.item()
            running_loss_inv += loss_invariant.item()
            running_loss_inc += loss_inclusive.item()
            global_step = epoch * IterCount + iteration

            # 每N次迭代记录一次详细信息
            if iteration % 10 == 0:
                # 计算当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                
                # 记录损失
                writer.add_scalar('Iteration/Loss_Total', loss.item(), global_step)
                writer.add_scalar('Iteration/Loss_Seg', loss_seg.item(), global_step)
                writer.add_scalar('Iteration/Loss_Invariant', loss_invariant.item(), global_step)
                writer.add_scalar('Iteration/Loss_Inclusive', loss_inclusive.item(), global_step)
                writer.add_scalar('Iteration/Learning_Rate', current_lr, global_step)

                # 记录梯度范数
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar('Iteration/Gradient_Norm', total_norm, global_step)

                # 记录预测结果可视化
                if iteration % 50 == 0:  # 每50次迭代记录一次图像
                    # 将分割结果转换为RGB图像
                    C0_rgb = mask_to_rgb(seg_C0[0])
                    LGE_rgb = mask_to_rgb(seg_LGE[0])
                    T2_rgb = mask_to_rgb(seg_T2[0])
                    
                    # 记录原始图像 (单通道图像需要重复三次来显示为灰度图)
                    writer.add_image('Images/C0_Input', img_C0[0].repeat(3, 1, 1), global_step)
                    writer.add_image('Images/LGE_Input', img_LGE[0].repeat(3, 1, 1), global_step)
                    writer.add_image('Images/T2_Input', img_T2[0].repeat(3, 1, 1), global_step)
                    
                    # 记录分割结果
                    writer.add_image('Segmentation/C0', C0_rgb, global_step)
                    writer.add_image('Segmentation/LGE', LGE_rgb, global_step)
                    writer.add_image('Segmentation/T2', T2_rgb, global_step)

                # 打印训练信息
                print(f"Iteration: {iteration+1}/{IterCount} - "
                      f"LR: {current_lr:.6f} | "
                      f"Loss: seg={loss_seg.item():.4f}, "
                      f"inv={loss_invariant.item():.4f}, "
                      f"inc={loss_inclusive.item():.4f}, "
                      f"total={loss.item():.4f}")

        # 更新学习率
        lr_scheduler.step()

        # 计算epoch平均损失
        avg_loss = running_loss / IterCount
        avg_loss_seg = running_loss_seg / IterCount
        avg_loss_inv = running_loss_inv / IterCount
        avg_loss_inc = running_loss_inc / IterCount

        # 记录epoch级别的统计信息
        writer.add_scalar('Epoch/Loss_Total', avg_loss, epoch)
        writer.add_scalar('Epoch/Loss_Seg', avg_loss_seg, epoch)
        writer.add_scalar('Epoch/Loss_Invariant', avg_loss_inv, epoch)
        writer.add_scalar('Epoch/Loss_Inclusive', avg_loss_inc, epoch)
        writer.add_scalar('Epoch/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # 记录模型参数分布
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param.data.cpu().numpy(), epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad.data.cpu().numpy(), epoch)

        # 保存检查点逻辑
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_loss': best_loss,
            'avg_loss': avg_loss,
            'avg_loss_seg': avg_loss_seg,
            'avg_loss_inv': avg_loss_inv,
            'avg_loss_inc': avg_loss_inc
        }

        # 按频率保存
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint at {save_path}")

        # 保存最佳模型
        if args.save_best and avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved best model with loss: {best_loss:.6f}")

        # 保存最新模型
        if args.save_last:
            save_path = os.path.join(args.save_dir, 'last_model.pth')
            torch.save(checkpoint, save_path)

        # 记录到日志文件
        with open('log_training.txt', 'a') as log:
            log.write(f"Epoch [{epoch+1}/{args.end_epoch}] - "
                     f"Avg Loss: {avg_loss:.6f}, "
                     f"Seg Loss: {avg_loss_seg:.6f}, "
                     f"Inv Loss: {avg_loss_inv:.6f}, "
                     f"Inc Loss: {avg_loss_inc:.6f}, "
                     f"Best Loss: {best_loss:.6f}, "
                     f"LR: {float(optimizer.param_groups[0]['lr']):.6f}\n")

    writer.close()
    print("Training completed!")









def MyoPSNetTrain(args):
    model = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3)).cuda()
    # model = MyoPSNet(modalities=args.modalities, out_chs=(3, 3, 3, 3)).cuda()
    model.apply(weights_init)

    criterion = MyoPSLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    os.makedirs('checkpoints', exist_ok=True)
    writer = SummaryWriter()

    train_dataset = CrossModalDataLoader(path=args.path, file_name='train.txt',
                                         dim=args.dim, max_iters=100 * args.batch_size,
                                         stage='Train', modalities=args.modalities)

    train_loader = cycle(DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True))
    iter_per_epoch = len(train_dataset) // args.batch_size

    for epoch in range(args.start_epoch, args.end_epoch):
        model.train()

        for iteration in range(iter_per_epoch):
            data = next(train_loader)
            img_C0, img_LGE, img_T2, img_T1m, img_T2starm, label_cardiac, label_scar, label_edema, _ = data

            img_C0 = img_C0.cuda(non_blocking=True)
            img_LGE = img_LGE.cuda(non_blocking=True)
            img_T2 = img_T2.cuda(non_blocking=True)
            img_T1m = img_T1m.cuda(non_blocking=True)
            img_T2starm = img_T2starm.cuda(non_blocking=True)
            label_cardiac = label_cardiac.cuda(non_blocking=True)
            label_scar = label_scar.cuda(non_blocking=True)
            label_edema = label_edema.cuda(non_blocking=True)

            outputs = model(img_C0, img_LGE, img_T2, img_T1m, img_T2starm)
            seg = {'C0': outputs[0], 'LGE': outputs[1], 'T2': outputs[2], 'mapping': outputs[3]}
            label = {'cardiac': label_cardiac, 'scar': label_scar, 'edema': label_edema}

            loss_seg, loss_invariant, loss_inclusive, total_loss = criterion(seg, label)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            step = epoch * iter_per_epoch + iteration
            writer.add_scalar('Loss/Segmentation', loss_seg.item(), step)
            writer.add_scalar('Loss/Invariant', loss_invariant.item(), step)
            writer.add_scalar('Loss/Inclusive', loss_inclusive.item(), step)
            writer.add_scalar('Loss/Total', total_loss.item(), step)

            with open('log_training.txt', 'a') as log:
                log.write(f"==> Epoch: {epoch+1:03}/{args.end_epoch:03} || Iter: {iteration+1:03}/{iter_per_epoch:03} - ")
                log.write(f"LR: {optimizer.param_groups[0]['lr']:.6f} | ")
                log.write(f"loss_seg: {loss_seg.item():.6f} + ")
                log.write(f"loss_inv: {loss_invariant.item():.6f} + ")
                log.write(f"loss_inc: {loss_inclusive.item():.6f} = ")
                log.write(f"total_loss: {total_loss.item():.6f}\n")

        scheduler.step()

        # 可选验证保存
        # avg_dice = Validation2d(args, epoch, model, Valid_Image, Valid_loader, writer, 'result_validation_2d.txt', tensorboardImage=True)
        # if avg_dice > args.threshold:
        #     torch.save(model.state_dict(), os.path.join('checkpoints', f'{avg_dice:.4f}[{epoch+1}].pth'))




if __name__ == '__main__':
    args = config()
    print(args)
    TrainProcess(args)



