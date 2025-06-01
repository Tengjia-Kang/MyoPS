# import os
# import torch
# from itertools import cycle
# import torch.optim as optim
# from criterion.loss import MyoPSLoss
# from utils.tools import weights_init
# from tensorboardX import SummaryWriter
# from torch.utils.data import DataLoader
# from network.model import MyoPSNet
# from validation import Validation2d
# from utils.dataloader import CrossModalDataLoader
#
#
# def MyoPSNetTrain(args):
#
#     # C0(5,3) LGE(2,3) T2(2,3) mapping(3,3)
#     model = MyoPSNet(in_chs=(5,2,2,3), out_chs=(3,3,3,3)).cuda()
#     model.apply(weights_init)
#
#     mlsc_loss = MyoPSLoss().cuda()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
#     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6, last_epoch=-1)
#
#     if not os.path.exists('checkpoints'):
#         os.makedirs('checkpoints')
#
#     Train_Image = CrossModalDataLoader(path=args.path, file_name='train.txt',
#                                        dim=args.dim, max_iters=100 * args.batch_size,
#                                        stage='Train', modalities=args.modalities)
#
#     Train_loader = cycle(DataLoader(Train_Image, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True))
#
#     # Valid_Image = CrossModalDataLoader(path=args.path, file_name='validation.txt', dim=args.dim, max_iters=None, stage='Valid')
#     # Valid_loader = cycle(DataLoader(Valid_Image, batch_size=1, shuffle=False, num_workers=0, drop_last=False))
#
#     writer = SummaryWriter()
#
#     for epoch in range(args.start_epoch, args.end_epoch):
#
#         # Train
#         model.train()
#
#         train_C0 = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
#         train_LGE = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
#         train_T2 = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
#         train_T1m = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
#         train_T2starm = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
#
#         cardiac_gd = torch.FloatTensor(args.batch_size, 3, args.dim, args.dim).cuda()
#         scar_gd = torch.FloatTensor(args.batch_size, 3, args.dim, args.dim).cuda()
#         edema_gd = torch.FloatTensor(args.batch_size, 3, args.dim, args.dim).cuda()
#
#         IterCount = int(len(Train_Image)/args.batch_size)
#
#         for iteration in range(IterCount):
#
#             # Sup
#             img_C0, img_LGE, img_T2, img_T1m, img_T2starm, label_cardiac, label_scar, label_edema, _ = next(Train_loader)
#
#             train_C0.copy_(img_C0)
#             train_LGE.copy_(img_LGE)
#             train_T2.copy_(img_T2)
#             train_T1m.copy_(img_T1m)
#             train_T2starm.copy_(img_T2starm)
#
#             cardiac_gd.copy_(label_cardiac)
#             scar_gd.copy_(label_scar)
#             edema_gd.copy_(label_edema)
#
#             seg_C0, seg_LGE, seg_T2, seg_mapping = model(train_C0, train_LGE, train_T2, train_T1m, train_T2starm)
#
#             seg = {'C0': seg_C0, 'LGE': seg_LGE, 'T2': seg_T2, 'mapping': seg_mapping}
#             label = {'cardiac': cardiac_gd, 'scar': scar_gd, 'edema': edema_gd}
#
#             loss_seg, loss_invariant, loss_inclusive, loss = mlsc_loss(seg, label)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # write to log
#             with open('log_training.txt', 'a') as segment_log:
#                 segment_log.write("==> Epoch: {:0>3d}/{:0>3d} || ".format(epoch + 1, args.end_epoch))
#                 segment_log.write("Iteration: {:0>3d}/{:0>3d} - ".format(iteration + 1, IterCount))
#                 segment_log.write("LR: {:.6f} | ".format(float(optimizer.param_groups[0]['lr'])))
#                 segment_log.write("loss_seg: {:.6f} + ".format(loss_seg.detach().cpu()))
#                 segment_log.write("loss_invariant: {:.6f} + ".format(loss_invariant.detach().cpu()))
#                 segment_log.write("loss_inclusive: {:.6f} + ".format(loss_inclusive.detach().cpu()))
#                 segment_log.write("loss: {:.6f}\n".format(loss.detach().cpu()))
#
#             # write to tensorboard
#             writer.add_scalar('seg loss', loss_seg.detach().cpu(), epoch * (IterCount + 1) + iteration)
#             writer.add_scalar('invariant loss', loss_invariant.detach().cpu(), epoch * (IterCount + 1) + iteration)
#             writer.add_scalar('inclusive loss', loss_inclusive.detach().cpu(), epoch * (IterCount + 1) + iteration)
#             writer.add_scalar('total loss', loss.detach().cpu(), epoch * (IterCount + 1) + iteration)
#
#         lr_scheduler.step()
#
#         # Validation
#         # model.eval()
#         # avg_dice_2d = Validation2d(args, epoch, model, Valid_Image, Valid_loader, writer, 'result_validation_2d.txt', tensorboardImage=True)
#
#         # if avg_dice_2d > args.threshold:
#         #     torch.save(model.state_dict(), os.path.join('checkpoints', str(avg_dice_2d) + '['+ str(epoch+1) + '].pth'))


import os
import torch
from itertools import cycle
import torch.optim as optim
from criterion.loss import MyoPSLoss
from utils.tools import weights_init
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from network.model import MyoPSNet
# from validation import Validation2d  # 可选启用
from utils.dataloader import CrossModalDataLoader


def MyoPSNetTrain(args):
    # model = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3)).cuda()
    model = MyoPSNet(modalities=args.modalities, out_chs=(3, 3, 3, 3)).cuda()
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

