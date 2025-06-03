import argparse


def config(name='SMR_Segment'):

    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('--modalities', nargs='+', default=['C0', 'LGE','T2'],
                        help='Modalities to use, e.g., C0 LGE T2  T1m T2starm')

    parser.add_argument('--path', type=str, default='./datasets/Competition_Dataset/Train', help="data path")

    # 推理 & 测试
    parser.add_argument('--load_path', type=str, default='checkpoints', help="load path")
    parser.add_argument('--predict_mode', type=str, default='multiple', help="predict mode: single or multiple")
    parser.add_argument('--test_path', type=str, default='datasets/Competition_Dataset/Train', help="test path")
    
    # 断点续训相关参数
    parser.add_argument('--resume', action='store_true', help="whether to resume training from checkpoint")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/last_model.pth', help="checkpoint path")
    parser.add_argument('--save_freq', type=int, default=5, help="save checkpoint frequency (epochs)")
    parser.add_argument('--save_best', action='store_true', default=True, help="whether to save best model")
    parser.add_argument('--save_last', action='store_true', default=True, help="whether to save last model")
    parser.add_argument('--save_dir', type=str, default='checkpoints', help="directory to save checkpoints")
    
    # 参数
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dim', type=int, default=192, help="dimension of 2D image")
    parser.add_argument('--feature_dim', type=int, default=1024, help="dimensions of feature")
    parser.add_argument('--lr', type=float, default=1e-4, help="starting learning rate")
    
    # 设置
    parser.add_argument('--threshold', type=float, default=0.50, help="the minimum dice to predict model")
    parser.add_argument('--start_epoch', type=int, default=0, help="flag to indicate the start epoch")
    parser.add_argument('--end_epoch', type=int, default=200, help="flag to indicate the final epoch")

    args = parser.parse_args()
    return args
    