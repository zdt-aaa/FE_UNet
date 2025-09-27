import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainers import trainer_spines
from dataset_spine import SpineDataset

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/zdt/MCF-main/dataset/Spine_modified', help='root dir for data')
parser.add_argument('--dataset', type=str, default='Spines', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Spine', help='list dir')
parser.add_argument('--num_classes', type=int, default=16, help='output channel of network')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='比例值，用于计算预热步数（总训练步数的百分比）')
parser.add_argument('--output_dir', type=str,
                    default='./results/+OptimizedDecoderBlock', help='output dir')

args = parser.parse_args()

if __name__ == "__main__":
    # 确定性训练设置
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 数据集配置
    dataset_name = args.dataset
    dataset_config = {
        'Spines': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Spine',
            'num_classes': 16,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True  # 关闭预训练标志




    # 初始化ViT模型
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    # if not hasattr(config_vit, 'skip_channels'):
    #     config_vit.skip_channels = [512, 256, 64, 16]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.in_channels = 1  # 输入通道数设为1（单通道）

    # 根据ViT类型调整Patch Grid
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size)
        )

    # 创建模型（不加载预训练权重）
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    import pdb


    # 添加参数统计代码（兼容DataParallel）
    def count_parameters(model):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module  # 获取原始模型
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    total_params = count_parameters(net)
    print(f"模型总可训练参数量: {total_params / 1e6:.2f}M")  # 以百万为单位显示
    pdb.set_trace()

    # 创建数据集
    train_dataset = SpineDataset(
        base_dir=args.root_path,
        split="train",
        list_dir=args.list_dir,
        resize_size=(args.img_size, args.img_size)
    )

    # 启动训练
    trainer = {'Spines': trainer_spines}
    trainer[dataset_name](args, net, args.output_dir)