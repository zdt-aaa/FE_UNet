import argparse
import logging
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dataset_spine import SpineDataset
from utilsV0 import DiceLoss
import albumentations as A

from torch.optim.lr_scheduler import LambdaLR
import math
from torchviz import make_dot
import pdb

from graphviz import Digraph
import torch.nn as nn

from graphviz import Digraph
import os


def create_simplified_architecture_diagram():
    # 创建有向图 - 移除正交边设置
    dot = Digraph('VisionTransformer', format='svg',
                  graph_attr={'rankdir': 'TB', 'dpi': '300'},
                  node_attr={'shape': 'box', 'style': 'filled', 'fillcolor': '#F0F8FF'})

    # 颜色定义
    input_color = '#FFE4B5'
    process_color = '#E0FFFF'
    trans_color = '#E6E6FA'
    decode_color = '#F0FFF0'
    output_color = '#FFF0F5'

    # 添加主要节点
    dot.node('input', 'Input Image\n(1×224×224)', shape='ellipse', fillcolor=input_color)

    # 频域处理模块
    with dot.subgraph(name='cluster_freq') as c:
        c.attr(label='Frequency Domain Processing', style='filled', fillcolor='#F5F5DC', color='#DAA520')
        c.node('edge_enhance', 'Frequency Edge Enhancement', fillcolor=process_color)
        dot.edge('input', 'edge_enhance')

    # Transformer编码器模块
    with dot.subgraph(name='cluster_transformer') as c:
        c.attr(label='Transformer Encoder', style='filled', fillcolor='#F0F8FF', color='#4682B4')
        c.node('encoder', '12 Transformer Blocks\nwith Positional Encoding', fillcolor=trans_color)
        dot.edge('edge_enhance', 'encoder')

    # 解码器模块
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='Decoder Cup', style='filled', fillcolor='#FFF5EE', color='#CD5C5C')
        c.node('decoder', '4 Enhanced Decoder Blocks\nwith Skip Connections', fillcolor=decode_color)
        dot.edge('encoder', 'decoder')

    # 输出模块
    dot.node('seg_head', 'Segmentation Head', fillcolor=output_color)
    dot.node('output', 'Output Mask\n(1×224×224)', shape='ellipse', fillcolor=input_color)

    # 添加连接
    dot.edge('decoder', 'seg_head')
    dot.edge('seg_head', 'output')

    # 添加跳过连接说明
    dot.node('skip_note', 'Skip connections from encoder\nto decoder blocks', shape='note', fillcolor='#F5F5F5')
    dot.edge('encoder', 'skip_note', style='dashed', color='gray', arrowhead='none')
    dot.edge('skip_note', 'decoder', style='dashed', color='gray', arrowhead='none')

    return dot

class WarmupCosineSchedule(LambdaLR):
    """ 带预热的余弦退火调度器 """
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return current_step / max(1, self.warmup_steps)
        progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

# 配置日志过滤器（防止第三方库的日志干扰）
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno in (logging.DEBUG, logging.INFO)


def trainer_spines(args, model, snapshot_path):
    # 初始化日志系统
    # 修正后的日志配置（移除了filename参数）
    log_file = os.path.join(snapshot_path, "log.txt")

    # 创建日志目录
    os.makedirs(snapshot_path, exist_ok=True)

    # 配置日志处理器
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]

    # 正确的配置方式
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers  # 仅通过handlers参数指定处理器
    )

    logger = logging.getLogger()
    logger.info(f"Training configuration:\n{str(args)}")
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.to(device)


    # 数据增强配置
    # train_transform = Compose([
    #     RandomRotate90(p=0.5),
    #     ElasticTransform(alpha=120, sigma=8, p=0.3),
    #     GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
    #     # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    #     HorizontalFlip(p=0.5),
    #     Rotate(limit=30, interpolation=1, border_mode=4, p=0.3)
    # ])#new删
    train_transform = A.Compose([
        A.Rotate(limit=15, p=0.5),  # 旋转±15度
        # A.Affine(shift_limit=0.05, scale_limit=0, rotate_limit=0, p=0.3),  # 平移±5%
        A.HorizontalFlip(p=0.5)  # 水平翻转（需确认解剖合理性）
    ])

    # 数据集加载
    train_set = SpineDataset(
        base_dir=args.root_path,
        split="train",
        # list_dir=args.list_dir,
        transform=train_transform,
        resize_size=(args.img_size, args.img_size)
    )
    val_set = SpineDataset(
        base_dir=args.root_path,
        split="val",
        # list_dir=args.list_dir,
        transform=None,
        resize_size=(args.img_size, args.img_size)
    )

    # 数据加载器
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed + worker_id)
        random.seed(worker_seed + worker_id)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size * args.n_gpu,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size * args.n_gpu,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 优化器配置
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=0.0001
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.1, patience=5, verbose=True
    # )#new删
    max_iterations = args.max_epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * max_iterations)#new
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=max_iterations
    )#new
    # 损失函数
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=args.num_classes)
    # focal_loss = FocalLoss()

    # 训练监控
    writer = SummaryWriter(os.path.join(snapshot_path, 'logs'))
    logger.info(f"TensorBoard logs directory: {os.path.join(snapshot_path, 'logs')}")

    # 训练状态变量
    best_dice = 0.0
    best_epoch = 0
    iter_num = 0
    # max_iterations = args.max_epochs * len(train_loader)
    # warmup_steps = int(args.warmup_ratio * max_iterations)#new
    logger.info(f"Training details: {len(train_loader)} iterations/epoch, {max_iterations} total iterations")

    # 训练循环
    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_dice = 0.0
        start_time = time.time()

        # 迭代进度条
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.max_epochs}", unit="batch")

        for batch in batch_iter:
            # 数据准备
            images = batch['image'].to(device)
            labels = batch['label'].to(device).long().squeeze(1)

            # 适配三通道输入
            # if images.size(1) == 1:
            #     images = images.repeat(1, 3, 1, 1)

            # 前向传播
            outputs = model(images)
            # 创建并保存简化架构图
            # dot = create_simplified_architecture_diagram()
            # dot.render('simplified_vision_transformer', directory='model_architecture', cleanup=True, format='svg')
            # print("简化架构图已保存到: model_architecture/simplified_vision_transformer.svg")

            #网络架构可视化
            # # 创建SVG图
            # dot = make_dot(outputs,
            #                params=dict(model.named_parameters()),
            #                show_attrs=True,
            #                show_saved=True)
            #
            # # 保存为SVG文件
            # dot.format = 'svg'
            # dot.render(filename='model_architecture/vision_transformer', directory='.', cleanup=True)
            #
            # print("SVG文件已保存到: model_architecture/vision_transformer.svg")

            # pdb.set_trace()

            # 计算损失
            loss_ce = ce_loss(outputs, labels.squeeze(1))
            loss_dice = dice_loss(outputs, labels, softmax=True)
            # loss_focal = focal_loss(outputs, labels)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # 记录训练指标
            iter_num += 1
            epoch_loss += loss.item()
            epoch_ce += loss_ce.item()
            epoch_dice += loss_dice.item()

            # 实时指标显示
            batch_iter.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': optimizer.param_groups[0]['lr']
            })

            # 记录迭代级指标
            if iter_num % 10 == 0:
                writer.add_scalar('train/iter_loss', loss.item(), iter_num)
                writer.add_scalar('train/iter_lr', optimizer.param_groups[0]['lr'], iter_num)
                writer.add_scalar('train/iter_ce', loss_ce.item(), iter_num)

            # 可视化样本
            if iter_num % 100 == 0:
                model_to_check = model.module if hasattr(model, 'module') else model  # 兼容单GPU和多GPU
                if hasattr(model_to_check, 'freq_edge_enhance'):
                    model_to_check.freq_edge_enhance._visualize_to_tensorboard(writer, iter_num)
                with torch.no_grad():
                    img = images[0].cpu().numpy().transpose(1, 2, 0)  # (H,W,C)
                    # 如果 img 是 float 类型且范围不在 [0, 1]，手动归一化
                    if img.dtype == np.float32 or img.dtype == np.float64:
                        img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0, 1]

                    # 如果 img 是 uint8 类型，确保范围在 [0, 255]
                    elif img.dtype == np.uint8:
                        pass  # 无需处理
                    else:
                        img = img.astype(np.float32)  # 转换类型
                        img = (img - img.min()) / (img.max() - img.min())
                    # print("Image shape:", img.shape)  # 检查维度（HWC/HW/CHW）
                    # print("Image dtype:", img.dtype)  # 应该是 uint8 或 float32
                    # print("Min/Max:", img.min(), img.max())  # 应该在 [0, 1] 或 [0, 255]
                    pred = torch.argmax(outputs[0], dim=0).cpu().numpy()  # (H,W)
                    label = labels[0].cpu().numpy()  # 确保是(H,W)

                    writer.add_image('train/image', img, iter_num, dataformats='HWC')
                    writer.add_image('train/pred', pred * 50, iter_num, dataformats='HW')
                    writer.add_image('train/gt', label * 50, iter_num, dataformats='HW')  # 现在label是二维

        # 计算epoch指标
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        avg_ce = epoch_ce / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)

        # 验证阶段
        val_metrics = validate(model, val_loader, device, args.num_classes)
        # scheduler.step(val_metrics['loss'])#new删

        # 记录epoch级指标
        writer.add_scalar('epoch/train_loss', avg_loss, epoch)
        writer.add_scalar('epoch/train_ce', avg_ce, epoch)
        writer.add_scalar('epoch/train_dice', avg_dice, epoch)
        writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
        writer.add_scalar('epoch/val_dice', val_metrics['dice'], epoch)
        writer.add_scalar('epoch/lr', optimizer.param_groups[0]['lr'], epoch)

        # 记录类别Dice
        for cls in range(args.num_classes):
            writer.add_scalar(f'class_dice/class_{cls}', val_metrics['class_dice'][cls], epoch)

        # 格式化日志输出
        class_dice_str = ' | '.join([f'C{i}:{v:.3f}' for i, v in enumerate(val_metrics['class_dice'])])
        logger.info(
            f"Epoch {epoch + 1}/{args.max_epochs} [{epoch_time:.1f}s] "
            f"Train Loss: {avg_loss:.4f} (CE {avg_ce:.4f}, Dice {avg_dice:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} [{class_dice_str}] | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # 保存最佳模型
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))
            logger.info(f"New best model saved at epoch {epoch + 1} with Dice {best_dice:.4f}")

        # 定期保存检查点
        if (epoch + 1) % 50 == 0 or epoch == args.max_epochs - 1:
            save_path = os.path.join(snapshot_path, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)
            logger.info(f"Checkpoint saved at epoch {epoch + 1}")

        # 早停机制
        if epoch - best_epoch >= args.patience:
            logger.info(f"No improvement for {args.patience} epochs, early stopping...")
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'early_stop.pth'))
            break

    writer.close()
    logger.info(f"Training completed! Best model: epoch {best_epoch + 1} with Dice {best_dice:.4f}")
    return "Training Finished!"



def validate(model, dataloader, device, num_classes, ignore_index=0):
    """改进的验证函数，忽略背景类别评估"""
    model.eval()
    total_loss = 0.0
    ce_loss = CrossEntropyLoss(ignore_index=ignore_index)  # 忽略背景类
    dice_loss = DiceLoss(num_classes, ignore_index=ignore_index)

    # 仅统计前景类别
    dice_scores = np.zeros(num_classes - 1)  # 排除背景类
    counts = np.zeros(num_classes - 1)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', unit='batch'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device).long().squeeze(1)

            # 适配三通道输入
            # if images.size(1) == 1:
            #     images = images.repeat(1, 3, 1, 1)

            outputs = model(images)

            # 计算损失（自动忽略背景类）
            loss_ce = ce_loss(outputs, labels.squeeze(1))
            loss_dice = dice_loss(outputs, labels, softmax=True)
            total_loss += 0.5 * loss_ce.item() + 0.5 * loss_dice.item()

            # 计算前景类Dice
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = labels.squeeze(1).cpu().numpy()

            # 仅评估前景类别 (1到num_classes-1)
            for cls in range(1, num_classes):
                pred_mask = (preds == cls)
                true_mask = (targets == cls)
                intersection = np.sum(pred_mask & true_mask)
                union = np.sum(pred_mask) + np.sum(true_mask)

                if union > 0:  # 仅在有该类别时统计
                    dice = (2. * intersection) / (union + 1e-6)
                    dice_scores[cls - 1] += dice  # 索引调整为0开始
                    counts[cls - 1] += 1

    # 计算平均Dice（仅统计出现过的类别）
    valid_classes = counts > 0
    mean_dice = np.sum(dice_scores[valid_classes]) / (np.sum(counts[valid_classes]) + 1e-6)

    # 重建完整类别Dice列表（背景类置0）
    full_class_dice = [0.0] * num_classes
    for cls in range(1, num_classes):
        full_class_dice[cls] = dice_scores[cls - 1] / (counts[cls - 1] + 1e-6) if counts[cls - 1] > 0 else 0.0

    return {
        'loss': total_loss / len(dataloader),
        'dice': mean_dice,
        'class_dice': full_class_dice  # 保持与类别数一致的长度
    }


