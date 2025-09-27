import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_spine import SpineDataset
# from dataset_synapse import Synapse_dataset
# from datasets.dataset_ACDC import ACDCdataset
from utils import test_single_volume
# 从文档1导入的组件
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainers import trainer_spines
# from datasets.dataset_spine import SpineDataset

parser = argparse.ArgumentParser()
# 文档1的主程序逻辑
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, default='/home/zdt/MCF-main/dataset/Spine_modified')
parser.add_argument('--dataset', type=str, default='Spines')
parser.add_argument('--num_classes', type=int, default=16)
parser.add_argument('--list_dir', type=str, default='./lists/lists_Spines')
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--is_savenii', action="store_true")
parser.add_argument('--n_skip', type=int, default=3)
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
# parser.add_argument('--test_save_dir', type=str, default='../predictions')
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--vit_patches_size', type=int, default=16)
parser.add_argument('--output_dir', type=str,
                    default='./results/transunetqing', help='output dir')
parser.add_argument('--test_save_dir', default='./results/transunetqing/predictions', help='saving prediction as nii!')

args = parser.parse_args()



def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    # 初始化指标累加器和计数器
    metric_sum = np.zeros((args.num_classes - 1, 5))  # 形状: (num_classes-1, 5) [dice, hd95, iou, ppv, tpr]
    class_count = np.zeros(args.num_classes - 1)  # 每个类别出现的次数

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path,
                                      case=case_name,
                                      z_spacing=args.z_spacing)

        # 处理每个类别的指标
        for class_idx in range(args.num_classes - 1):
            dice, hd95, iou, ppv, tpr = metric_i[class_idx]

            # 检查是否是有效值（非 NaN）
            if not np.isnan(dice):
                metric_sum[class_idx, 0] += dice
                metric_sum[class_idx, 1] += hd95
                metric_sum[class_idx, 2] += iou
                metric_sum[class_idx, 3] += ppv
                metric_sum[class_idx, 4] += tpr
                class_count[class_idx] += 1

        # 计算当前样本的平均指标（仅考虑存在的类别）
        valid_metrics = [m for m in metric_i if not np.isnan(m[0])]
        if valid_metrics:
            sample_dice = np.mean([m[0] for m in valid_metrics])
            sample_hd95 = np.mean([m[1] for m in valid_metrics])
            sample_iou = np.mean([m[2] for m in valid_metrics])
            sample_ppv = np.mean([m[3] for m in valid_metrics])
            sample_tpr = np.mean([m[4] for m in valid_metrics])
        else:
            sample_dice = sample_hd95 = sample_iou = sample_ppv = sample_tpr = 0.0

        logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_iou %f mean_ppv %f mean_tpr %f' %
                     (i_batch, case_name, sample_dice, sample_hd95, sample_iou, sample_ppv, sample_tpr))

    # 计算每个类别的平均指标
    valid_classes = []  # 存储有效类别的索引
    for i in range(1, args.num_classes):
        class_idx = i - 1  # 从0开始

        if class_count[class_idx] > 0:
            mean_dice = metric_sum[class_idx, 0] / class_count[class_idx]
            mean_hd95 = metric_sum[class_idx, 1] / class_count[class_idx]
            mean_iou = metric_sum[class_idx, 2] / class_count[class_idx]
            mean_ppv = metric_sum[class_idx, 3] / class_count[class_idx]
            mean_tpr = metric_sum[class_idx, 4] / class_count[class_idx]
            valid_classes.append(class_idx)
            logging.info('Class %d (n=%d) mean_dice %f mean_hd95 %f mean_iou %f mean_ppv %f mean_tpr %f' %
                         (i, class_count[class_idx], mean_dice, mean_hd95, mean_iou, mean_ppv, mean_tpr))
        else:
            logging.info('Class %d not present in any test sample' % i)

    # 计算总体性能（只考虑存在的类别）
    if valid_classes:
        mean_dice_all = np.mean(metric_sum[valid_classes, 0] / class_count[valid_classes])
        mean_hd95_all = np.mean(metric_sum[valid_classes, 1] / class_count[valid_classes])
        mean_iou_all = np.mean(metric_sum[valid_classes, 2] / class_count[valid_classes])
        mean_ppv_all = np.mean(metric_sum[valid_classes, 3] / class_count[valid_classes])
        mean_tpr_all = np.mean(metric_sum[valid_classes, 4] / class_count[valid_classes])
    else:
        mean_dice_all = mean_hd95_all = mean_iou_all = mean_ppv_all = mean_tpr_all = 0.0

    logging.info('Overall performance (n=%d classes): mean_dice %f mean_hd95 %f mean_iou %f mean_ppv %f mean_tpr %f' %
                 (len(valid_classes), mean_dice_all, mean_hd95_all, mean_iou_all, mean_ppv_all, mean_tpr_all))

    # 保存病例数据
    from utils import save_pending_cases
    save_pending_cases(test_save_path, args.z_spacing)

    return "Testing Finished!"


# def inference(args, model, test_save_path=None):
#     db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
#     sample = db_test[0]
#     print("Sample image type:", type(sample["image"]))  # 应为 torch.Tensor
#     print("Sample label type:", type(sample["label"]))  # 应为 torch.Tensor
#     print("Image shape:", sample["image"].shape)  # 应为 (1, H, W)
#     print("Label shape:", sample["label"].shape)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
#     logging.info("{} test iterations per epoch".format(len(testloader)))
#     model.eval()
#     metric_list = 0.0
#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         h, w = sampled_batch["image"].size()[2:]
#         image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
#         metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
#                                       test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
#         metric_list += np.array(metric_i)
#         logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
#     from utils.utils import save_pending_cases  # 如果_save_case在utils20中定义
#     save_pending_cases(test_save_path, args.z_spacing)
#     metric_list = metric_list / len(db_test)
#     for i in range(1, args.num_classes):
#         logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
#     performance = np.mean(metric_list, axis=0)[0]
#     mean_hd95 = np.mean(metric_list, axis=0)[1]
#     logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
#     return "Testing Finished!"


if __name__ == "__main__":

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

    dataset_config = {
        'Spines': {
            'Dataset':SpineDataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Spine',
            'num_classes': 16,
            'z_spacing': 4.4
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    vit_config = CONFIGS_ViT_seg[args.vit_name]
    vit_config.n_classes = args.num_classes
    vit_config.n_skip = args.n_skip
    vit_config.patches.size = (args.vit_patches_size, args.vit_patches_size)

    if 'R50' in args.vit_name:
        vit_config.patches.grid = (args.img_size // args.vit_patches_size,) * 2

    net = ViT_seg(vit_config, img_size=args.img_size, num_classes=vit_config.n_classes).cuda()
    # net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    # 在 tests.py 中添加打印语句确认权重路径
    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    print(f"Loading model from: {snapshot}")
    assert os.path.exists(snapshot), "Model weight file not found!"
    # 修改权重加载部分，捕获异常
    try:
        net.load_state_dict(torch.load(snapshot))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit()
    # 在 tests.py 中打印数据路径
    print(f"Volume path: {args.volume_path}")
    print(f"List directory: {args.list_dir}")

    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/spines+transunetqing/'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
        print(test_save_path)
    else:
        test_save_path = None

    inference(args, net, test_save_path)


