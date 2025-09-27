import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import cv2
import os
import pdb


class FocalLoss(nn.Module):
    def __init__(self, gamma=4, alpha=None):
        """
        Args:
            gamma: 调制系数 (γ >= 0)
            alpha: 类别平衡权重 (Tensor)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')  # (B,H,W)
        pt = torch.exp(-ce_loss)  # 预测概率p_t

        # 应用α平衡权重
        if self.alpha is not None:
            alpha = self.alpha[target].to(pred.device)  # (B,H,W)
            focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# 修改DiceLoss实现（utils20.py）
class DiceLoss(nn.Module):
    def __init__(self, n_classes, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, input, target, softmax=False):
        # 确保target是整数索引
        if target.dtype != torch.long:
            target = target.long()

        # 确保target是2D或3D索引 (B,H,W) 或 (B,1,H,W)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)  # 从 [B,1,H,W] 变为 [B,H,W]

        # 检查target值是否在有效范围内
        # print(target.size)
        # print(self.n_classes)
        assert target.max() < self.n_classes, f"Target contains invalid class index {target.max()} >= {self.n_classes}"

        # 转换为one-hot编码
        target_onehot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()

        # 其余代码保持不变...
        if softmax:
            input = F.softmax(input, dim=1)
        smooth = 1e-5
        input = input.contiguous()
        target_onehot = target_onehot.contiguous()
        intersection = (input * target_onehot).sum(dim=(2, 3))
        union = (input + target_onehot).sum(dim=(2, 3))
        loss = 1 - (2. * intersection + smooth) / (union + smooth)
        return loss.mean()


def calculate_metric_percase(pred, gt):
    """
    修改后的指标计算函数，当真实标签中不存在当前类别时返回 NaN
    返回: dice, hd95, iou, ppv, tpr
    """
    # 检查真实标签中是否存在当前类别
    if gt.sum() == 0:  # 真实标签中没有当前类别
        return np.nan, np.nan, np.nan, np.nan, np.nan  # 返回5个NaN

    # 如果真实标签中存在当前类别，正常计算指标
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    # 计算 Dice 系数
    dice = metric.binary.dc(pred, gt)

    # 计算 HD95
    if pred.sum() > 0:  # 预测结果不为空
        hd95 = metric.binary.hd95(pred, gt)
    else:  # 预测结果为空（全0）
        hd95 = 0  # HD95 设为0

    # 计算IOU (Jaccard Index)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / (union + 1e-6)

    # 计算PPV (Precision)
    true_pos = np.logical_and(pred == 1, gt == 1).sum()
    false_pos = np.logical_and(pred == 1, gt == 0).sum()
    ppv = true_pos / (true_pos + false_pos + 1e-6)

    # 计算TPR (Recall/Sensitivity)
    false_neg = np.logical_and(pred == 0, gt == 1).sum()
    tpr = true_pos / (true_pos + false_neg + 1e-6)

    return dice, hd95, iou, ppv, tpr


import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import cv2
import os


# --------------------- 优化后处理核心 ---------------------
class EdgeAwarePostprocessor:
    def __init__(self, classes):
        self.classes = classes
        self.edge_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=np.float32)

    def _get_edge_map(self, mask):
        """获取精细边缘图"""
        edges = cv2.filter2D(mask.astype(np.float32), -1, self.edge_kernel)
        return np.abs(edges) > 0.1

    def process(self, pred, image):
        """
        边缘感知的后处理
        Args:
            pred: 原始预测 (H,W)
            image: 输入图像 (H,W)
        Returns:
            优化后的预测
        """
        processed = pred.copy()

        # 确保图像和预测尺寸一致
        if image.shape != pred.shape:
            image = zoom(image,
                         (pred.shape[0] / image.shape[0],
                          pred.shape[1] / image.shape[1]),
                         order=1)

        # 对每个前景类别单独处理
        for cls in range(1, self.classes):
            mask = (pred == cls).astype(np.uint8)
            if np.sum(mask) < 10:  # 忽略小区域
                continue

            # Step 1: 提取边缘区域
            edges = self._get_edge_map(mask)

            # Step 2: 在边缘区域进行精确优化
            if np.any(edges):
                # 自适应形态学操作
                kernel_size = max(1, int(0.5 * np.sqrt(np.sum(mask))))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

                # 仅对边缘区域做精细化处理
                refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                processed[edges] = refined[edges] * cls

                # 基于图像梯度的边缘修正
                img_gray = (image * 255).astype(np.uint8)
                if img_gray.shape != edges.shape:
                    img_gray = cv2.resize(img_gray, (edges.shape[1], edges.shape[0]))

                img_grad = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)
                if img_grad.shape == edges.shape:  # 确保尺寸匹配
                    grad_thresh = np.percentile(img_grad[edges], 75) if edges.sum() > 0 else 0
                    strong_edges = (img_grad > grad_thresh) & edges
                    processed[strong_edges] = mask[strong_edges] * cls  # 保留强边缘处的原始预测

        return processed


import numpy as np
import cv2


class PrecisionOptimizer:
    def __init__(self, classes):
        self.classes = classes
        # 可调节参数
        self.edge_thresh = 0.15  # 边缘检测阈值
        self.grad_weight = 0.7  # 梯度权重
        self.size_thresh = 15  # 最小处理区域大小
        self.dilation_iters = 1  # 膨胀迭代次数

    def _safe_resize(self, img, target_shape):
        """安全的尺寸调整方法"""
        if img.shape != target_shape:
            return cv2.resize(img, (target_shape[1], target_shape[0]))
        return img

    def _get_enhanced_edges(self, mask, image):
        """增强型边缘检测（修复尺寸问题）"""
        # 统一尺寸
        target_shape = mask.shape
        image = self._safe_resize(image, target_shape)

        # 结构边缘检测
        edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150) / 255

        # 图像梯度融合（确保尺寸一致）
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.abs(grad_x) + np.abs(grad_y)
        grad_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
        grad_norm = self._safe_resize(grad_norm, target_shape)

        # 融合边缘信息
        combined = edges * (1 - self.grad_weight) + grad_norm * self.grad_weight
        return combined > self.edge_thresh

    def _adaptive_morph(self, mask):
        """自适应形态学操作"""
        area = np.sum(mask)
        if area < 100:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            kernel_size = max(3, int(np.sqrt(area) / 8))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated = cv2.dilate(mask, kernel, iterations=self.dilation_iters)
            return cv2.erode(dilated, kernel, iterations=self.dilation_iters)

    def process(self, pred, image):
        """精度优化主流程（修复尺寸问题）"""
        # 统一基准尺寸
        target_shape = pred.shape
        image = self._safe_resize(image, target_shape)
        image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-6)
        processed = pred.copy()

        for cls in range(1, self.classes):
            mask = (pred == cls).astype(np.uint8)
            if np.sum(mask) < self.size_thresh:
                continue

            # 增强边缘检测
            edges = self._get_enhanced_edges(mask, image_normalized)

            if np.any(edges):
                # 核心区域保护
                kernel_p = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                protected = cv2.erode(mask, kernel_p)

                # 边缘优化
                optimized = self._adaptive_morph(mask)

                # 结果融合
                processed[(edges > 0) & (protected == 0)] = optimized[(edges > 0) & (protected == 0)] * cls

        return processed


import numpy as np
import cv2


class ConservativePostprocessor:
    def __init__(self, classes):
        self.classes = classes
        # 超参数（可根据数据集调整）
        self.max_kernel_size = 3  # 最大形态学核大小
        self.min_area = 5  # 最小处理区域面积阈值

    def process(self, pred, image=None):
        """
        保守型后处理：仅执行必要的平滑和去噪
        参数：
            pred: 模型预测结果 (H,W)
            image: 可选，原始图像用于参考（本方案未使用）
        返回：
            后处理结果
        """
        processed = pred.copy()

        for cls in range(1, self.classes):
            mask = (pred == cls).astype(np.uint8)

            # 跳过小区域
            if np.sum(mask) < self.min_area:
                continue

            # Step 1: 去除孤立点（先开后闭）
            kernel = np.ones((self.max_kernel_size, self.max_kernel_size), np.uint8)
            processed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)

            # Step 2: 仅更新发生变化的区域
            change_region = (processed_mask != mask)
            processed[change_region] = processed_mask[change_region] * cls

        return processed

    import cv2
    import numpy as np

class CompactPostprocessor:
    def __init__(self, classes):
        self.classes = classes
        # 针对224×224优化的参数
        self.kernel_sizes = {
            'small': (2, 2),  # 小区域处理
            'medium': (3, 3),  # 中等区域
            'large': (5, 5)  # 大区域（慎用）
        }
        self.area_thresholds = {
            'isolated': 10,  # 孤立点面积阈值
            'small': 50,  # 小区域阈值
            'medium': 200  # 中等区域阈值
        }

    def _get_adaptive_kernel(self, area):
        """根据区域面积自适应选择核大小"""
        if area < self.area_thresholds['small']:
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_sizes['small'])
        elif area < self.area_thresholds['medium']:
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_sizes['medium'])
        else:
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_sizes['large'])

    def process(self, pred):
        """
        针对224×224的轻量后处理
        处理逻辑：
        1. 去除孤立点（面积<10px）
        2. 对小区域(50px)使用2x2核平滑
        3. 对中等区域(200px)使用3x3核
        4. 大区域保持原样
        """
        processed = pred.copy()

        for cls in range(1, self.classes):
            mask = (pred == cls).astype(np.uint8)
            area = np.sum(mask)

            # 跳过极小区域
            if area < self.area_thresholds['isolated']:
                processed[mask > 0] = 0  # 直接清除孤立点
                continue

            # 自适应处理
            kernel = self._get_adaptive_kernel(area)

            # 先开后闭组合去噪
            temp = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)

            # 仅更新发生变化的区域
            changed = (temp != mask)
            processed[changed] = temp[changed] * cls

        return processed

# --------------------- 完整测试流程 ---------------------
_case_buffer = {}


def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1):
    """
    修改后的测试函数，包含尺寸安全的优化后处理
    """
    global _case_buffer

    # 数据准备
    image_slice = image.squeeze(0).cpu().numpy().squeeze()
    label_slice = label.squeeze(0).cpu().numpy().squeeze()

    # 尺寸调整
    if image_slice.shape != patch_size:
        image_slice = zoom(image_slice,
                           (patch_size[0] / image_slice.shape[0],
                            patch_size[1] / image_slice.shape[1]),
                           order=3)

    input_tensor = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0).float().cuda()

    # 模型预测
    with torch.no_grad():
        net.eval()
        outputs = net(input_tensor)
        pred_slice = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze().cpu().numpy()

    # 尺寸还原
    if pred_slice.shape != label_slice.shape:
        pred_slice = zoom(pred_slice,
                          (label_slice.shape[0] / pred_slice.shape[0],
                           label_slice.shape[1] / pred_slice.shape[1]),
                          order=0)

    # 应用尺寸适配的后处理
    # postprocessor = CompactPostprocessor(classes)
    # processed_pred = postprocessor.process(pred_slice)

    # 病例数据堆叠
    if case not in _case_buffer:
        _case_buffer[case] = {
            'image_volume': [],
            'label_volume': [],
            'pred_volume': []
        }
    _case_buffer[case]['image_volume'].append(image_slice)
    _case_buffer[case]['label_volume'].append(label_slice)
    _case_buffer[case]['pred_volume'].append(pred_slice)

    # 指标计算
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(pred_slice == i, label_slice == i))

    return metric_list



def save_pending_cases(test_save_path, z_spacing=1):
    """保存所有缓存的病例数据"""
    global _case_buffer
    if not test_save_path:
        _case_buffer = {}
        return

    os.makedirs(test_save_path, exist_ok=True)
    for case, data in _case_buffer.items():
        # 堆叠切片为3D体积 (D,H,W)
        for vol_type in ['image_volume', 'label_volume', 'pred_volume']:
            vol = np.stack(data[vol_type], axis=0)
            vol = np.rot90(vol, k=1, axes=(1, 2))

            itk_image = sitk.GetImageFromArray(vol.astype(np.float32))
            itk_image.SetSpacing((1, 1, z_spacing))
            sitk.WriteImage(itk_image,
                            os.path.join(test_save_path, f"{case}_{vol_type[:4]}.nii.gz"))
    _case_buffer = {}



