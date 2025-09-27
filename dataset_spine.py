import os
import numpy as np
import nibabel as nib  # 使用 nibabel 来加载 nii.gz 格式
from torch.utils.data import Dataset, Sampler
import torch
from PIL import Image
import cv2
import bisect
import itertools
import random
from scipy import ndimage
from scipy.ndimage import zoom
import matplotlib.pyplot as plt  # 添加可视化支持
import albumentations as A


from albumentations.core.transforms_interface import DualTransform

import albumentations as A
import numpy as np
import random
from albumentations.core.transforms_interface import DualTransform


class BalancedClassAugmentation(DualTransform):
    def __init__(self, p=0.7, minority_classes=None, class_weights=None):
        super().__init__(p)
        self.minority_classes = minority_classes or []  # 确保有默认值
        self.class_weights = class_weights or [1.0] * 16  # 默认权重

        # 修正1：使用新版本的GaussNoise参数格式
        self.transforms = A.Compose([
            A.ElasticTransform(alpha=80, sigma=6, p=0.4),
            A.RandomGamma(gamma_limit=(70, 130), p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),

            # 修正重点：新版Albumentations使用mean和std参数
            A.GaussNoise(
                mean_range= (0.0, 0.0),
                std_range=(0.2, 0.44),
                # mean=0,  # 噪声均值
                # std=(10.0, 30.0),  # 标准差范围
                per_channel=True,  # 每个通道独立应用
                p=0.6
            ),

            A.MotionBlur(blur_limit=5, p=0.4),
        ])

    def apply(self, img, mask=None, **params):
        # 修正2：正确检查mask中的少数类别
        if mask is not None:
            present_classes = np.unique(mask)
            minority_present = any(
                cls_id in self.minority_classes
                for cls_id in present_classes
            )
        else:
            minority_present = False

        # 修正3：使用整体权重而非单个类别权重
        if minority_present and random.random() < self.p * 1.5:  # 少数类存在时增强概率提高
            result = self.transforms(image=img, mask=mask)
            return result['image']
        return img

class ProblemClassAugmentation(DualTransform):
    def __init__(
        self,
        p=0.7,  # 增强概率
        always_apply=False,
        noise_var_limit=(10.0, 50.0),  # 噪声方差范围
        blur_limit=7,  # 模糊强度
        ghost_alpha=0.3  # 伪影透明度
    ):
        super().__init__(always_apply, p)
        self.noise_var_limit = noise_var_limit
        self.blur_limit = blur_limit
        self.ghost_alpha = ghost_alpha

        # # 修改 ProblemClassAugmentation 的 transforms 部分
        # self.transforms = A.Compose([
        #     A.ElasticTransform(alpha=120, sigma=8, p=0.3),  # 新增弹性变形
        #     A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # 新增对比度调整
        #     A.GaussNoise(var_limit=self.noise_var_limit, p=0.7),
        #     A.GaussianBlur(blur_limit=(3, self.blur_limit), p=0.7),
        #     A.MotionBlur(blur_limit=self.blur_limit, p=0.5),
        #     A.Lambda(name="GhostArtifact", image=self.apply_ghost, mask=self.apply_to_mask)
        # ])

        # 定义新的增强组合（无条件触发）
        self.transforms = A.Compose([
            A.GaussNoise(var_limit=self.noise_var_limit, p=0.7),
            A.GaussianBlur(blur_limit=(3, self.blur_limit), p=0.7),
            A.MotionBlur(blur_limit=self.blur_limit, p=0.5),
            A.Lambda(name="GhostArtifact",
                     image=self.apply_ghost,
                     mask=self.apply_to_mask)
        ])

    def apply_ghost(self, image, **params):
        """自定义伪影：图像平移叠加"""
        if random.random() < 0.5:
            h, w = image.shape[:2]
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
            translated = np.roll(image, dx, axis=1) if dx != 0 else image.copy()
            translated = np.roll(translated, dy, axis=0) if dy != 0 else translated
            ghost_image = cv2.addWeighted(image, 1 - self.ghost_alpha,
                                         translated, self.ghost_alpha, 0)
            return ghost_image
        return image

    def apply_to_mask(self, mask, **params):
        # Mask保持原样
        return mask

    def apply(self, img, **params):
        return self.transforms(image=img)['image']

    def __call__(self, force_apply=False, **kwargs):
        # 无条件应用增强（不再检查类别）
        return super().__call__(force_apply=force_apply, **kwargs)


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size, target_depth=12):
        self.output_size = output_size
        self.target_depth = target_depth

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        current_depth = label.shape[2]
        if current_depth > self.target_depth:
            z = np.random.randint(0, current_depth - self.target_depth)
            image = image[:, :, z:z + self.target_depth]
            label = label[:, :, z:z + self.target_depth]
        elif current_depth < self.target_depth:
            pad_depth = self.target_depth - current_depth
            image = np.pad(image, ((0, 0), (0, 0), (0, pad_depth)), mode="constant", constant_values=0)
            label = np.pad(label, ((0, 0), (0, 0), (0, pad_depth)), mode="constant", constant_values=0)

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class CenterCrop:
    def __init__(self, output_size):
        assert len(output_size) == 2, "output_size must be a tuple of (height, width)"
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode="constant", constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode="constant", constant_values=0)

        w, h = image.shape
        w1 = int(round((w - self.output_size[0]) / 2.0))
        h1 = int(round((h - self.output_size[1]) / 2.0))

        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

        return {"image": image, "label": label}

class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode="constant", constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode="constant", constant_values=0)

        w, h, d = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {"image": image, "label": label}

class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image_clahe = np.zeros_like(image, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

        for i in range(image.shape[2]):
            slice_image = image[:, :, i]
            slice_image = cv2.normalize(slice_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image_clahe[:, :, i] = clahe.apply(slice_image)

        return {"image": image_clahe, "label": label}

class ToTensor:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = image[np.newaxis, ...].astype(np.float32)  # 确保为 float32
        image = torch.from_numpy(image)  # 不在此处移动至CUDA
        label = torch.from_numpy(label).long()
        return {"image": image, "label": label}
# class ToTensor:
#     def __call__(self, sample):
#         image, label = sample["image"], sample["label"]
#         image = image[np.newaxis, ...].astype(np.float64)  # Add channel dimension
#         return {"image": torch.from_numpy(image), "label": torch.from_numpy(label).long()}

class Resize:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        original_height, original_width = image.shape[0], image.shape[1]
        # print(image.shape[0])
        # print(image.shape[1])
        target_height, target_width = self.output_size[0],self.output_size[1]
        # print(self.output_size[0])
        # print(self.output_size[1])
        height_scale = target_height / original_height
        width_scale = target_width / original_width
        # print(height_scale)
        # print(width_scale)
        min_scale = min(height_scale, width_scale)

        image = ndimage.zoom(image, [height_scale, width_scale], order=3)
        label = ndimage.zoom(label, [height_scale, width_scale], order=0)

        return {"image": image, "label": label}

from collections import Counter
class SpineDataset(Dataset):
    def __init__(self, base_dir, split="train", list_dir=None, transform=None, common_transform=None, sp_transform=None,
                 target_depth=12, resize_size=(448, 448)):
        self._base_dir = base_dir
        self._list_dir = list_dir
        self.transform = transform
        self.common_transform = common_transform
        self.sp_transform = sp_transform
        self.target_depth = target_depth
        self.resize_size = resize_size
        self.split = split  # 新增split属性
        self.class_counts = None  # 用于存储类别统计
        self.num_classes = 16  # 假设有16个类别

        split_dir = os.path.join(self._base_dir, split)
        mr_folder = os.path.join(split_dir, "MR")
        mask_folder = os.path.join(split_dir, "Mask")

        self.image_list = sorted([os.path.join(mr_folder, f) for f in os.listdir(mr_folder) if f.endswith(".nii.gz")])
        self.label_list = sorted(
            [os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith(".nii.gz")])

        self.valid_slices_info = []
        self.cumulative_depths = []
        current_depth = 0

        for image_path, label_path in zip(self.image_list, self.label_list):
            image = nib.load(image_path).get_fdata()
            original_depth = image.shape[2]

            # 根据split类型决定切片范围
            # if self.split == "train":
            #     # 训练集：去掉前2后2切片
            #     start_slice = 2
            #     end_slice = original_depth - 2
            #     valid_depth = max(original_depth - 4, 0)
            # else:
                # 验证/测试集：保留所有切片
            start_slice = 0
            end_slice = original_depth
            valid_depth = original_depth

            if valid_depth <= 0:
                print(f"Warning: {image_path} has only {original_depth} slices (no valid slices after trimming)")
                continue

            self.valid_slices_info.append({
                "image_path": image_path,
                "label_path": label_path,
                "start_slice": start_slice,
                "end_slice": end_slice
            })

            current_depth += valid_depth
            self.cumulative_depths.append(current_depth)

        print(f"Total {self.cumulative_depths[-1] if self.cumulative_depths else 0} valid slices in {split} set.")
        # self.center_crop = CenterCrop((896, 896))
        self.resize = Resize(resize_size)

    def calculate_class_counts(self):
        """计算整个数据集的每个类别的像素总数"""
        class_counts = np.zeros(self.num_classes, dtype=np.int64)

        # 进度条显示
        from tqdm import tqdm
        print(f"Calculating class counts for {self.split} set...")

        for idx in tqdm(range(len(self))):
            sample = self[idx]
            label = sample["label"]

            # 统计当前切片中每个类别的像素数量
            for cls in range(self.num_classes):
                class_counts[cls] += np.sum(label == cls)

        # 避免零计数
        class_counts = np.maximum(class_counts, 1)
        self.class_counts = class_counts
        return class_counts

    # 添加获取类别统计的方法
    def get_class_counts(self):
        """获取类别统计，如果尚未计算则先计算"""
        if self.class_counts is None:
            return self.calculate_class_counts()
        return self.class_counts

    def __len__(self):
        return self.cumulative_depths[-1] if self.cumulative_depths else 0

    def __getitem__(self, idx):
        if not self.cumulative_depths:
            raise IndexError("No valid slices available")

        # 找到对应的文件索引
        sample_idx = bisect.bisect_right(self.cumulative_depths, idx)
        if sample_idx > 0:
            start_depth = self.cumulative_depths[sample_idx - 1]
        else:
            start_depth = 0

        # 获取该文件的有效切片信息
        valid_info = self.valid_slices_info[sample_idx]
        local_slice_idx = idx - start_depth

        # 计算原始切片索引
        original_slice_idx = valid_info["start_slice"] + local_slice_idx

        # 加载数据
        image = nib.load(valid_info["image_path"]).get_fdata().astype(np.float32)
        label = nib.load(valid_info["label_path"]).get_fdata().astype(np.uint8)

        # 确保索引在有效范围内
        if original_slice_idx >= valid_info["end_slice"]:
            raise IndexError(f"Slice index {original_slice_idx} out of bounds for {valid_info['image_path']}")

            # 应用通用增强（如旋转、翻转）
            if self.transform:
                transformed = self.transform(image=image_slice, mask=label_slice)
                image_slice = transformed['image']
                label_slice = transformed['mask']

            # 应用针对问题类别的增强
            if self.sp_transform and self.split == "train":
                transformed = self.sp_transform(image=image_slice, mask=label_slice)
                image_slice = transformed['image']
                label_slice = transformed['mask']
        # 后续处理保持不变...
        # ... [原有代码保持不变]
        image_slice = image[:, :, original_slice_idx]
        label_slice = label[:, :, original_slice_idx]

        # 后续处理（裁剪、resize等保持不变）
        # cropped_sample = self.center_crop({"image": image_slice, "label": label_slice})
        # image_slice = cropped_sample["image"]
        # label_slice = cropped_sample["label"]

        resized_sample = self.resize({"image": image_slice, "label": label_slice})
        image_slice = resized_sample["image"]
        label_slice = resized_sample["label"]

        # 添加通道维度
        image_slice = image_slice[np.newaxis, ...]
        # label_slice = label_slice[np.newaxis, ...]

        # 处理标签（将 >=10 的类别设为背景）
        # label_slice = np.where(label_slice >= 10, 0, label_slice)#10类
        label_slice = np.clip(label_slice, 0, 19)

        # 提取 case_name
        case_name = os.path.basename(valid_info["image_path"]).split('.')[0].split('_')[-1]

        return {"image": image_slice, "label": label_slice, "case_name": case_name}


def test_class_counts():
    base_dir = "/home/zdt/MCF-main/dataset/Spine"
    dataset = SpineDataset(base_dir=base_dir, split="train", resize_size=(896, 896))

    # 计算并打印类别统计
    class_counts = dataset.calculate_class_counts()
    print("Class counts:")
    for cls, count in enumerate(class_counts):
        print(f"Class {cls}: {count} pixels")

    # 计算并打印类别权重
    class_weights = 1.0 / np.sqrt(class_counts)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print("\nClass weights:")
    for cls, weight in enumerate(class_weights):
        print(f"Class {cls}: {weight:.4f}")


if __name__ == "__main__":
    test_class_counts()
    base_dir = "/home/zdt/MCF-main/dataset/Spine_modified"

    dataset = SpineDataset(base_dir=base_dir, split="train",list_dir= '/home/zdt/MCF-main/lists/lists_Spines', resize_size=(224, 224))

    sample = dataset[0]
    print("Image type:", sample["image"].dtype)  # 应输出 float32
    print("Label type:", sample["label"].dtype)  # 应输出 uint8

    for i in range(10):
        sample = dataset[i]
        print("shape:", sample["image"].shape)
        print("shape:", sample["label"].shape)

        # 检查张量的形状
        print("Tensor shape before squeeze:", sample["image"].shape)
        if sample["image"].shape[0] == 1:
            image = sample["image"].squeeze(0)  # 移除通道维度
            label = sample["label"]
        else:
            image = sample["image"]  # 不需要移除通道维度
            label = sample["label"]

        plt.figure(figsize=(10, 5))

        # 显示图像
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(image, cmap="gray")
        plt.axis("off")

        # 显示标签F
        plt.subplot(1, 2, 2)
        plt.title("Label")
        plt.imshow(label, cmap="jet")
        plt.axis("off")

        plt.show()