import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
import os

class FrequencyDomainEdgeEnhancement(nn.Module):
    """改进版频域边缘增强模块（局部高斯模糊）"""

    def __init__(self, high_pass_ratio=0.3, blend_ratio=0.6, blur_region_ratio=0.3):
        super().__init__()
        self.high_pass_ratio = high_pass_ratio
        self.blend_ratio = blend_ratio
        self.blur_region_ratio = blur_region_ratio  # 控制模糊区域宽度比例

        # 高斯模糊核（σ=0.8）
        self.gaussian_blur = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 1, kernel_size=3, bias=False)
        )
        kernel = torch.tensor([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=torch.float32) / 16.0
        self.gaussian_blur[1].weight.data.copy_(kernel.view(1,1,3,3))
        self.gaussian_blur[1].weight.requires_grad = False

    def forward(self, x):
        # 转换为灰度图
        if x.size(1) > 1:
            x_gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
            x_gray = x_gray.unsqueeze(1)
        else:
            x_gray = x.clone()

        # 创建局部模糊区域掩码（中间偏左侧）
        _, _, h, w = x.shape
        blur_mask = torch.zeros((h, w), device=x.device)
        center_w = int(w * 0.2)  # 偏左侧中心
        region_width = int(w * self.blur_region_ratio)
        blur_mask[:, max(0, center_w-region_width//2):min(w, center_w+region_width//2)] = 1

        # 对选定区域进行高斯模糊
        blurred = self.gaussian_blur(x_gray)
        x_gray = x_gray * (1 - blur_mask) + blurred * blur_mask

        # 傅里叶变换与频域处理
        fft_complex = torch.fft.fft2(x_gray)
        fft_shifted = torch.fft.fftshift(fft_complex)

        # 动态高通滤波
        center_h, center_w = h // 2, w // 2
        cut_off = int(min(h, w) * self.high_pass_ratio)
        mask = torch.ones((h, w), device=x.device)
        mask[center_h - cut_off:center_h + cut_off, center_w - cut_off:center_w + cut_off] = 0

        # 边缘提取
        fft_highpass = fft_shifted * mask
        edge_map = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_highpass)))

        # 增强边缘对比度
        edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)
        edge_map = torch.pow(edge_map, 0.5)  # 增强低强度边缘
        edge_map = edge_map * (1 + 0.5 * torch.sin(edge_map * 3.14))  # 非线性增强

        # 与原图融合
        enhanced = x * (1 - self.blend_ratio) + edge_map * self.blend_ratio
        return enhanced.clamp(0, 1)

def test_and_save_results(image_path, output_dir):
    """测试并保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 加载图像
    img = Image.open(image_path).convert('RGB')
    img_tensor = ToTensor()(img).unsqueeze(0)

    # 初始化增强器（调整参数控制模糊区域）
    enhancer = FrequencyDomainEdgeEnhancement(
        high_pass_ratio=0.2,
        blend_ratio=0.2,
        blur_region_ratio=0.2  # 模糊区域占图像宽度的20%
    )

    # 处理图像
    with torch.no_grad():
        enhanced_img = enhancer(img_tensor)

    # 保存结果
    def save_image(tensor, filename):
        arr = tensor.squeeze(0).permute(1, 2, 0).numpy()
        plt.imsave(os.path.join(output_dir, filename), np.clip(arr, 0, 1))

    save_image(img_tensor, "original.png")
    save_image(enhanced_img, "enhanced.png")

    # 可视化对比
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img[0].permute(1, 2, 0))
    plt.title("Enhanced (局部模糊+边缘增强)")
    plt.axis('off')

    plt.savefig(os.path.join(output_dir, "comparison.png"), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"结果已保存至: {output_dir}")

if __name__ == "__main__":
    test_and_save_results(
        image_path="/home/zdt/see/0Unet/images/Case1_img_slice_006.png",
        output_dir="/home/zdt/see/enhanced2"
    )