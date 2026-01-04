import cv2
import numpy as np
from collections import Counter

# ⚠️ 这里的路径要换成你 dataset/cityscapes/train_labels 里的任意一张图片！
# 比如: './dataset/cityscapes/train_labels/train1.png'
IMG_PATH = './dataset/cityscapes/train_labels/train1.png'


def analyze_colors():
    print(f"正在读取图片: {IMG_PATH}")
    mask = cv2.imread(IMG_PATH)

    if mask is None:
        print("❌ 错误：找不到图片，请检查路径是否正确！")
        return

    # OpenCV 默认读入是 BGR，我们要转成 RGB 方便人类阅读
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # 把图片展平成像素列表
    pixels = mask_rgb.reshape(-1, 3)
    # 转成 tuple 方便统计
    pixels = [tuple(p) for p in pixels]

    # 统计出现最多的颜色
    counts = Counter(pixels)

    print("\n🔍 这张图里出现最多的前 10 种颜色是 (R, G, B):")
    print("-" * 30)
    for i, (color, count) in enumerate(counts.most_common(10)):
        print(f"{i + 1}. 颜色 {color} \t -> 出现了 {count} 次")
    print("-" * 30)

    # 检查我们的标准颜色是否存在
    print("正在比对标准颜色...")
    standard_road = (128, 64, 128)
    if standard_road in counts:
        print(f"✅ 标准紫色 (路) 存在！")
    else:
        print(f"❌ 标准紫色 (路) 不存在！代码里的字典写错了！")


if __name__ == '__main__':
    analyze_colors()