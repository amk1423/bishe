import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import os
import traceback

# ================= 配置区域 =================
# === 修改点 0: 必须加载那个新的 5分类模型 ===
MODEL_PATH = './best_model_camvid_5classes.pth'

# 测试图片
# IMAGE_PATH = './test02.jpeg'
IMAGE_PATH = './0016E5_04350.png'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_HEIGHT = 384
INPUT_WIDTH = 480

# === 修改点 1: 颜色定义增加 ID=4 (蓝色) ===
# 0=背景, 1=路(绿), 2=人(黄), 3=车(红), 4=骑行者(蓝)
CLASS_COLORS = {
    0: (0, 0, 0),  # 背景
    1: (0, 255, 0),  # 路
    2: (255, 255, 0),  # 人
    3: (255, 0, 0),  # 车
    4: (0, 0, 255)  # 骑行者 (蓝)
}


def colorize_mask(class_map):
    colored = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)
    for id, color in CLASS_COLORS.items():
        colored[class_map == id] = color
    return colored


if __name__ == '__main__':
    os.environ['HF_HUB_OFFLINE'] = '1'
    print(f"🚀 设备: {DEVICE}")
    print(f"📂 加载模型: {MODEL_PATH}")

    try:
        map_loc = None if torch.cuda.is_available() else 'cpu'
        loaded_obj = torch.load(MODEL_PATH, map_location=map_loc, weights_only=False)

        # === 修改点 2: 骨架也要改成 5 类 ===
        model = smp.DeepLabV3Plus(
            encoder_name='mobilenet_v2',
            encoder_weights=None,
            classes=5,  # 👈 必须是 5
            activation=None
        )

        if isinstance(loaded_obj, dict):
            state_dict = loaded_obj
        else:
            state_dict = loaded_obj.state_dict()

        model.load_state_dict(state_dict)
        print("✅ 模型加载成功！")

    except FileNotFoundError:
        print(f"❌ 找不到文件 '{MODEL_PATH}'")
        print("请先运行 train.py 训练出新模型！")
        exit()
    except Exception as e:
        print(f"❌ 错误: {e}")
        traceback.print_exc()
        exit()

    model.to(DEVICE)
    model.eval()

    # 读取与预处理
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"❌ 找不到图片: {IMAGE_PATH}")
        exit()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))

    try:
        prep_fn = smp.encoders.get_preprocessing_fn('mobilenet_v2', 'imagenet')
        img_prep = prep_fn(img_resized)
    except:
        img_prep = img_resized / 255.0

    img_prep = img_prep.transpose(2, 0, 1).astype('float32')
    tensor = torch.from_numpy(img_prep).unsqueeze(0).to(DEVICE)

    # 推理
    print("🤖 正在推理...")
    with torch.no_grad():
        output = model(tensor)
        pred = np.argmax(output.squeeze().cpu().numpy(), axis=0)

    # 结果
    unique_classes = np.unique(pred)
    print(f"🔍 检测结果类别 ID: {unique_classes}")

    colored_mask = colorize_mask(pred)
    mask_bool = (pred > 0)[:, :, None]
    blended = np.where(mask_bool, img_resized * 0.6 + colored_mask * 0.4, img_resized)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(blended.astype(np.uint8))
    plt.axis('off')
    plt.show()