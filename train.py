import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

# ================= 1. 全局配置 =================
DATA_DIR = './dataset/camvid'  # 只读取 CamVid
ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 4  # 显存够大可以改 8
LR = 0.0001
EPOCHS = 100  # 建议跑 100 轮
INPUT_HEIGHT = 384
INPUT_WIDTH = 480

# === 修改点 0: 给新模型起个新名字，避开旧文件 ===
MODEL_SAVE_PATH = './best_model_camvid_5classes.pth'

# ================= 2. 关键：修正后的类别映射 =================
# 目标: 0=背景, 1=路, 2=人, 3=车, 4=骑行者(新增)
CAMVID_MAPPING = {
    'road': 1, 'lane_marking_driving': 1,
    'pedestrian': 2, 'child': 2,

    # === 修改点 1: 把骑车的人单独分出来 (ID=4) ===
    'bicyclist': 4,
    # 'motorcyclist': 4, # 如果有摩托车也加这里

    'car': 3, 'truck': 3, 'bus': 3, 'train': 3, 'heavy_vehicle': 3, 'pickup_truck': 3, 'van': 3
}

CAMVID_CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
                  'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']


# ================= 3. 数据集类 (含内存加速) =================
class CamVidDataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # 内存缓存
        self.images_cache = [None] * len(self.ids)
        self.masks_cache = [None] * len(self.ids)

    def __getitem__(self, i):
        # 1. 查缓存
        if self.images_cache[i] is not None:
            image = self.images_cache[i]
            mask = self.masks_cache[i]
        else:
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0)

            image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)

            self.images_cache[i] = image
            self.masks_cache[i] = mask

        # 2. 映射标签
        target_mask = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.longlong)
        for class_name, target_id in CAMVID_MAPPING.items():
            if class_name in CAMVID_CLASSES:
                src_id = CAMVID_CLASSES.index(class_name)
                target_mask[mask == src_id] = target_id

        # 3. 增强
        if self.augmentation:
            sample = self.augmentation(image=image, mask=target_mask)
            image, target_mask = sample['image'], sample['mask']

        # 4. 预处理
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=target_mask)
            image, target_mask = sample['image'], sample['mask']

        return image, target_mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=0),
        albu.RandomBrightnessContrast(p=0.2),
        albu.GaussNoise(p=0.1),
        albu.Perspective(p=0.5),
    ])


def get_preprocessing(preprocessing_fn):
    def to_tensor(x, **kwargs): return x.transpose(2, 0, 1).astype('float32')

    def transform(image, mask):
        image = preprocessing_fn(image)
        image = to_tensor(image)
        mask = torch.from_numpy(mask).long()
        return {"image": image, "mask": mask}

    return transform


# ================= 4. 主程序 =================
if __name__ == '__main__':
    print(f"========== 🚀 启动 CamVid 训练 (5分类版) ==========")
    print(f"📂 模型将保存为: {MODEL_SAVE_PATH}")

    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'train_labels')
    if not os.path.exists(x_train_dir):
        print("❌ 错误：找不到 CamVid 路径")
        exit()

    prep_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = CamVidDataset(
        x_train_dir, y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(prep_fn)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 构建模型
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🔄 发现已存在的5分类模型，正在加载: {MODEL_SAVE_PATH}")
        model = torch.load(MODEL_SAVE_PATH, weights_only=False)
    else:
        print("✨ 未发现同名模型，创建全新的 5分类 模型...")
        # === 修改点 2: 类别数改为 5 ===
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
            classes=5, activation=None
        )
    model.to(DEVICE)

    loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    min_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        print(f"\nEpoch {epoch + 1}/{EPOCHS} ...")

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 20 == 0:
                print(f"  Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch + 1} 结束 | Avg Loss: {avg_loss:.4f}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model, MODEL_SAVE_PATH)
            print(f"  💾 最佳模型已保存到: {MODEL_SAVE_PATH} (Loss: {min_loss:.4f})")