import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

# ================= 1. 全局配置 =================
DATA_DIR = './dataset/camvid'  # 请确认您的数据路径是否正确
ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 显存如果不够（报错 OOM），把这里改成 4
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 500
INPUT_HEIGHT = 384
INPUT_WIDTH = 480

MODEL_SAVE_PATH = './best_model_camvid_5classes.pth'

# ================= 2. 类别映射配置 (5分类) =================
# 0=背景(其他所有), 1=路, 2=人, 3=车, 4=骑行者
CAMVID_MAPPING = {
    'road': 1, 'lane_marking_driving': 1,
    'pedestrian': 2, 'child': 2,
    'bicyclist': 4,       # 单独分类
    'car': 3, 'truck': 3, 'bus': 3, 'train': 3, 'heavy_vehicle': 3, 'pickup_truck': 3, 'van': 3
}

# CamVid 原始标签顺序（必须与数据集一致）
CAMVID_CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
                  'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']

# ================= 3. 数据集类 =================
class CamVidDataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # 简单的内存缓存
        self.images_cache = [None] * len(self.ids)
        self.masks_cache = [None] * len(self.ids)

    def __getitem__(self, i):
        # 1. 读取图像与标签（带缓存）
        if self.images_cache[i] is not None:
            image = self.images_cache[i]
            mask = self.masks_cache[i]
        else:
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0) # 灰度读取

            image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)

            self.images_cache[i] = image
            self.masks_cache[i] = mask

        # 2. 生成目标掩码 (Mapping)
        target_mask = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.uint8)
        
        for class_name, target_id in CAMVID_MAPPING.items():
            if class_name in CAMVID_CLASSES:
                src_id = CAMVID_CLASSES.index(class_name)
                target_mask[mask == src_id] = target_id

        # ============ 【关键修复】 ============
        # 强制转换为 uint8，防止 OpenCV 报错 "int64 is not supported"
        target_mask = target_mask.astype(np.uint8)
        # ====================================

        # 3. 数据增强
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

# ================= 4. 辅助函数 =================
def get_training_augmentation():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=0),
        albu.RandomBrightnessContrast(p=0.2),
        albu.GaussNoise(p=0.1),
        albu.Perspective(p=0.5),
    ])

def get_preprocessing(preprocessing_fn):
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def transform(image, mask):
        image = preprocessing_fn(image)
        image = to_tensor(image)
        mask = torch.from_numpy(mask).long()
        return {"image": image, "mask": mask}

    return transform

# ================= 5. 主训练循环 =================
if __name__ == '__main__':
    print(f"========== 🚀 启动 CamVid 训练 (5分类版) ==========")
    print(f"⚙️  设备: {DEVICE}")
    print(f"📂 模型保存路径: {MODEL_SAVE_PATH}")

    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'train_labels')

    if not os.path.exists(x_train_dir):
        print(f"❌ 错误：找不到数据集路径: {x_train_dir}")
        print("请检查 DATA_DIR 变量设置是否正确。")
        exit()

    # 获取预处理函数
    try:
        prep_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    except:
        print("⚠️ 无法获取预处理函数（网络问题？），使用默认处理。")
        prep_fn = smp.encoders.get_preprocessing_fn(ENCODER, "imagenet")

    # 创建 Dataset 和 DataLoader
    train_dataset = CamVidDataset(
        x_train_dir, y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(prep_fn)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 模型加载/创建
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🔄 加载已有模型继续训练: {MODEL_SAVE_PATH}")
        model = torch.load(MODEL_SAVE_PATH, weights_only=False)
    else:
        print("✨ 创建全新 DeepLabV3+ (MobileNetV2) 模型...")
        try:
            model = smp.DeepLabV3Plus(
                encoder_name=ENCODER, 
                encoder_weights=ENCODER_WEIGHTS, 
                classes=5, 
                activation=None
            )
        except Exception as e:
            print(f"⚠️ 预训练权重下载失败 ({e})，正在尝试不使用预训练权重启动...")
            model = smp.DeepLabV3Plus(
                encoder_name=ENCODER, 
                encoder_weights=None, 
                classes=5, 
                activation=None
            )

    model.to(DEVICE)

    # 损失函数与优化器
    loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    min_loss = float('inf')

    # 开始训练
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

            if i % 10 == 0:
                print(f"  Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch + 1} 完成 | Avg Loss: {avg_loss:.4f}")

        # 保存最佳模型
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model, MODEL_SAVE_PATH)
            print(f"  💾 模型已更新，Loss 降至: {min_loss:.4f}")