import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset as BaseDataset

# ================= 1. 全局配置 =================
CAMVID_DIR = './dataset/camvid'
CITY_DIR = './dataset/cityscapes'

ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 6
LR = 0.0001
EPOCHS = 50
INPUT_HEIGHT = 384
INPUT_WIDTH = 480

MODEL_SAVE_PATH = './best_model_mixed.pth'

# ================= 2. CamVid 映射 (保持不变) =================
CAMVID_MAPPING = {
    'road': 1, 'lane_marking_driving': 1,
    'pedestrian': 2, 'bicyclist': 2, 'child': 2,
    'car': 3, 'truck': 3, 'bus': 3, 'train': 3, 'heavy_vehicle': 3, 'pickup_truck': 3, 'van': 3
}
CAMVID_CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
                  'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']


# ================= 3. 数据集类 (核心升级：模糊匹配) =================
class SegmentationDataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, source_type='camvid', augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.source_type = source_type
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # 内存缓存
        self.images_cache = [None] * len(self.ids)
        self.masks_cache = [None] * len(self.ids)

    def __getitem__(self, i):
        # 1. 缓存读取
        if self.images_cache[i] is not None:
            image = self.images_cache[i]
            mask = self.masks_cache[i]
        else:
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.source_type == 'cityscapes':
                mask = cv2.imread(self.masks_fps[i])
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            else:
                mask = cv2.imread(self.masks_fps[i], 0)

            image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)

            self.images_cache[i] = image
            self.masks_cache[i] = mask

        # 2. 标签转换 (这里用了更智能的模糊匹配！)
        target_mask = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.longlong)

        if self.source_type == 'camvid':
            for class_name, target_id in CAMVID_MAPPING.items():
                if class_name in CAMVID_CLASSES:
                    src_id = CAMVID_CLASSES.index(class_name)
                    target_mask[mask == src_id] = target_id

        elif self.source_type == 'cityscapes':
            # === 核心升级：RGB 范围匹配 (容错 +/- 10) ===
            # 我们不再匹配具体的 (128,64,128)，而是匹配一个“紫色范围”

            # 1. 路 (紫色系) - 标准(128, 64, 128)
            # 只要 R在118-138, G在54-74, B在118-138 之间，都算路！
            is_road = (mask[:, :, 0] > 118) & (mask[:, :, 0] < 138) & \
                      (mask[:, :, 1] > 54) & (mask[:, :, 1] < 74) & \
                      (mask[:, :, 2] > 118) & (mask[:, :, 2] < 138)
            target_mask[is_road] = 1

            # 2. 人 (红色系) - 标准(220, 20, 60)
            is_person = (mask[:, :, 0] > 200) & \
                        (mask[:, :, 1] < 50) & \
                        (mask[:, :, 2] < 100)
            target_mask[is_person] = 2

            # 3. 车 (深红/深蓝系)
            # 你的数据集车是深红 (142, 0, 0)
            is_car_red = (mask[:, :, 0] > 130) & (mask[:, :, 0] < 160) & \
                         (mask[:, :, 1] < 30) & \
                         (mask[:, :, 2] < 30)
            # 兼容标准版的深蓝 (0, 0, 142)
            is_car_blue = (mask[:, :, 2] > 130) & (mask[:, :, 2] < 160) & \
                          (mask[:, :, 0] < 30) & \
                          (mask[:, :, 1] < 30)

            target_mask[is_car_red | is_car_blue] = 3

        # 3. 增强与预处理
        if self.augmentation:
            sample = self.augmentation(image=image, mask=target_mask)
            image, target_mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=target_mask)
            image, target_mask = sample['image'], sample['mask']

        return image, target_mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.RandomBrightnessContrast(p=0.2),
        albu.ShiftScaleRotate(scale_limit=0.05, rotate_limit=5, shift_limit=0.05, p=0.5, border_mode=0),
    ])


# ✅ 替换成这个（定义一个可序列化的类）
class PreprocessingTransform:
    def __init__(self, preprocessing_fn):
        self.preprocessing_fn = preprocessing_fn

    def __call__(self, image, mask):
        # 1. 图像预处理 (归一化)
        image = self.preprocessing_fn(image)
        # 2. 转置 (H,W,C) -> (C,H,W)
        image = image.transpose(2, 0, 1).astype('float32')
        # 3. 标签转 LongTensor
        mask = torch.from_numpy(mask).long()
        return {"image": image, "mask": mask}

def get_preprocessing(preprocessing_fn):
    return PreprocessingTransform(preprocessing_fn)


# ================= 4. 主程序 =================
if __name__ == '__main__':
    print("========== 🚀 启动抗干扰混合训练 (适配一切颜色偏差) ==========")

    prep_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # 加载数据集
    ds_camvid = SegmentationDataset(
        os.path.join(CAMVID_DIR, 'train'), os.path.join(CAMVID_DIR, 'train_labels'),
        source_type='camvid', augmentation=get_training_augmentation(), preprocessing=get_preprocessing(prep_fn)
    )

    full_dataset = ds_camvid
    if os.path.exists(CITY_DIR):
        print(f"检测到 Cityscapes，正在合并...")
        ds_city = SegmentationDataset(
            os.path.join(CITY_DIR, 'train'), os.path.join(CITY_DIR, 'train_labels'),
            source_type='cityscapes', augmentation=get_training_augmentation(), preprocessing=get_preprocessing(prep_fn)
        )
        full_dataset = ConcatDataset([ds_camvid, ds_city])
        print(f"✅ 合并成功！")

    print(f"总图片数: {len(full_dataset)}")
    # 开启 pin_memory 加速，显存允许可开 num_workers=2
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 模型与优化器
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🔄 加载存档: {MODEL_SAVE_PATH}")
        model = torch.load(MODEL_SAVE_PATH, weights_only=False)
    else:
        print("✨ 创建新模型...")
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
            classes=4, activation=None
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

            if i % 50 == 0:
                print(f"  Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch + 1} 结束 | Avg Loss: {avg_loss:.4f}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model, MODEL_SAVE_PATH)
            print(f"  💾 最佳模型已保存! (Loss: {min_loss:.4f})")