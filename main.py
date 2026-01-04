import torch

print("========== 显卡激活测试 ==========")

# 1. 看看是不是 GPU 版本
print(f"PyTorch 版本: {torch.__version__}")

# 2. 关键时刻：显卡能不能用？
if torch.cuda.is_available():
    print("✅ 成功！显卡已激活！(NVIDIA CUDA is ready)")
    print(f"🚀 当前显卡型号: {torch.cuda.get_device_name(0)}")

    # 测试一下显存
    mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"💾 显存大小: {mem:.2f} GB")

    # 做个 GPU 运算测试
    a = torch.rand(1000, 1000).to('cuda')
    b = torch.rand(1000, 1000).to('cuda')
    print("⚡ GPU 计算测试通过！")
else:
    print("❌ 失败... 当前依然是 CPU 模式")
    print("请检查是否安装了正确的 CUDA 版本 PyTorch")

print("==================================")