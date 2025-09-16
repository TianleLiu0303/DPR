import torch
from torch import nn

# 导入你的 GoalPredictor 类（确保路径正确）
from AAAI2025.DPR.model.Decoder import GoalPredictor  

def main():
    # 构造输入字典
    inputs = {
        'agents': torch.randn(2, 64, 11, 8),
        'anchors': torch.randn(2, 64, 64, 2),
        'agents_type': torch.randint(0, 10, (128,)),
        'agents_mask': torch.ones(2, 64),
        'maps_mask': torch.ones(2, 256),
        'traffic_lights_mask': torch.ones(2, 16),
        'relation_encodings': torch.randn(2, 336, 336, 256),
        'encodings': torch.randn(2, 336, 256)
    }

    # 初始化模型
    model = GoalPredictor(future_len=80, action_len=2, agents_len=32)

    # 前向传播
    actions, scores = model(inputs)

    # 打印结果维度
    print("actions shape:", actions.shape)  # 期望：[2, 32, 64, 40, 2]
    print("scores shape:", scores.shape)    # 期望：[2, 32, 64]

if __name__ == "__main__":
    main()
