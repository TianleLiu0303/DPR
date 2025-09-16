import torch
import torch.nn as nn

# 假设你在 decoder.py 中定义了 Decoder、DiT、AnchorTrajectoryFusionEncoder
from AAAI2025.DPR.model.Decoder import Decoder  

# 配置类（模拟你的配置）
class Config:
    def __init__(self):
        self.decoder_drop_path_rate = 0.1
        self.predicted_neighbor_num = 32
        self.future_len = 40
        self.hidden_dim = 256
        self.decoder_depth = 3
        self.num_heads = 8
        self.diffusion_model_type = "x_start"  # or "score"

# 构造模拟输入
def generate_mock_inputs(config, B=2):  # B: batch size
    P = config.predicted_neighbor_num
    T = config.future_len

    encoder_outputs = {
        'encodings': torch.randn(B, 336, config.hidden_dim),  # 例如：336 = 32*10 + 16
        'anchors': torch.randn(B, P, 64, 2)
    }

    noisy_actions = torch.randn(B, P, T, 2)  # 每个agent预测 (40+1)*2 个值 (x,y)
    diffusion_step = torch.randint(low=0, high=1000, size=(B, P))  # 只使用最后一列
    agents_future = torch.randn(B, P, T + 1, 5)  # 模拟未来轨迹 [B, 32, 41, 5]

    return encoder_outputs, noisy_actions, diffusion_step, agents_future

def main():
    config = Config()
    decoder = Decoder()
    decoder.train()  # 设置为训练模式

    # 生成模拟输入
    encoder_outputs, noisy_actions, diffusion_step, agents_future = generate_mock_inputs(config)
    # print(encoder_outputs['encodings'].shape)
    # print(encoder_outputs['anchors'].shape)
    # print(noisy_actions.shape)
    # print(diffusion_step.shape)
    # print(agents_future.shape)
    # 前向传播
    with torch.no_grad():
        output = decoder(
            encoder_outputs=encoder_outputs,
            noisy_actions=noisy_actions,
            diffusion_step=diffusion_step,
            agents_future=agents_future
        )

    # 输出检查
    print("Decoder 输出：")
    for k, v in output.items():
        print(f"{k}: shape = {v.shape}")

if __name__ == '__main__':
    main()
