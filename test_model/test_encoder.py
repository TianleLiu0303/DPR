import torch
import numpy as np
from AAAI2025.DPR.model.Encoder import Encoder

def test_encoder_forward_shapes():
    # 构造输入数据
    batch_size = 2
    num_agents = 64
    num_future = 81
    num_type = 4
    num_traffic_lights = 16
    num_polylines = 256
    polyline_width = 30
    polyline_feat = 5
    relation_dim = 3
    relation_total = 336
    anchor_num = 64

    # 构造输入字典
    '''
输入的数据格式：
agents_history: torch.float32, torch.Size([2, 64, 11, 8])
           agents_interested: torch.int32, torch.Size([2, 64])
           agents_future: torch.float32, torch.Size([2, 64, 81, 5])
           agents_type: torch.int32, torch.Size([2, 64])
           traffic_light_points: torch.float32, torch.Size([2, 16, 3])
           polylines: torch.float32, torch.Size([2, 256, 30, 5])
           polylines_valid: torch.int32, torch.Size([2, 256])
           relations: torch.float32, torch.Size([2, 336, 336, 3])
           anchors: torch.float32, torch.Size([2, 64, 64, 2])
'''
    inputs = {
        'agents_history': torch.randn(batch_size, num_agents, 11, 8, dtype=torch.float32),
        'agents_interested': torch.randint(0, 2, (batch_size, num_agents), dtype=torch.int32),
        'agents_future': torch.randn(batch_size, num_agents, num_future, 5, dtype=torch.float32),
        'agents_type': torch.randint(0, num_type, (batch_size, num_agents), dtype=torch.int32),
        'traffic_light_points': torch.randn(batch_size, num_traffic_lights, 3, dtype=torch.float32),
        'polylines': torch.randn(batch_size, num_polylines, polyline_width, polyline_feat, dtype=torch.float32),
        'polylines_valid': torch.randint(0, 2, (batch_size, num_polylines), dtype=torch.int32),
        'relations': torch.randn(batch_size, relation_total, relation_total, relation_dim, dtype=torch.float32),
        'anchors': torch.randn(batch_size, anchor_num, anchor_num, 2, dtype=torch.float32),
    }

    # 转换int32为torch.bool或torch.long以适配模型
    inputs['agents_interested'] = inputs['agents_interested'].to(torch.long)
    inputs['agents_type'] = inputs['agents_type'].to(torch.long)
    inputs['polylines_valid'] = inputs['polylines_valid'].to(torch.bool)

    # 实例化Encoder
    encoder = Encoder()
    encoder.eval()

    # 前向传播
    with torch.no_grad():
        outputs = encoder(inputs)

    # 检查输出字典的关键字段和shape
    assert isinstance(outputs, dict)
    assert 'agents' in outputs
    assert 'anchors' in outputs
    assert 'agents_type' in outputs
    assert 'agents_mask' in outputs
    assert 'maps_mask' in outputs
    assert 'traffic_lights_mask' in outputs
    assert 'relation_encodings' in outputs
    assert 'encodings' in outputs

    # 检查shape
    print("outputs['agents'].shape:", outputs['agents'].shape)
    print("outputs['anchors'].shape:", outputs['anchors'].shape)
    print("outputs['agents_type'].shape:", outputs['agents_type'].shape)
    print("outputs['agents_mask'].shape:", outputs['agents_mask'].shape)
    print("outputs['maps_mask'].shape:", outputs['maps_mask'].shape)
    print("outputs['traffic_lights_mask'].shape:", outputs['traffic_lights_mask'].shape)
    print("outputs['relation_encodings'].shape:", outputs['relation_encodings'].shape)
    print("outputs['encodings'].shape:", outputs['encodings'].shape)

if __name__ == "__main__":
    test_encoder_forward_shapes()
    print("Encoder forward test passed.")