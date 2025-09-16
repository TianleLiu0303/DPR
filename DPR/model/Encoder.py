import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from .model_utils import (batch_transform_trajs_to_local_frame,
                         batch_transform_polylines_to_local_frame,
                         batch_transform_trajs_to_global_frame,
                         roll_out)

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
class Encoder(nn.Module):
    def __init__(self, layers=6, version='v1'):
        super().__init__()
        self._version = version
        if self._version == 'v1':
            self.agent_encoder = AgentEncoder()
        else:
            self.agent_encoder = AgentEncoderV2()
        self.map_encoder = MapEncoder()
        self.traffic_light_encoder = TrafficLightEncoder()
        self.relation_encoder = FourierEmbedding(input_dim=3) # 编码相关性
        # 最后的融合代码块
        self.transformer_encoder = TransformerEncoder(layers=layers)

    def forward(self, inputs):
        # agents encoding
        agents = inputs['agents_history'] # 维度是 (B,64,11,8)
        agents_type = inputs['agents_type'] # 维度是 (B,64)
        agents_interested = inputs['agents_interested'] # 维度是 (B,64)
        agents_local = batch_transform_trajs_to_local_frame(agents) # 维度是 (B,64,11,8)
 
        B, A, T, D = agents_local.shape # B是批量大小，64，11，8
        agents_local = agents_local.reshape(B*A, T, D) # 将维度变为 (B*A, T, D)
        agents_type = agents_type.reshape(B*A) # 将维度变为 (B*A)
        encoded_agents = self.agent_encoder(agents_local, agents_type)  # 维度是 (B*A, 256)
        encoded_agents = encoded_agents.reshape(B, A, -1) # 维度是 (B, A, 256)
        agents_mask = torch.eq(agents_interested, 0) # （B，64）的维度

        # map and traffic light encoding
        map_polylines = inputs['polylines']  # 维度是 (B, 256, 30, 5)
        map_polylines_local = batch_transform_polylines_to_local_frame(map_polylines) # 维度是 (B, 256, 30, 5)
        encoded_map_lanes = self.map_encoder(map_polylines_local)  # 维度是 (B, 256, 256)
        maps_mask = inputs['polylines_valid'].logical_not() # 逻辑取反, 维度为 (B, 256)

        traffic_lights = inputs['traffic_light_points'] # 维度是 (B, 16, 3)
        encoded_traffic_lights = self.traffic_light_encoder(traffic_lights) # 维度是 (B, 16, 256)
        traffic_lights_mask = torch.eq(traffic_lights.sum(-1), 0) # 维度是 (B, 16)

        # relation encoding
        relations = inputs['relations'] # 维度是 (B，336, 336, 3)
        relations = self.relation_encoder(relations)  # 维度是 (B, 336, 336, 256)
        
        # transformer encoding
        encoder_outputs = {}
        encoder_outputs['agents'] = agents
        encoder_outputs['anchors'] = inputs['anchors'][:,:32,:,:] # 维度为 (B, 64, 64， 2)
        encoder_outputs['agents_type'] = agents_type
        encoder_outputs['agents_mask'] = agents_mask
        encoder_outputs['maps_mask'] = maps_mask
        encoder_outputs['traffic_lights_mask'] = traffic_lights_mask
        encoder_outputs['relation_encodings'] = relations # 维度是 (B, 336, 336, 256)
        
        encodings = self.transformer_encoder(relations, encoded_agents, encoded_map_lanes, encoded_traffic_lights, agents_mask, maps_mask, traffic_lights_mask)
        encoder_outputs['encodings'] = encodings # 维度是 (B, 336, 256)

        return encoder_outputs # 把上述编码的内容都放在一个字典中返回

#-----------------------------Encoder utils-----------------------------------#
class AgentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个GRU层，用于对历史轨迹进行编码
        # 输入维度为8，隐藏层维度为256，层数为2，使用batch_first=True，添加0.2的dropout
        self.motion = nn.GRU(8, 256, 2, batch_first=True, dropout=0.2)
        # 定义一个嵌入层，用于对代理类型进行嵌入
        # 输入类别数为4，嵌入维度为256，padding_idx=0表示类别0为填充类别
        self.type_embed = nn.Embedding(4, 256, padding_idx=0)

    def forward(self, history, type):
        """
        Args:
            history: [B, T, 8]，历史轨迹，B为批量大小，T为时间步长，8为特征维度
            type: [B]，代理类型，B为批量大小
        Returns:
            output: [B, 256]，编码后的特征，B为批量大小，256为隐藏层维度
        """
        # 使用GRU对历史轨迹进行编码，返回所有时间步的隐藏状态traj和最后一个隐藏状态_
        traj, _ = self.motion(history)
        # 取最后一个时间步的隐藏状态作为当前帧的特征
        output = traj[:, -1]
        # 对代理类型进行嵌入
        type_embed = self.type_embed(type)
        # 将轨迹特征和类型嵌入相加
        output = output + type_embed

        return output
    
class AgentEncoderV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.type_embed = nn.Embedding(4, 256, padding_idx=0)
        
        
    def forward(self, history, type):
        cur = history[:, -1, 3:] # only take [vel_x, vel_y, length, width, height]
        output = self.motion(cur)
        type_embed = self.type_embed(type)
        output = output + type_embed

        return output
    
class MapEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 256))
        self.traffic_light_embed = nn.Embedding(8, 256)
        self.type_embed = nn.Embedding(21, 256, padding_idx=0)

    def forward(self, inputs):
        # inputs [B, M, W, 5]
        output = self.point(inputs[..., :3]) 
        output = torch.max(output, dim=-2).values # max pooling on W

        traffic_light_type = inputs[:, :, 0, 3].long().clamp(0, 7)
        traffic_light_embed = self.traffic_light_embed(traffic_light_type)
        polyline_type = inputs[:, :, 0, 4].long().clamp(0, 20)
        type_embed = self.type_embed(polyline_type)
        output = output + traffic_light_embed + type_embed

        return output

class TrafficLightEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_embed = nn.Embedding(8, 256)

    def forward(self, inputs):
        # inputs [B, TL, 3]
        traffic_light_type = inputs[:, :, 2].long().clamp(0, 7)
        type_embed = self.type_embed(traffic_light_type)
        output = type_embed

        return output

class FourierEmbedding(nn.Module):
    '''编码相关性'''
    def __init__(self, input_dim, hidden_dim=256, num_freq_bands=64):
        super().__init__()
        # 输入维度
        self.input_dim = input_dim
        # 隐藏层维度
        self.hidden_dim = hidden_dim

        # 频率嵌入层，如果输入维度不为0，则创建一个嵌入层
        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None

        # 为每个输入维度创建一个多层感知机（MLP）
        self.mlps = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_freq_bands * 2 + 1, hidden_dim),  # 输入为频率嵌入后的维度
                nn.LayerNorm(hidden_dim),  # 层归一化
                nn.ReLU(inplace=True),  # 激活函数
                nn.Linear(hidden_dim, hidden_dim),  # 输出为隐藏层维度
            ) for _ in range(input_dim)]
        )
        
        # 输出层，进一步处理隐藏层输出
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),  # 层归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(hidden_dim, hidden_dim),  # 输出为隐藏层维度
        )

    def forward(self, continuous_inputs):
        """
        Args:
            continuous_inputs: [B, N, input_dim]，连续输入张量
        Returns:
            输出: [B, N, hidden_dim]，经过嵌入后的特征
        """
        # 将输入与频率嵌入权重相乘，并乘以 2π
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        # 计算 cos 和 sin，并将原始输入拼接到最后一维
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        # 对每个输入维度应用对应的 MLP，并将结果相加
        x = torch.stack([self.mlps[i](x[:, :, :, i]) for i in range(self.input_dim)]).sum(dim=0)
        # 通过输出层进一步处理
        return self.to_out(x)


#-----------------------------最后的融合模块-----------------------------------#
class QCMHA(nn.Module):
    """
    Quadratic Complexity Multi-Head Attention module.
    
    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Default is 0.1.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3*embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj.weight)
        xavier_uniform_(self.out_proj.weight)
        constant_(self.in_proj.bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, query, rel_pos, attn_mask=None):
        '''
        Forward pass of the QCMHA module.
        
        Args:
            query (torch.Tensor): The input query tensor of shape [batch_size, query_length, embed_dim].
            rel_pos (torch.Tensor): The relative position tensor of shape [batch_size, query_length, key_length, embed_dim].
            attn_mask (torch.Tensor, optional): The attention mask tensor of shape [batch_size, query_length, key_length].
        
        Returns:
            torch.Tensor: The output tensor of shape [batch_size, query_length, embed_dim].
        '''
        query = self.in_proj(query)
        b, t, d = query.shape
        query = query.reshape(b, t, self.num_heads, self.head_dim*3)

        res = torch.split(query, self.head_dim, dim=-1)
        q, k, v = res
    
        rel_pos_q = rel_pos_v = rel_pos

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        
        dot_score = torch.matmul(q, k)
    
        if rel_pos is not None:
            rel_pos_q = rel_pos_q.reshape(b, t, t, self.num_heads, self.head_dim)
            rel_pos_q = rel_pos_q.permute(0, 3, 1, 4, 2) #[b, h, q, d, k]
            #[b, h, q, 1, d] * [b, h, q, d, k] -> [b, h, q, 1, k] 
            dot_score_rel = torch.matmul(q.unsqueeze(-2), rel_pos_q).squeeze(-2)
            dot_score += dot_score_rel

        dot_score = dot_score / np.sqrt(self.head_dim)

        if attn_mask is not None:
            dot_score = dot_score - attn_mask.float() * 1e9

        dot_score = F.softmax(dot_score, dim=-1)
        dot_score = self.dropout(dot_score)

        value = torch.matmul(dot_score, v)

        if rel_pos is not None:
            rel_pos_v = rel_pos_v.reshape(b, t, t, self.num_heads, self.head_dim)
            rel_pos_v = rel_pos_v.permute(0, 3, 1, 2, 4) #[b, h, q, k, d]
            # [b, h, q, 1, k] * [b, h, q, k, d] -> [b, h, q, d]
            value_rel = torch.matmul(dot_score.unsqueeze(-2), rel_pos_v).squeeze(-2)
            value += value_rel

        value = value.permute(0, 2, 1, 3) #[b, t, h, d//h]
        value = value.reshape(b, t, self.embed_dim)
        value = self.out_proj(value)

        return value

# 此函数调用QCMHA
class SelfTransformer(nn.Module):
    """Encoder layer block.
    输入的维度：
              inputs: (B, 336, 256)
              relations: (B, 336, 336, 256)
              mask: (B, 336, 336)"""
    def __init__(self):
        super().__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.qc_attention = QCMHA(dim, heads, dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), 
                                 nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, inputs, relations, mask=None):
        attention_output = self.qc_attention(inputs, relations, mask) # [B, 336, 256]
        attention_output = self.norm_1(attention_output + inputs) # [B, 336, 256]
        output = self.norm_2(self.ffn(attention_output) + attention_output) # [B, 336, 256]

        return output # [B, 336, 256]

# 此函数调用SelfTransformer
class TransformerEncoder(nn.Module):
    def __init__(self, layers=6):
        super().__init__()
        # 定义多个自注意力层（SelfTransformer），数量由layers参数指定
        self.layers = nn.ModuleList([SelfTransformer() for _ in range(layers)])

    def forward(self, encoded_relations, encoded_trajs, encoded_polylines, encoded_traffic_lights,
                trajs_mask, polylines_mask, traffic_lights_mask):
        """
        前向传播
        Args:
            encoded_relations: [B, N+M+TL, N+M+TL, 256]，关系编码特征
            encoded_trajs: [B, N, 256]，代理轨迹编码特征
            encoded_polylines: [B, M, 256]，地图多段线编码特征
            encoded_traffic_lights: [B, TL, 256]，交通灯编码特征
            trajs_mask: [B, N]，代理掩码
            polylines_mask: [B, M]，多段线掩码
            traffic_lights_mask: [B, TL]，交通灯掩码
        Returns:
            encodings: [B, N+M+TL, 256]，融合后的编码特征
        """
        # 将代理、地图多段线和交通灯的编码特征在第1维拼接，得到总特征 [B, N+M+TL（336）, 256]
        encodings = torch.cat([encoded_trajs, encoded_polylines, encoded_traffic_lights], dim=1)
        # 拼接所有掩码，得到总掩码 [B, N+M+TL（336）]
        encodings_mask = torch.cat([trajs_mask, polylines_mask, traffic_lights_mask], dim=-1)
        # 构造注意力掩码 [B, N+M+TL, N+M+TL]，用于屏蔽无效元素
        attention_mask = encodings_mask.unsqueeze(-1).repeat(1, 1, encodings_mask.shape[1])
        # 增加一维以适配多头注意力的输入 [B, 1, N+M+TL, N+M+TL]
        attention_mask = attention_mask.unsqueeze(1)

        # 依次通过每一层自注意力层
        for layer in self.layers:
            # 每一层输入当前编码特征、关系特征和注意力掩码，输出更新后的编码特征
            encodings = layer(encodings, encoded_relations, attention_mask)

        # 返回最终融合后的编码特征
        return encodings

