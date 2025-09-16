import math
import torch
import torch.nn as nn
from timm.layers import Mlp
from timm.layers import DropPath

# from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
# from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
# from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
# from diffusion_planner.model.module.mixer import MixerBlock
from AAAI2025.DPR.model.dit import TimestepEmbedder, DiTBlock, FinalLayer


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
#--------------------------Predictor-------------------------------#

class GoalPredictor(nn.Module):
    '''The behavior predictor decoder '''
    def __init__(self, future_len=80, action_len=5, agents_len=32):
        super().__init__()
        self._agents_len = agents_len  # 最大的代理数量
        self._future_len = future_len  # 未来时间步长
        self._action_len = action_len  # 每个动作的时间步长

        # 定义四个交叉注意力层
        self.attention_layers = nn.ModuleList([CrossTransformer() for _ in range(4)])
        # 锚点编码器，用于将锚点坐标编码为高维特征
        self.anchor_encoder = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 256))
        # 动作解码器，用于解码动作序列
        self.act_decoder = nn.Sequential(nn.Linear(256, 256), nn.ELU(), nn.Dropout(0.1),
                                         nn.Linear(256, (self._future_len)*2))
        # 分数解码器，用于解码每个锚点的得分
        self.score_decoder = nn.Sequential(nn.Linear(256, 128), nn.ELU(), nn.Dropout(0.1),
                                           nn.Linear(128, 1))

    def forward(self, inputs):
        # 获取锚点坐标并对其进行编码
        anchors_points = inputs['anchors'][:, :self._agents_len]  # 维度为[B,32,64,2]
        anchors = self.anchor_encoder(anchors_points)  #  维度 [B, 32, 64, 256]
        encodings = inputs['encodings']  # 维度是：[B, 336, 256]
        # 将锚点特征与编码特征相加，生成查询
        query = encodings[:, :self._agents_len, None] + anchors  # [B, 32, 64, 256]

        # 获取批量大小、代理数量、查询数量
        num_batch, num_agents, num_queries, _ = query.shape

        # 合并代理、地图和交通灯的掩码
        mask = torch.cat([inputs['agents_mask'], inputs['maps_mask'], 
                          inputs['traffic_lights_mask']], dim=-1)  # 维度是 [B, 336]
        relations = inputs['relation_encodings']  # 维度是 (B, 336, 336, 256)

        actions = []  # 用于存储动作序列
        scores = []  # 用于存储锚点得分
        for i in range(self._agents_len):
            # 第一个交叉注意力层 query[:,i]的维度是[B, 64, 256]
            #  qruery_content的维度为 [B, 64, 256]
            query_content = self.attention_layers[0](query[:, i], encodings, relations[:, i], key_mask=mask)
            # 第二个交叉注意力层
            query_content = self.attention_layers[1](query_content, encodings, relations[:, i], key_mask=mask)
            # 添加残差连接
            query_content = query_content + query[:, i]
            # 第三个交叉注意力层
            query_content = self.attention_layers[2](query_content, encodings, relations[:, i], key_mask=mask)
            # 第四个交叉注意力层
            query_content = self.attention_layers[3](query_content, encodings, relations[:, i], key_mask=mask)
            # 解码动作序列 维度为 [B, 64, 16*2]
            actions.append(self.act_decoder(query_content).reshape(
                num_batch, num_queries, self._future_len, 2
            ))  # [B, Q, T, 2]
            # 解码锚点得分 维度为 [B, 64, 1]
            scores.append(self.score_decoder(query_content).squeeze(-1))  # [B, Q]

        # 将动作序列和得分堆叠
        actions = torch.stack(actions, dim=1)  # [B, 32, 64, 16*2]
        scores = torch.stack(scores, dim=1)  # [B, 32,64，1]

        return actions, scores

    def reset_agent_length(self, agents_len):
        """
        重置代理数量
        Args:
            agents_len: 新的代理数量
        """
        self._agents_len = agents_len

class CrossTransformer(nn.Module):
    '''Decoder layer block.'''
    def __init__(self):
        super().__init__()
        heads, dim, dropout = 8, 256, 0.1
        # 定义多头交叉注意力层，输入维度256，8个头，dropout为0.1，batch_first=True
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        # 第一层归一化
        self.norm_1 = nn.LayerNorm(dim)
        # 第二层归一化
        self.norm_2 = nn.LayerNorm(dim)
        # 前馈神经网络，包含两层线性层和GELU激活，dropout
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(dim*4, dim), 
            nn.Dropout(dropout)
        )

    def forward(self, query, key, relations, attn_mask=None, key_mask=None):
        """
        前向传播
        Args:
            query: 查询张量，[B, Q, 256]
            key: 键张量，[B, K, 256]
            relations: 关系编码，[B, K, 256]，与key相加
            attn_mask: 注意力掩码（可选），[Q, K] 或 [B*heads, Q, K]
            key_mask: 键的padding掩码（可选），[B, K]
        Returns:
            output: 输出张量，[B, Q, 256]
        """
        # 将关系编码加到key和value上
        key = key + relations
        value = key

        # 如果提供了key_mask，则用于屏蔽无效的key
        if key_mask is not None:
            attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=key_mask)
        # 如果提供了attn_mask，则用于限制注意力范围
        elif attn_mask is not None:
            attention_output, _ = self.cross_attention(query, key, value, attn_mask=attn_mask)
        # 否则不使用掩码
        else:
            attention_output, _ = self.cross_attention(query, key, value)

        # 第一层归一化
        attention_output = self.norm_1(attention_output)
        # 前馈网络+残差连接+第二层归一化
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output

# --------------------------Decoder-------------------------------#
 
class Decoder(nn.Module):
    def __init__(self, decoder_drop_path_rate=0.1, action_len=2, predicted_agents_num=32, future_len=40, hidden_dim=256, decoder_depth=3, num_heads=8):
        super().__init__()

        # 初始化解码器的配置参数
        dpr = decoder_drop_path_rate  # 0.1
        self._predicted_agents_num = predicted_agents_num  # 预测的邻居数量  32
        self._future_len = future_len  # 未来时间步长 40
        self._action_len = action_len  # 每个动作的时间步长 5
        
        # 初始化DiT模型
        self.dit = DiT(
            route_encoder=AnchorTrajectoryFusionEncoder(d_model=hidden_dim),
            depth=decoder_depth,  #  3
            output_dim= future_len * 2,  # 输出维度为 (未来时间步长 + 1) * 4 (x, y, cos, sin)
            hidden_dim=hidden_dim,  # 256
            heads=num_heads,  # 6
            dropout=dpr,  # 0.1
        )
    
    
    def forward(self, encoder_outputs, noisy_actions, diffusion_step, agents_future):

        diffusion_time = diffusion_step[:, -1] # 维度为[B,]
        ego_neighbor_encoding = encoder_outputs['encodings'] # 维度是 [B, 336, 256]
        Anchors = encoder_outputs['anchors']  # [B, 32, 64, 2]
        future_traj = agents_future  # [B, 32, 41, 5]
        # agents_current = agents_future[:, :, 0, :2]  # [B, 32, 2]，当前智能体的状态
        current_mask =encoder_outputs['agents_mask'][:, :self._predicted_agents_num]  # [B, 32]，邻居车辆的有效性掩码
        # current_mask  = torch.sum(torch.ne(agents_current[..., :2], 0), dim=-1) == 0 
  
        B, P, _ , _= future_traj.shape
        assert P == self._predicted_agents_num  # 检测当前状态的数量是否正确
        
        sampled_actions_trajectories = noisy_actions.reshape(B, P, -1) # 维度为 [B, 32, 40*2]，将动作序列展平为 [B, 32, 80*2] 

        return self.dit(
                    sampled_actions_trajectories,  # 采样的当前-未来自车和邻居状态 [B, 32, 40*2]
                    diffusion_time, # 维度为[B,]
                    ego_neighbor_encoding, # [B, 336, 256]
                    Anchors,   # [B, 32, 64, 2]
                    future_traj,  # [B, 32, 41, 2]，未来轨迹序列
                    current_mask  # [B, 32] (邻居车辆的有效性掩码)
                ).reshape(B, P, -1, 2)  # 维度为 [B, 32, 41, 2]


class AnchorTrajectoryFusionEncoder(nn.Module):
    '''输入的维度;
              anchors:   [B, 32, 64, 2]，锚点序列
              fut_traj:  [B, 32, 41, 2]，未来轨迹序列
       输出的维度：
              global_feat: [B, D]，全局导航特征'''
    def __init__(self, d_model=256):


        super().__init__()
        self.d_model = d_model

        # 编码锚点序列的 MLP（64个锚点）
        self.anchor_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, d_model)
        )

        # 编码未来轨迹序列的 MLP（80步未来）
        self.traj_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, d_model)
        )

        # 融合后的 MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, anchors, fut_traj):
        """
        anchors:   [B, 32, 64, 2]
        fut_traj:  [B, 32, 41, 5]
        return:    [B, D]
        """
        B, N, _, _ = anchors.shape

        # (1) 编码锚点
        anchor_feat = self.anchor_mlp(anchors)         # → [B, 32, 64, D]
        anchor_feat = anchor_feat.mean(dim=2)          # → [B, 32, D]

        # (2) 编码未来轨迹
        traj_feat = fut_traj[:,:,:,:2]            # → [B, 32, 40, D]
        traj_feat = self.traj_mlp(traj_feat)            # → [B, 32, 41, D]
        traj_feat = traj_feat.mean(dim=2)              # → [B, 32, D]

        # (3) 融合每个智能体的特征
        fused = torch.cat([anchor_feat, traj_feat], dim=-1)  # → [B, 32, 2D]
        fused = self.fusion_mlp(fused)                       # → [B, 32, D]

        # (4) 聚合所有智能体，得到全局导航特征
        global_feat = fused.mean(dim=1)                # → [B, D]

        return global_feat

class DiT(nn.Module):
    def __init__(self, route_encoder: nn.Module, depth, output_dim, hidden_dim=256, heads=6, dropout=0.1, mlp_ratio=4.0):

        '''
        初始化的参数：
                route_encoder: 导航信息编码器
                depth: 几层DiTBlock
                output_dim: 输出维度
                hidden_dim: 隐藏层维度
                heads: 多头注意力的头数 
                dropout: dropout概率
                mlp_ratio: MLP隐藏层和dim的比率
        '''
        super().__init__()
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        # 引入的解码decoder中代码块
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
               
    def forward(self, x, t, cross_c, Anchors, fut_traj, current_mask):
        '''           
            # print("sampled_trajectories shape:", sampled_trajectories.shape)   [B, 32, 41*2]
            # print("diffusion_time shape:", diffusion_time.shape)               [B,]
            # print("ego_neighbor_encoding shape:", ego_neighbor_encoding.shape) [B, 336, 256]
            # print("Anchors shape:", Anchors.shape)                             [B, 32, 64, 2]
            # print("future_traj shape:", future_traj.shape)                     [B, 32, 41, 5]
            # print("current_mask shape:", current_mask.shape)                   [B, 32]'''
        '''输入的维度：
        self.dit(
                    sampled_trajectories,  # 采样的当前-未来自车和邻居状态 [B, 32, 41*2]
                    diffusion_time, # 维度为[B,]
                    ego_neighbor_encoding, # [B, 336, 256]
                    Anchors, # [B, 32, 64, 2]
                    fut_traj:  [B, 32, 40, 2]，未来轨迹序列
                    neighbor_current_mask  # [B, 32] (邻居车辆的有效性掩码)
                )'''

        B, P, _ = x.shape # [B, 32, 40*2]

        x = self.preproj(x) # 维度为[B, 32, 256]

        navigation_encoding = self.route_encoder(Anchors, fut_traj) # 维度是（B, 256）
        y = navigation_encoding
        y = y + self.t_embedder(t)  # # 维度是（B, 256）

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask = current_mask 
        
        for block in self.blocks:
            '''
        输入的维度：
              x: 主序列输入张量，形状为 (B, 32， 256)
              cross_c: 交叉注意力条件张量，形状为 (B, 336, 256)
              y: 条件张量，形状为 (B, 256)
              attn_mask: 注意力掩码，形状为 (B, 32)
        输出:
              x: 形状为 (B, 32, 256)
            '''
            x = block(x, cross_c, y, attn_mask)   # 维度是 [B, 32, 256]
            
        x = self.final_layer(x, y) 
        
        return x
