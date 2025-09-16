import torch
import lightning.pytorch as pl
from AAAI2025.DPR.model.Encoder import Encoder
from AAAI2025.DPR.model.Decoder import Decoder, GoalPredictor
from AAAI2025.DPR.model.utils import DDPM_Sampler
from AAAI2025.DPR.model.model_utils import inverse_kinematics, roll_out, batch_transform_trajs_to_global_frame
from torch.nn.functional import smooth_l1_loss, cross_entropy
import torch.nn.functional as F
import copy

class DPR_GRPO(pl.LightningModule):
    """
    RL-fine tuning of DPR model with GRPO
    """

    def __init__(self, cfg: dict):
        """
        Initialize the DPR_GRPO model.
        """
        # 调用父类的初始化方法
        super().__init__()
        self.save_hyperparameters()
        
        # 配置参数字典
        self.cfg = cfg
        # 未来时间步的长度
        self._future_len = cfg['future_len']
        # 考虑的代理数量
        self._agents_len = cfg['agents_len']
        # 动作序列的长度
        self._action_len = cfg['action_len']
        # 扩散步骤的数量 50步数
        self._diffusion_steps = cfg['diffusion_steps']
        # 编码器的层数
        self._encoder_layers = cfg['encoder_layers']
        # 编码器的版本，默认为 'v1'
        self._encoder_version = cfg.get('encoder_version', 'v1')
        # 动作的均值，用于归一化
        self._action_mean = cfg['action_mean']
        # 动作的标准差，用于归一化
        self._action_std = cfg['action_std']
        
        self._train_encoder = cfg.get('train_encoder', True)
        self._train_denoiser = cfg.get('train_denoiser', True)
        self._train_predictor = cfg.get('train_predictor', True)
        self._with_predictor = cfg.get('with_predictor', False)
        self._prediction_type = cfg.get('prediction_type', 'sample')
        self._schedule_type = cfg.get('schedule_type', 'cosine')
        self._replay_buffer = cfg.get('replay_buffer', False)
        # 嵌入维度，默认为 5（默认情况下嵌入是加噪轨迹，因此维度为 5）
        self._embeding_dim = cfg.get('embeding_dim', 5)
        # 后期微调从这里开始，设置编码器的路径
        encoder_path = cfg.get("encoder_ckpt", None)  # 编码器检查点路径
        
        # 编码器
        self.encoder = Encoder(self._encoder_layers, version=self._encoder_version)
      
        if encoder_path is not None:
            model_dict = torch.load(encoder_path, map_location=torch.device("cpu"))["state_dict"]
            for key in list(model_dict.keys()):
                if not key.startswith("encoder."):  # 只保留编码器相关的权重，如果键名不以 "encoder." 开头，则删除对应的权重。
                    del model_dict[key]
            print("Load Encoder Weights")  # 打印加载编码器权重信息
            self.encoder.load_state_dict(model_dict, strict=False)  # 加载权重
        # else:
        #     cfg["train_encoder"] = True
        #     raise Warning("Encoder path is not provided")  # 如果未提供编码器路径，抛出警告
        
        # 解码器
        self.denoiser = Decoder(
           decoder_drop_path_rate=cfg['decoder_drop_path_rate'],
           action_len=cfg['action_len'],
           predicted_agents_num=cfg['predicted_agents_num'],
           future_len=self._future_len,
           hidden_dim=cfg['hidden_dim'],
           decoder_depth=cfg['decoder_depth'],
           num_heads=cfg['num_heads'],       
        )

        self.reference_model = copy.deepcopy(self.denoiser)  # 参考模型，深拷贝解码器
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

        if self._with_predictor:
            self.predictor = GoalPredictor(
                future_len=self._future_len,
                agents_len=self._agents_len,
                action_len=self._action_len,
            )
        else:
            self.predictor = None
            self._train_predictor = False

        self.noise_scheduler = DDPM_Sampler(
            steps=self._diffusion_steps,
            schedule=self._schedule_type,
            s = cfg.get('schedule_s', 0.0),
            e = cfg.get('schedule_e', 1.0),
            tau = cfg.get('schedule_tau', 1.0),
            scale = cfg.get('schedule_scale', 1.0),
        )

        # GRPO
        self.group_size = cfg.get('group_size', 6)  # 每个样本生成的轨迹数量
        self.beta = cfg.get('beta', 0.1)  # KL惩罚系数
                
        self.register_buffer('action_mean', torch.tensor(self._action_mean))  
        self.register_buffer('action_std', torch.tensor(self._action_std))
    
################### Training Setup ###################
    #------------------- Optimizer and Scheduler -------------------
    def configure_optimizers(self):
        '''
        This function is called by Lightning to create the optimizer and learning rate scheduler.
        '''
        if not self._train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not self._train_denoiser: 
            for param in self.denoiser.parameters():
                param.requires_grad = False
        if self._with_predictor and (not self._train_predictor):
            for param in self.predictor.parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)              
        
        assert len(params_to_update) > 0, 'No parameters to update'
        
        optimizer = torch.optim.AdamW(
            params_to_update, 
            lr=self.cfg['lr'],
            weight_decay=self.cfg['weight_decay']
        )
        
        lr_warmpup_step = self.cfg['lr_warmup_step']
        lr_step_freq = self.cfg['lr_step_freq']
        lr_step_gamma = self.cfg['lr_step_gamma']

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma ** n
        
            if lr_scale < 1e-2:
                lr_scale = 1e-2
            elif lr_scale > 1:
                lr_scale = 1
        
            return lr_scale
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step, 
                lr_warmpup_step, 
                lr_step_freq,
                lr_step_gamma,
            )
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
##########################################前向传播函数######################################################
# 把forward_denoiser和forward_predictor的输出融合成一个字典
    def forward(self, inputs, noised_actions_normalized, diffusion_step, agents_future):
        """
        Forward pass of the VBD model.

        Args:
            inputs: Input data.
            noised_actions: noised actions.
            diffusion_step: Diffusion step.

        Returns:
            output_dict: Dictionary（字典化） containing the model outputs.
        """
        # Encode scene
        output_dict = {}
        encoder_outputs = self.encoder(inputs)
        
        if self._train_denoiser:
            denoiser_outputs = self.forward_denoiser(encoder_outputs, noised_actions_normalized, diffusion_step, agents_future)
            output_dict.update(denoiser_outputs)
            
        if self._train_predictor:
            predictor_outputs = self.forward_predictor(encoder_outputs)
            output_dict.update(predictor_outputs)
            
        return output_dict

#---------------------------------Denoiser forward----------------------------------------#

    def forward_denoiser(self, encoder_outputs, noised_actions_normalized, diffusion_step, agents_future):
        """
        Denoiser模块的前向传播。

        Args:
            encoder_outputs: 编码器模块的输出，包含场景编码信息。
            noised_actions_normalized: 加噪后的归一化动作张量。维度为 [B, 32, 40, 2]。
            diffusion_step: 当前扩散步骤。维度为 [B, 32]。

        Returns:
            denoiser_outputs: 包含去噪器输出的字典。
        """
        # 将归一化的加噪动作反归一化，恢复到原始尺度
        noised_actions = self.unnormalize_actions(noised_actions_normalized)
        denoiser_output = self.denoiser(encoder_outputs, noised_actions, diffusion_step, agents_future)
        denoiser_output_old = self.reference_model(encoder_outputs, noised_actions, diffusion_step, agents_future)
        # print('denoiser_output:', denoiser_output["score"].shape)  # [B, 32, 40, 2]
        # denoiser_output= denoiser_output['score']  # 去噪器的动作输出，维度是 [B, 32, 40, 2]
        # denoiser_output_old = denoiser_output_old['score']  # 参考模型的动作输出，维度是 [B, 32, 40, 2]
        # 根据扩散步骤和预测类型，计算初始值
        # 此处会根据预测类类型的不同调整算法

        denoised_actions_normalized = self.noise_scheduler.q_x0(
            denoiser_output, 
            diffusion_step, 
            noised_actions_normalized,
            prediction_type=self._prediction_type
        )
        denoised_actions_normalized_old = self.noise_scheduler.q_x0(
            denoiser_output_old, 
            diffusion_step, 
            noised_actions_normalized,
            prediction_type=self._prediction_type
        )
    
        # 获取当前状态（最后一个时间步的状态）
        current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, '考虑的代理数量过多'
        
        # 将去噪后的归一化动作反归一化，恢复到原始尺度
        denoised_actions = self.unnormalize_actions(denoised_actions_normalized)
        denoised_actions_old = self.unnormalize_actions(denoised_actions_normalized_old)
        # 根据去噪后的动作进行轨迹回滚，生成去噪后的轨迹
        denoised_trajs = roll_out(
            current_states, 
            denoised_actions,
            action_len=self.denoiser._action_len, 
            global_frame=True
        )
        denoised_trajs_old = roll_out(  
            current_states, 
            denoised_actions_old,
            action_len=self.denoiser._action_len, 
            global_frame=True
        )
        
        # 返回包含去噪器输出的字典
        return {
            'denoiser_output': denoiser_output,  # 去噪器的动作输出 维度是 [B, 32, 40, 2]
            'denoised_actions_normalized': denoised_actions_normalized,  # 去噪后的归一化动作
            'denoised_actions': denoised_actions,  # 去噪后反归一化的动作
            'denoised_trajs': denoised_trajs,  # 去噪后的轨迹
            'denoiser_output_old': denoiser_output_old,  # 参考模型的动作输出
            'denoised_actions_normalized_old': denoised_actions_normalized_old,  # 参考模型的去噪后的归一化动作
            'denoised_actions_old': denoised_actions_old,  # 参考模型的去噪后反归一化的动作
            'denoised_trajs_old': denoised_trajs_old,  # 参考模型的去噪后的轨迹
        }
#--------------------------------------Predictor forward------------------------------------------------------------------#
    def forward_predictor(self, encoder_outputs):
        """
        Predictor模块的前向传播。

        Args:
            encoder_outputs: 编码器模块的输出，包含场景编码信息。

        Returns:
            predictor_outputs: 包含预测器输出的字典。
        """
        # 使用预测器预测目标动作的归一化值和目标分数
        goal_actions_normalized, goal_scores = self.predictor(encoder_outputs)
        
        # 获取当前状态（最后一个时间步的状态）
        current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, '考虑的代理数量过多'

        # 将归一化的目标动作反归一化，恢复到原始尺度
        goal_actions = self.unnormalize_actions(goal_actions_normalized)    

        # 根据目标动作生成轨迹
        goal_trajs = roll_out(
            current_states[:, :, None, :],  # 当前状态
            goal_actions,                  # 目标动作
            action_len=self.predictor._action_len,  # 动作长度
            global_frame=True              # 是否在全局坐标系中生成轨迹
        )
        
        # 返回包含预测器输出的字典
        return {
            'goal_actions_normalized': goal_actions_normalized,  # 归一化后的目标动作
            'goal_actions': goal_actions,                        # 目标动作
            'goal_scores': goal_scores,                          # 目标分数
            'goal_trajs': goal_trajs,                            # 目标轨迹
        }

###########################################开始记录训练过程的损失函数#########################################
    #  调用forward函数
    def forward_and_get_loss(self, batch, prefix='', debug=False):
        """
        模型前向传播并计算损失。支持为每个样本生成 G 条轨迹，用于 GRPO。
        """
        # 1. Ground-truth 处理
        agents_future = batch['agents_future'][:, :self._agents_len]  # [B, 32, 81, 5]
        agents_future_valid = torch.ne(agents_future.sum(-1), 0)  # [B, 32, 81]
        agents_interested = batch['agents_interested'][:, :self._agents_len]  # [B, 32]
        anchors = batch['anchors'][:, :self._agents_len]  # [B, 32, 64, 2]
        gt_actions, gt_actions_valid = inverse_kinematics(agents_future, agents_future_valid, dt=0.1, action_len=self._action_len)  # [B, 32, 40, 2]， [B,32,40]True为有效

        current = agents_future[:, :, 0:1, :]  # [B, 32, 1, 5]
        future_downsampled = agents_future[:, :, 1::2, :]  # [B, 32, 40, 5]
        agents_future_41 = torch.cat([current, future_downsampled], dim=2)  # [B, 32, 41, 5]

        gt_actions_normalized = self.normalize_actions(gt_actions)  # [B, 32, 40, 2]
        B, A, T, D = gt_actions_normalized.shape # 维度是 [B, 32, 40, 2]

        log_dict, debug_outputs = {}, {}
        total_loss = 0

        # 2. 编码器前向传播
        encoder_outputs = self.encoder(batch)

        # 3. Denoiser 去噪模块
        if self._train_denoiser:
           G = self.group_size   # 每个样本生成 G 条轨迹

           diffusion_steps = torch.randint(0, self.noise_scheduler.num_steps, (B,), device=agents_future.device).long().unsqueeze(-1).repeat(1, A).view(B, A, 1, 1)  # [B, 32, 1, 1]

           # 生成 G 个 noisy 动作
           noised_action_list = []
           for _ in range(G):
              noise = torch.randn(B, A, T, D).type_as(gt_actions_normalized)
              noised_action = self.noise_scheduler.add_noise(gt_actions_normalized, noise, diffusion_steps)
              noised_action_list.append(noised_action)
 
           noised_action_normalized = torch.stack(noised_action_list, dim=0)  # [G, B, 32, 40, 2]，在这里，G1里面的第一个和G2里面的第一个是同一个样本加噪声后的动作，组应该是每个G中的相同样本的

        # 拓展其他输入维度
           diffusion_steps_g = diffusion_steps.unsqueeze(0).repeat(G, 1, 1, 1, 1)  # [G, B, 32, 1, 1]
           agents_future_41_g = agents_future_41.unsqueeze(0).repeat(G, 1, 1, 1, 1)  # [G, B, 32, 41, 5]
           encoder_outputs_g = {k: v.unsqueeze(0).repeat(G, *([1] * v.ndim)) for k, v in encoder_outputs.items()}

        # G 组前向传播
           denoise_outputs_list = []
           for g in range(G):
              out_g = self.forward_denoiser({k: v[g] for k, v in encoder_outputs_g.items()}, noised_action_normalized[g], diffusion_steps_g[g].view(B, A), agents_future_41_g[g])
              denoise_outputs_list.append(out_g)
            
        # 合并调试输出
           debug_outputs['noised_action_normalized'] = noised_action_normalized
           debug_outputs['diffusion_steps'] = diffusion_steps
           debug_outputs['denoise_outputs'] = denoise_outputs_list
        
        #--------------------------------------后续计算的轨迹和动作值----------------------------------------#
        # 用于后续 GRPO 的 denoised_trajs: [G, B, 32, 80, 5]
           denoised_trajs = torch.stack([out['denoised_trajs'] for out in denoise_outputs_list], dim=0) # 模型的去噪后的轨迹 [G, B, 32, 80, 5]
        #    denoised_trajs_old = torch.stack([out['denoised_trajs_old'] for out in denoise_outputs_list], dim=0)  # 参考模型的去噪后的轨迹 [G, B, 32, 80, 5]

        # 用于后续 GRPO 的 denoised_actions: [G, B, 32, 40, 2]
           denoised_actions = torch.stack([out['denoised_actions'] for out in denoise_outputs_list], dim=0)
           denoised_actions_old = torch.stack([out['denoised_actions_old'] for out in denoise_outputs_list], dim=0)  # 参考模型的去噪后的动作 [G, B, 32, 40, 2]

           rewards = self.compute_grpo_trajectory_reward(denoised_trajs, denoised_actions, 5.0, 2.0) # 维度是 [G, B]
           
           action_differences =[]
           action_differences_old = []
           for g in range(G):
                action_diff = F.mse_loss(denoised_actions[g], gt_actions_normalized, reduction='none')  # [B, 32, 40, 2]
                action_diff = action_diff * gt_actions_valid.unsqueeze(-1)  # [B, 32, 40, 2]
                action_diff = action_diff.mean(dim=(1, 2, 3))  # [B]
                action_differences.append(action_diff)

                action_diff_old = F.mse_loss(denoised_actions_old[g], gt_actions_normalized, reduction='none')  # [B, 32, 40, 2]
                action_diff_old = action_diff_old * gt_actions_valid.unsqueeze(-1) 
                action_diff_old = action_diff_old.mean(dim=(1, 2, 3))  # [B]
                action_differences_old.append(action_diff_old)
            
           action_differences = torch.stack(action_differences, dim=0)  # [G, B]
           action_differences_old = torch.stack(action_differences_old, dim=0)  # [G, B]
           
           # 6. 计算 ratio: 参考差异 - 当前差异（模拟 log-ratio）
           # 模拟一个 ref_policy 的差异（这里以 GT 为准，添加恒定噪声或使用 ref 模型更好）
           ref_diff = action_differences_old
           ratios = torch.exp(ref_diff - action_differences)  # [G, B]
           
           # 7. GRPO核心损失计算
           rewards = rewards.permute(1, 0).contiguous().view(B, G)  # [B, G]
           ratios = ratios.permute(1, 0).contiguous().view(B, G)    # [B, G]

           group_mean = rewards.mean(dim=1, keepdim=True)  # [B, 1]
           advantages = rewards - group_mean               # [B, G]
           log_ratios = torch.log(torch.clamp(ratios, min=1e-8, max=10))  # [B, G]

           policy_loss = -(advantages * log_ratios).mean()
           
           # 8. KL惩罚
           kl_div = (ratios - 1 - log_ratios).mean()
           kl_penalty = self.beta * kl_div
           total_loss = policy_loss + kl_penalty
           
           log_dict.update({
                prefix+'GRPO_loss': total_loss,
                prefix+'policy_loss': policy_loss,
                prefix+'kl_penalty': kl_penalty,
            })
            
            # 计算去噪的 ADE 和 FDE 指标
           denoise_ade, denoise_fde = self.calculate_metrics_denoise(denoised_trajs, agents_future, agents_future_valid, agents_interested, 8)

            # 更新日志字典
           log_dict.update({
                prefix + 'denoise_ADE': denoise_ade,
                prefix + 'denoise_FDE': denoise_fde,
            })
        
        ############### 行为先验预测模块 ############
        if self._train_predictor:
            # 使用预测器进行前向传播
            goal_outputs = self.forward_predictor(encoder_outputs)
            debug_outputs.update(goal_outputs)
            # for k, v in goal_outputs.items():
            #     print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else type(v)}")
            # 计算目标损失和分数损失
            goal_scores = goal_outputs['goal_scores']
            goal_trajs = goal_outputs['goal_trajs']
            
            goal_loss_mean, score_loss_mean = self.goal_loss(
                goal_trajs, goal_scores, agents_future,
                agents_future_valid, anchors,
                agents_interested,
            )

            # 计算预测损失
            pred_loss = goal_loss_mean + 0.05 * score_loss_mean
            total_loss += 1.0 * pred_loss 
            
            # 计算预测的 ADE 和 FDE 指标
            pred_ade, pred_fde = self.calculate_metrics_predict(
                goal_trajs, agents_future, agents_future_valid, agents_interested, 8
            )
            
            # 更新日志字典
            log_dict.update({
                prefix + 'goal_loss': goal_loss_mean.item(),
                prefix + 'score_loss': score_loss_mean.item(),
                prefix + 'pred_ADE': pred_ade,
                prefix + 'pred_FDE': pred_fde,
            })
        
        # 更新总损失到日志字典
        log_dict[prefix + 'loss'] = total_loss.item()
        
        # 如果启用调试模式，返回调试输出
        if debug:
            return total_loss, log_dict, debug_outputs
        else:
            return total_loss, log_dict

    def training_step(self, batch, batch_idx):
        """
        模型的训练步骤。
        """        
        # 前向传播并计算损失
        loss, log_dict = self.forward_and_get_loss(batch, prefix='train/')
        # 记录日志
        '''on_step=True: 每个训练步骤都会记录日志。
           on_epoch=False: 不在每个 epoch 结束时记录日志。
           sync_dist=True: 在分布式训练中同步日志。
           prog_bar=True: 将日志显示在进度条中，便于实时监控。'''
        # self.log_dict用于批量记录， self.log用于单个记录
        self.log_dict(
            log_dict, 
            on_step=True, on_epoch=False, sync_dist=True,
            prog_bar=True
        )
        
        return loss
    
    def on_train_epoch_end(self):
         self.reference_model.load_state_dict(copy.deepcopy(self.denoiser.state_dict()))
         self.reference_model.eval()


    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.
        """
        loss, log_dict = self.forward_and_get_loss(batch, prefix='val/')
        self.log_dict(log_dict, 
                      on_step=False, on_epoch=True, sync_dist=True,
                      prog_bar=True)
        
        return loss

################### forward_and_get_loss的配套函数 ###################

#——————------------------------------计算metric----------------------

    @torch.no_grad()
    def calculate_metrics_denoise(self, 
            denoised_trajs, agents_future, agents_future_valid,
            agents_interested, top_k = None
        ):
        """
        计算去噪轨迹的评估指标。

        Args:
            denoised_trajs (torch.Tensor): 去噪后的轨迹，形状为 [G，B, A, T, 2]。
            agents_future (torch.Tensor): 真实未来轨迹，形状为 [B, A, T, 2]。
            agents_future_valid (torch.Tensor): 未来轨迹的有效性掩码，形状为 [B, A, T]。
            agents_interested (torch.Tensor): 感兴趣的代理掩码，形状为 [B, A]。
            top_k (int, optional): 考虑的前 K 个代理，默认为 None。

        Returns:
            Tuple[float, float]: 包含去噪的 ADE（平均位移误差）和 FDE（最终位移误差）的元组。
        """
        if not top_k:
            top_k = self._agents_len  # 如果未指定 top_k，则使用代理总数
        
        # 选择前 top_k 个代理的去噪轨迹和真实轨迹
        denoised_trajs_avg = denoised_trajs.mean(dim=0)[..., :2]  # [B, A, T, 2]
        pred_traj = denoised_trajs_avg[:, :top_k, :, :]  # [B, A, T, 2]
        gt = agents_future[:, :top_k, 1:, :2]  # [B, A, T, 2]
        
        # 计算有效性掩码，考虑未来轨迹的有效性和感兴趣的代理
        gt_mask = (agents_future_valid[:, :top_k, 1:] \
            & (agents_interested[:, :top_k, None] > 0)).bool()  # [B, A, T] 

        # 计算去噪轨迹与真实轨迹的欧几里得距离
        denoise_mse = torch.norm(pred_traj - gt, dim=-1)  # [B, A, T]
        
        # 计算 ADE（平均位移误差）和 FDE（最终位移误差）
        denoise_ADE = denoise_mse[gt_mask].mean()  # ADE：所有时间步的平均误差
        denoise_FDE = denoise_mse[..., -1][gt_mask[..., -1]].mean()  # FDE：最后一个时间步的误差
        
        return denoise_ADE.item(), denoise_FDE.item()  # 返回 ADE 和 FDE

    def goal_loss(
        self, trajs, scores, agents_future,
        agents_future_valid, anchors,
        agents_interested
    ):
        """
        计算轨迹预测的损失。

        Args:
            trajs (torch.Tensor): 预测的轨迹张量，形状为 [B*A, Q, T, 3]。
            scores (torch.Tensor): 预测的分数张量，形状为 [B*A, Q]。
            agents_future (torch.Tensor): 未来代理状态张量，形状为 [B, A, T, 3]。
            agents_future_valid (torch.Tensor): 未来代理状态的有效性掩码，形状为 [B, A, T]。
            anchors (torch.Tensor): 锚点张量，形状为 [B, A, Q, 2]。
            agents_interested (torch.Tensor): 感兴趣的代理掩码，形状为 [B, A]。

        Returns:
            traj_loss_mean (torch.Tensor): 平均轨迹损失。
            score_loss_mean (torch.Tensor): 平均分数损失。
        """
        # 将锚点转换到全局坐标系
        current_states = agents_future[:, :, 0, :3]  # 当前状态，形状为 [B, A, 3]
        anchors_global = batch_transform_trajs_to_global_frame(anchors, current_states)  # 转换后的锚点，形状为 [B, A, Q, 2]
        num_batch, num_agents, num_query, _ = anchors_global.shape  # 获取批次大小、代理数量和查询数量

        # 获取掩码，表示有效的未来轨迹和感兴趣的代理
        traj_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)  # [B, A, T]

        # 将批次和代理维度展平
        goal_gt = agents_future[:, :, -1:, :2].flatten(0, 1)  # 真实目标点，形状为 [B*A, 1, 2]
        trajs_gt = agents_future[:, :, 1:, :3].flatten(0, 1)  # 真实轨迹，形状为 [B*A, T, 3]
        trajs = trajs.flatten(0, 1)[..., :3]  # 预测轨迹，形状为 [B*A, Q, T, 3]
        anchors_global = anchors_global.flatten(0, 1)  # 展平后的锚点，形状为 [B*A, Q, 2]

        # 找到与真实目标点最近的锚点索引
        idx_anchor = torch.argmin(torch.norm(anchors_global - goal_gt, dim=-1), dim=-1)  # [B*A]

        # 对于没有有效终点的代理，使用最小ADE（平均位移误差）
        dist = torch.norm(trajs[:, :, :, :2] - trajs_gt[:, None, :, :2], dim=-1)  # 计算预测轨迹与真实轨迹的欧几里得距离，形状为 [B*A, Q, T]
        dist = dist * traj_mask.flatten(0, 1)[:, None, :]  # 应用掩码，形状为 [B*A, Q, T]
        idx = torch.argmin(dist.mean(-1), dim=-1)  # 找到最小ADE的索引，形状为 [B*A]

        # 根据有效性选择索引
        idx = torch.where(agents_future_valid[..., -1].flatten(0, 1), idx_anchor, idx)  # [B*A]
        trajs_select = trajs[torch.arange(num_batch * num_agents), idx]  # 根据索引选择轨迹，形状为 [B*A, T, 3]

        # 计算轨迹损失
        traj_loss = smooth_l1_loss(trajs_select, trajs_gt, reduction='none').sum(-1)  # [B*A, T]
        traj_loss = traj_loss * traj_mask.flatten(0, 1)  # 应用掩码，形状为 [B*A, T]

        # 计算分数损失
        scores = scores.flatten(0, 1)  # 展平分数张量，形状为 [B*A, Q]
        score_loss = cross_entropy(scores, idx, reduction='none')  # 交叉熵损失，形状为 [B*A]
        score_loss = score_loss * (agents_interested.flatten(0, 1) > 0)  # 应用感兴趣代理掩码，形状为 [B*A]

        # 计算平均损失
        traj_loss_mean = traj_loss.sum() / traj_mask.sum()  # 平均轨迹损失
        score_loss_mean = score_loss.sum() / (agents_interested > 0).sum()  # 平均分数损失

        return traj_loss_mean, score_loss_mean  # 返回平均轨迹损失和平均分数损失

    @torch.no_grad()
    def calculate_metrics_predict(self,
            goal_trajs, agents_future, agents_future_valid,
            agents_interested, top_k = None
        ):
        """
        计算目标轨迹预测的评估指标。
        """

        if not top_k:
            top_k = self._agents_len  # 如果未指定 top_k，则使用代理总数
        
        # 选择前 top_k 个代理的目标轨迹和真实轨迹
        goal_trajs = goal_trajs[:, :top_k, :, :, :2]  # [B, A, Q, T, 2]
        gt = agents_future[:, :top_k, 1:, :2]  # [B, A, T, 2]
        
        # 计算有效性掩码，考虑未来轨迹的有效性和感兴趣的代理
        gt_mask = (agents_future_valid[:, :top_k, 1:] \
            & (agents_interested[:, :top_k, None] > 0)).bool()  # [B, A, T] 
                   
        # 计算目标轨迹与真实轨迹的欧几里得距离
        goal_mse = torch.norm(goal_trajs - gt[:, :, None, :, :], dim=-1)  # [B, A, Q, T]
        goal_mse = goal_mse * gt_mask[..., None, :]  # [B, A, Q, T]
        
        
        # 找到与真实轨迹最接近的预测轨迹索引
        best_idx = torch.argmin(goal_mse.sum(-1), dim=-1)  # [B, A]

        # 根据最优索引选择预测轨迹
        best_goal_mse = goal_mse[torch.arange(goal_mse.shape[0])[:, None],
                                 torch.arange(goal_mse.shape[1])[None, :],
                                 best_idx]  # [B, A, T]
        
        # 计算 ADE（平均位移误差）和 FDE（最终位移误差）
        goal_ADE = best_goal_mse.sum() / gt_mask.sum()  # ADE：所有时间步的平均误差
        goal_FDE = best_goal_mse[..., -1].sum() / gt_mask[..., -1].sum()  # FDE：最后一个时间步的误差
        
        return goal_ADE.item(), goal_FDE.item()  # 返回 ADE 和 FDE
#——————------------------------------计算轨迹的奖励值--------------------------------------------

    def compute_grpo_trajectory_reward(self,
      trajectories, 
      actions=None, 
      v_target=5.0, 
      collision_dist=2.0, 
      weights=None
):
      """
      后续可以考虑接入写好的模块
      
       trajectories: [G, B, N, T, 5] - [x, y, theta, v_x, v_y]
       actions: [G, B, N, A, 2] - [accel, yaw_rate] (optional)
      Returns:
         rewards: [G, B]
    """
      G, B, N, T, D = trajectories.shape
      assert D == 5, "Each state must be [x, y, theta, v_x, v_y]"

      if weights is None:
        weights = {
            "smoothness": 1.0,
            "speed": 1.0,
            "orientation": 1.0,
            "collision": 2.0,
            "action_accel": 1.0,
            "action_yaw": 1.0,
        }

      x, y, theta, v_x, v_y = (
        trajectories[..., 0], 
        trajectories[..., 1], 
        trajectories[..., 2], 
        trajectories[..., 3], 
        trajectories[..., 4]
    )
      v = torch.sqrt(v_x**2 + v_y**2)

    # 1. Smoothness Reward
      acc = v[..., 1:] - v[..., :-1]  # [G, B, N, T-1]
      smoothness_reward = -torch.mean(acc**2, dim=(2, 3))  # [G, B]

    # 2. Speed Reward
      speed_diff = (v - v_target) ** 2  # [G, B, N, T]
      speed_reward = -torch.mean(speed_diff, dim=(2, 3))  # [G, B]

    # 3. Orientation Reward
      d_theta = theta[..., 1:] - theta[..., :-1]  # [G, B, N, T-1]
      orientation_reward = -torch.mean(d_theta**2, dim=(2, 3))  # [G, B]

    # 4. Collision Penalty
      pos = torch.stack([x, y], dim=-1)  # [G, B, N, T, 2]
      collision_penalty = torch.zeros(G, B, device=trajectories.device)
      for g in range(G):
        for b in range(B):
            min_dist = []
            for t in range(T):
                dist = torch.cdist(pos[g, b, :, t], pos[g, b, :, t], p=2)  # [N, N]
                mask = ~torch.eye(N, dtype=torch.bool, device=trajectories.device)
                min_d = dist[mask].min()
                min_dist.append(min_d)
            min_dist = torch.stack(min_dist)
            penalty = torch.mean((collision_dist - min_dist).clamp(min=0.0) ** 2)
            collision_penalty[g, b] = penalty

    # 5. Action Penalty
      if actions is not None:
        G2, B2, N2, A, D2 = actions.shape
        assert (G2, B2, N2) == (G, B, N), "Action tensor shape mismatch"
        accel = actions[..., 0]
        yaw_rate = actions[..., 1]
        accel_penalty = -torch.mean(accel**2, dim=(2, 3))  # [G, B]
        yaw_penalty = -torch.mean(yaw_rate**2, dim=(2, 3))  # [G, B]
      else:
        accel_penalty = torch.zeros(G, B, device=trajectories.device)
        yaw_penalty = torch.zeros(G, B, device=trajectories.device)

    # Combine Weighted Rewards
      total_reward = (
        weights["smoothness"] * smoothness_reward +
        weights["speed"] * speed_reward +
        weights["orientation"] * orientation_reward -
        weights["collision"] * collision_penalty +
        weights["action_accel"] * accel_penalty +
        weights["action_yaw"] * yaw_penalty
    )  # [G, B]

      return total_reward

################### Helper Functions ##############
#------------------------设备转换------------------------#
    def batch_to_device(self, input_dict: dict, device: torch.device = 'cuda'):
        """
        将输入字典中的张量移动到指定的设备上。

        Args:
            input_dict (dict): 包含张量的输入字典。
            device (torch.device): 目标设备，例如 'cuda' 或 'cpu'。

        Returns:
            dict: 将张量移动到目标设备后的输入字典。
        """
        for key, value in input_dict.items():
            # 如果字典的值是一个张量
            if isinstance(value, torch.Tensor):
                # 将张量移动到指定设备
                input_dict[key] = value.to(device)

        # 返回更新后的字典
        return input_dict

#------------------动作进行归一化和反归一化------------------#
    def normalize_actions(self, actions: torch.Tensor):
        """
        使用存储的均值和标准差对给定的动作进行归一化。

        Args:
            actions (torch.Tensor): 要归一化的动作张量。

        Returns:
            torch.Tensor: 归一化后的动作张量。
        """
        # 使用公式 (actions - 均值) / 标准差 对动作进行归一化
        return (actions - self.action_mean) / self.action_std
    
    def unnormalize_actions(self, actions: torch.Tensor):
        """
        使用存储的均值和标准差对归一化的动作进行反归一化。

        Args:
            actions (torch.Tensor): 要反归一化的动作张量。

        Returns:
            torch.Tensor: 反归一化后的动作张量。
        """
        # 使用公式 actions * 标准差 + 均值 对动作进行反归一化
        return actions * self.action_std + self.action_mean