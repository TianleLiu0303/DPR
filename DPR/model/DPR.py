import torch
import lightning.pytorch as pl
from AAAI2025.DPR.model.Encoder import Encoder
from AAAI2025.DPR.model.Decoder import Decoder, GoalPredictor
from AAAI2025.DPR.model.utils import DDPM_Sampler
from AAAI2025.DPR.model.model_utils import inverse_kinematics, roll_out, batch_transform_trajs_to_global_frame
from torch.nn.functional import smooth_l1_loss, cross_entropy


class DPR(pl.LightningModule):
    """
    Versertile Behavior Diffusion model.
    """

    def __init__(self, cfg: dict):
        """
        Initialize the VBD model.

        Args:
            cfg (dict): Configuration parameters for the model.
        需要最少拥有以下的函数：
                __init__()	定义网络结构和超参数
                forward()	定义前向传播过程（用于推理），返回模型的输出
                training_step()	定义每一步训练过程，返回损失值
                validation_step()	定义验证集上的处理逻辑，无显式返回值（可以返回 metrics，但默认靠 self.log() 写日志）
                test_step()	定义测试集上的处理逻辑
                configure_optimizers()	指定优化器（及学习率调度器）
        """
        # 调用父类的初始化方法
        super().__init__()
        # 保存超参数到模型中，便于后续访问
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
        
        # 是否训练编码器，默认为 True
        self._train_encoder = cfg.get('train_encoder', True)
        # 是否训练去噪器，默认为 True
        self._train_denoiser = cfg.get('train_denoiser', True)
        # 是否训练预测器，默认为 True
        self._train_predictor = cfg.get('train_predictor', True)
        # 是否包含预测器模块，默认为 True
        self._with_predictor = cfg.get('with_predictor', True)
        # 预测类型，默认为 'sample'
        self._prediction_type = cfg.get('prediction_type', 'sample')
        # 调度器类型，默认为 'cosine'
        self._schedule_type = cfg.get('schedule_type', 'cosine')
        # 是否启用重放缓冲区，默认为 False
        self._replay_buffer = cfg.get('replay_buffer', False)
        # 嵌入维度，默认为 5（默认情况下嵌入是加噪轨迹，因此维度为 5）
        self._embeding_dim = cfg.get('embeding_dim', 5)
        
        # self.encoder_layers:用来规划最后一步的transformer层数
        self.encoder = Encoder(self._encoder_layers, version=self._encoder_version)
        
        self.denoiser = Decoder(
           decoder_drop_path_rate=cfg['decoder_drop_path_rate'],
           action_len=cfg['action_len'],
           predicted_agents_num=cfg['predicted_agents_num'],
           future_len=self._future_len,
           hidden_dim=cfg['hidden_dim'],
           decoder_depth=cfg['decoder_depth'],
           num_heads=cfg['num_heads'],    
        )

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
            noised_actions_normalized: 加噪后的归一化动作张量。维度为 [B, 32, 40, 2]
            diffusion_step: 当前扩散步骤。维度为 [B, 32]
            agents_future: 代理的未来状态张量。维度为 [B, 32, 41, 5]
        Returns:
            denoiser_outputs: 包含去噪器输出的字典。
        """
        # 将归一化的加噪动作反归一化，恢复到原始尺度
        noised_actions = self.unnormalize_actions(noised_actions_normalized) 

        denoiser_output = self.denoiser(encoder_outputs, noised_actions, diffusion_step, agents_future) # 去噪器的动作输出，维度是 [B, 32, 40, 2]

        # 根据扩散步骤和预测类型，计算初始值
        # 此处会根据预类型的不同调整算法
        denoised_actions_normalized = self.noise_scheduler.q_x0(
            denoiser_output, 
            diffusion_step, 
            noised_actions_normalized,
            prediction_type=self._prediction_type
        )
    
        # 获取当前状态（最后一个时间步的状态）
        current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, '考虑的代理数量过多'
   
        # 将去噪后的归一化动作反归一化，恢复到原始尺度
        denoised_actions = self.unnormalize_actions(denoised_actions_normalized)
        
        # 根据去噪后的动作进行轨迹回滚，生成去噪后的轨迹
        denoised_trajs = roll_out(
            current_states, 
            denoised_actions,
            action_len=self.denoiser._action_len, 
            global_frame=True
        )
        
        # 返回包含去噪器输出的字典
        return {
            'denoiser_output': denoiser_output,  # 去噪器的动作输出 维度是 [B, 32, 40, 2]
            'denoised_actions_normalized': denoised_actions_normalized,  # 去噪后的归一化动作
            'denoised_actions': denoised_actions,  # 去噪后反归一化的动作
            'denoised_trajs': denoised_trajs,  # 去噪后的轨迹
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
            'goal_actions_normalized': goal_actions_normalized,  # 归一化的目标动作
            'goal_actions': goal_actions,                        # 目标动作
            'goal_scores': goal_scores,                          # 目标分数
            'goal_trajs': goal_trajs,                            # 目标轨迹
        }

###########################################开始记录训练过程的损失函数#########################################
    #  调用forward函数
    def forward_and_get_loss(self, batch, prefix='', debug=False):
        '''
        batch['agents_history']: torch.Size([3, 64, 11, 8])
        batch['agents_interested']: torch.Size([3, 64])
        batch['agents_future']: torch.Size([3, 64, 81, 5])
        batch['agents_type']: torch.Size([3, 64])
        batch['traffic_light_points']: torch.Size([3, 16, 3])
        batch['polylines']: torch.Size([3, 256, 30, 5])
        batch['polylines_valid']: torch.Size([3, 256])
        batch['relations']: torch.Size([3, 336, 336, 3])
        batch['anchors']: torch.Size([3, 64, 64, 2])'''
        """
        模型的前向传播并计算损失。

        Args:
            batch: 输入批次数据。
            prefix: 损失键的前缀。val/表示验证集，train/表示训练集。
            debug: 是否启用调试模式。

        Returns:
            total_loss: 总损失值。
            log_dict: 包含损失值的字典。
            debug_outputs: 包含调试输出的字典。
        """
        ############## 数据准备模块 ##############
        # 以下为数据的真值
        agents_future = batch['agents_future'][:, :self._agents_len] # 维度是[B, 32, 81, 5] 状态包括x,y,theta, vx, vy
        agents_future_valid = torch.ne(agents_future.sum(-1), 0) #  维度[B, 32, 81] 全为0 就是无效的false
        agents_interested = batch['agents_interested'][:, :self._agents_len] # 维度为[B, 32]
        anchors = batch['anchors'][:, :self._agents_len] # 维度为[B, 32, 64，2]

        # agents_future_41: 维度为[B, 32, 41, 5]，将未来轨迹的第一个时间步和每隔两个时间步的状态保留
        current = agents_future[:, :, 0:1, :] # 维度为[B, 32, 1, 5]
        future_downsampled = agents_future[:, :, 1::2, :] # 维度是[B, 32, 40, 5]
        agents_future_41 = torch.cat([current, future_downsampled], dim=2) # 维度是[B, 32, 41, 5]
        
        # 从轨迹中获取动作和动作有效性 维度为gt_actions：[B, 32, 40, 2], gt_actions_valid：[B, 32, 40]
        gt_actions, gt_actions_valid = inverse_kinematics(agents_future, agents_future_valid, dt=0.1, action_len=self._action_len)
        gt_actions_normalized = self.normalize_actions(gt_actions) # 维度为[B, 32, 40, 2]

        B, A, T, D = gt_actions_normalized.shape  # B，A=32，T=40，D=2
        log_dict = {}  # 用于存储日志信息的字典
        debug_outputs = {}  # 用于存储调试输出的字典
        total_loss = 0  # 总损失loss
        
        ############## 编码器前向传播 ##############
        encoder_outputs = self.encoder(batch) # 返回一些编码后信息的字典

        ############### 去噪模块 ###################
        if self._train_denoiser:
            # 生成一个形状为 (B, 32, 1, 1)
            diffusion_steps = torch.randint(
                0, self.noise_scheduler.num_steps, (B,),
                device=agents_future.device
            ).long().unsqueeze(-1).repeat(1, A).view(B, A, 1, 1)

            # 生成随机噪声 维度是 [B, 32, 40, 2]
            noise = torch.randn(B, A, T, D).type_as(agents_future)

            # 对输入动作添加噪声，生成x_T 维度是 [B, 32, 40, 2]
            noised_action_normalized = self.noise_scheduler.add_noise(
                gt_actions_normalized, # 维度是 [B, 32, 40, 2]
                noise, # 维度是 [B, 32, 40, 2]
                diffusion_steps,# 维度是 [B, 32, 1, 1]  
            )

            if self._replay_buffer:
                with torch.no_grad():
                    # 前向传播一步
                    denoise_outputs = self.forward_denoiser(encoder_outputs, gt_actions_normalized, diffusion_steps.view(B, A), agents_future_41)
                   
                    x_0 = denoise_outputs['denoised_actions_normalized']
        
                    # 从 P(x_t-1 | x_t, x_0) 中采样
                    x_t_prev = self.noise_scheduler.step(
                        model_output=x_0,
                        timesteps=diffusion_steps,
                        sample=noised_action_normalized,
                        prediction_type=self._prediction_type if hasattr(self, '_prediction_type') else 'sample',
                    )
                    noised_action_normalized = x_t_prev.detach()
            
            # 使用去噪器进行前向传播 维度是 ：字典、 [B, 32, 40, 2]、 [B, 32]
            denoise_outputs = self.forward_denoiser(encoder_outputs, noised_action_normalized, diffusion_steps.view(B, A), agents_future_41)
            '''denoiser_output: torch.Size([3, 32, 40, 2])
               denoised_actions_normalized: torch.Size([3, 32, 40, 2])
               denoised_actions: torch.Size([3, 32, 40, 2])
               denoised_trajs: torch.Size([3, 32, 80, 5])'''
            
            # 更新调试输出
            debug_outputs.update(denoise_outputs)
            debug_outputs['noise'] = noise
            debug_outputs['diffusion_steps'] = diffusion_steps

            # 计算去噪损失
            denoised_trajs = denoise_outputs['denoised_trajs'] # 维度是shape=(2, 32, 80, 5)
            ''''sample'：直接生成具体的预测动作。

                'mean'：直接生成具体的预测动作，但是损失函数的计算方式不同。

                'error'：预测与噪声相关的误差，而不是直接的动作或轨迹。'''
            if self._prediction_type == 'sample': # 相当于用轨迹来算误差
                # 根据轨迹的误差计算loss
                state_loss_mean, yaw_loss_mean = self.denoise_loss(
                    denoised_trajs,
                    agents_future, agents_future_valid,
                    agents_interested,
                )

                denoise_loss = state_loss_mean + yaw_loss_mean  # 总去噪损失
                total_loss += denoise_loss  # 累加到总损失
                
                # 预测噪声和真实噪声之间的损失，不计入损失函数的反向传播，只在log字典中记录
                _, diffusion_loss = self.noise_scheduler.get_noise(
                    x_0=denoise_outputs['denoised_actions_normalized'],
                    x_t=noised_action_normalized,
                    timesteps=diffusion_steps,
                    gt_noise=noise,
                )
                                
                # 更新日志字典
                log_dict.update({
                    prefix + 'state_loss': state_loss_mean.item(),
                    prefix + 'yaw_loss': yaw_loss_mean.item(),
                    prefix + 'diffusion_loss': diffusion_loss.item()
                })

            elif self._prediction_type == 'error':
                # 计算噪声之间的均方误差损失
                denoiser_output = denoise_outputs['denoiser_output']
                denoise_loss = torch.nn.functional.mse_loss(
                    denoiser_output, noise, reduction='mean'
                )
                total_loss += denoise_loss

                # 更新日志字典
                log_dict.update({
                    prefix + 'diffusion_loss': denoise_loss.item(),
                })

            elif self._prediction_type == 'mean': # 相当于用动作来算误差
                # 计算动作损失
                pred_action_normalized = denoise_outputs['denoised_actions_normalized']
                denoise_loss = self.action_loss(
                    pred_action_normalized, gt_actions_normalized, gt_actions_valid, agents_interested
                )
                total_loss += denoise_loss

                # 更新日志字典
                log_dict.update({
                    prefix + 'action_loss': denoise_loss.item(),
                })
            else:
                raise ValueError('Invalid prediction type')
                
            # 计算去噪的 ADE 和 FDE 指标
            denoise_ade, denoise_fde = self.calculate_metrics_denoise(
                denoised_trajs, agents_future, agents_future_valid, agents_interested, 8
            )

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
        self.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.
        """
        loss, log_dict = self.forward_and_get_loss(batch, prefix='val/')
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        
        return loss
################### forward_and_get_loss的配套函数 ###################

#——————------------------------------以下函数根据不同的去噪类型，计算不同的损失值----------------------
    def denoise_loss(
            self, denoised_trajs,
            agents_future, agents_future_valid,
            agents_interested
        ):
        """
        计算去噪后的轨迹与真实未来轨迹之间的损失。

        Args:
            denoised_trajs (torch.Tensor): 去噪后的轨迹张量，形状为 [B, A, T, C]。
            agents_future (torch.Tensor): 真实未来轨迹张量，形状为 [B, A, T, 3]。
            agents_future_valid (torch.Tensor): 未来轨迹的有效性掩码，形状为 [B, A, T]。
            agents_interested (torch.Tensor): 感兴趣的代理掩码，形状为 [B, A]。

        Returns:
            state_loss_mean (torch.Tensor): 平均状态损失。
            yaw_loss_mean (torch.Tensor): 平均偏航角损失。
        """
        
        # 去掉第一个时间步，仅保留未来时间步的轨迹
        agents_future = agents_future[..., 1:, :3]
        # 计算未来时间步的有效性掩码，结合感兴趣的代理掩码
        future_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)

        # 计算状态损失（位置误差）
        # [B, A, T] 表示批次大小、代理数量和时间步数
        state_loss = smooth_l1_loss(denoised_trajs[..., :2], agents_future[..., :2], reduction='none').sum(-1)
        
        # 计算偏航角误差
        yaw_error = (denoised_trajs[..., 2] - agents_future[..., 2])
        # 将偏航角误差归一化到 [-π, π] 范围内
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        # 计算偏航角损失（绝对值误差）
        yaw_loss = torch.abs(yaw_error)
        
        # 过滤掉无效的状态损失
        state_loss = state_loss * future_mask
        yaw_loss = yaw_loss * future_mask
        
        # 计算平均状态损失
        state_loss_mean = state_loss.sum() / future_mask.sum()
        # 计算平均偏航角损失
        yaw_loss_mean = yaw_loss.sum() / future_mask.sum()
        
        # 返回状态损失和偏航角损失的均值
        return state_loss_mean, yaw_loss_mean
        
    def action_loss(
        self, actions, actions_gt, actions_valid, agents_interested
    ):
        """
        Calculates the loss for action prediction.

        Args:
            actions (torch.Tensor): Tensor of shape [B, A, T, 2] representing predicted actions.
            actions_gt (torch.Tensor): Tensor of shape [B, A, T, 2] representing ground truth actions.
            actions_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of actions.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            action_loss_mean (torch.Tensor): Mean action loss.
        """
        # Get Mask
        action_mask = actions_valid * (agents_interested[..., None] > 0)
        
        # Calculate the action loss
        action_loss = smooth_l1_loss(actions, actions_gt, reduction='none').sum(-1)
        action_loss = action_loss * action_mask
        
        # Calculate the mean loss
        action_loss_mean = action_loss.sum() / action_mask.sum()
        
        return action_loss_mean\

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
    def calculate_metrics_denoise(self, 
            denoised_trajs, agents_future, agents_future_valid,
            agents_interested, top_k = None
        ):
        """
        计算去噪轨迹的评估指标。

        Args:
            denoised_trajs (torch.Tensor): 去噪后的轨迹，形状为 [B, A, T, 2]。
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
        pred_traj = denoised_trajs[:, :top_k, :, :2]  # [B, A, T, 2]
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

    @torch.no_grad()
    def calculate_metrics_predict(self,
            goal_trajs, agents_future, agents_future_valid,
            agents_interested, top_k = None
        ):
        """
        计算目标轨迹预测的评估指标。

        Args:
            goal_trajs (torch.Tensor): 预测的目标轨迹，形状为 [B, A, Q, T, 2]。
            agents_future (torch.Tensor): 真实未来轨迹，形状为 [B, A, T, 2]。
            agents_future_valid (torch.Tensor): 未来轨迹的有效性掩码，形状为 [B, A, T]。
            agents_interested (torch.Tensor): 感兴趣的代理掩码，形状为 [B, A]。
            top_k (int, optional): 考虑的前 K 个代理，默认为 None。

        Returns:
            tuple: 包含目标 ADE（平均位移误差）和 FDE（最终位移误差）的元组。
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
        goal_mse = goal_mse * gt_mask[..., None, :]  # [B, A, Q, T]\


        
        
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
    
