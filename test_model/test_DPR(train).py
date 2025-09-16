import torch
import yaml
import datetime
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # ignore tensorflow warnings

from AAAI2025.DPR.model.DPR import DPR 
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from matplotlib import pyplot as plt


def load_config(file_path):
    """
    加载配置文件

    参数:
        file_path (str): 配置文件的路径

    返回:
        dict: 从 YAML 文件中加载的配置数据
    
    例子：
    database:
           host: localhost
           port: 3306
    
    output：  {'database': {'host': 'localhost', 'port': 3306}}
    """
    with open(file_path, "r") as file:  # 打开指定路径的 YAML 文件
        data = yaml.safe_load(file)  # 使用安全加载方法读取 YAML 文件内容
    return data  # 返回加载的配置数据

def build_parser():
    '''此处传的参数会覆盖VBD.yaml中的参数'''
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg", type=str, default="/home/ubuntu/LTL/My_Python/AAAI2025/config/DPR.yaml")
    
    # Params for override config
    parser.add_argument("-name", "--model_name", type=str, default=None)
    parser.add_argument("-log", "-log_dir", type=str, default=None)
    
    parser.add_argument("-step", "--diffusion_steps", type=int, default=None)
    parser.add_argument("-mean", "--action_mean", nargs=2, metavar=("accel", "yaw"),
                        type=float, default=None)
    parser.add_argument("-std", "--action_std", nargs=2, metavar=("accel", "yaw"),
                        type=float, default=None)
    parser.add_argument("-zD", "--embeding_dim", type=int, default=None)
    parser.add_argument("-clamp", "--clamp_value", type=float, default=None)
    parser.add_argument("-init", "--init_from", type=str, default=None)
    parser.add_argument("-encoder", "--encoder_ckpt", type=str, default=None)
    parser.add_argument("-nN", "--num_nodes", type=int, default=1)
    parser.add_argument("-nG", "--num_gpus", type=int, default=-1)
    parser.add_argument("-sType", "--schedule_type", type=str, default=None)
    parser.add_argument("-sS", "--schedule_s", type=float, default=None)
    parser.add_argument("-sE", "--schedule_e", type=float, default=None)
    parser.add_argument("-scale", "--schedule_scale", type=float, default=None)
    parser.add_argument("-sT", "--schedule_tau", type=float, default=None)
    parser.add_argument("-eV", "--encoder_version", type=str, default=None)
    parser.add_argument("-pred", "--with_predictor", type=bool, default=None)
    parser.add_argument("-type", "--prediction_type", type=str, default=None)
    
    return parser
    
def load_cfg(args):
    """
    加载并更新配置文件

    参数:
        args (argparse.Namespace): 命令行参数对象

    返回:
        dict: 更新后的配置字典
    """
    # 加载配置文件内容到字典
    cfg = load_config(args.cfg)
    
    # 从命令行参数中覆盖配置文件中的配置
    # 遍历命令行参数并更新配置字典
    for key, value in vars(args).items():
        if key == "cfg":  # 跳过配置文件路径参数
            pass
        elif value is not None:  # 如果参数值不为空，则覆盖配置
            cfg[key] = value
    return cfg  # 返回更新后的配置字典


def train(cfg):
    # for key, value in cfg.items():
    #  print(f"{key}: {value}")
    model = DPR(cfg)
    # 构造 batch
    batch = {
        "agents_history": torch.randn(2, 64, 11, 8),
        "agents_interested": torch.randint(0, 2, (2, 64)),
        "agents_future": torch.randn(2, 64, 81, 5),
        "agents_type": torch.randint(0, 4, (2, 64)),  # 假设最多10种类型
        "traffic_light_points": torch.randn(2, 16, 3),
        "polylines": torch.randn(2, 256, 30, 5),
        "polylines_valid": torch.randint(0, 2, (2, 256)),
        "relations": torch.randn(2, 336, 336, 3),
        "anchors": torch.randn(2, 64, 64, 2),
    }
    outputs, _ = model.forward_and_get_loss(batch)
    
    noised_actions = torch.randn(2, 32, 40, 2)  # 每个agent预测 (40+1)*2 个值 (x,y)
    diffusion_step = torch.randint(low=0, high=1000, size=(2, 32))  # 只使用最后一列
    agents_future = torch.randn(2, 32, 41, 5)  # 模拟未来轨迹 [B, 32, 41, 5]
    output_dict = model(batch, noised_actions, diffusion_step, agents_future)
    print("Decoder 输出：")
    for k, v in output_dict.items():
        print(f"{k}: shape = {v.shape}")
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_cfg(args)
    
    train(cfg)