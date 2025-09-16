import pickle

# 替换为你的pkl文件路径
pkl_file_path = '/home/ubuntu/LTL/My_Python/AAAI2025/data/test_scenario_ids.pkl'

with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

print(data)