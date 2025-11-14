# utils.py
import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"模型已保存至：{path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"加载模型：{path}")
    return model
