import torch
wmark_dim=4096
wmark_m=18000
w_key_seed=0
w_key_seed_2nd=1
wmark_seed=123
wmark_seed_2nd=124
lambda_=0
lr = 1e-4
#透かし用秘密鍵生成
def generate_secret_key(num_weights, watermark_dim, seed, device):
    torch.manual_seed(seed)
    return torch.randn(num_weights, watermark_dim, device=device)
#透かし用正解パターンの生成
def generate_watermark_pattern(watermark_dim, seed, device):
    torch.manual_seed(seed)
    return torch.randint(0, 2, (1, watermark_dim), device=device).float()

def generate_ALL0watermark(watermark_dim,seed,device):
    return torch.zeros((1,watermark_dim),device=device).float()

def generate_ALL1watermark(watermark_dim,seed,device):
    return torch.ones((1,watermark_dim),device=device).float()