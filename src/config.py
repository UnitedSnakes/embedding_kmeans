# import os
# import torch

def setup_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    total_memory_0 = torch.cuda.get_device_properties(0).total_memory
    max_memory_0 = 10 * 1024**3  # 10 GB for GPU 0
    print(f"GPU 0 Total Memory: {total_memory_0 / 1024**3:.2f} GB")
    print(f"GPU 0 Max Memory Allowed: {max_memory_0 / 1024**3:.2f} GB")
    fraction_0 = max_memory_0 / total_memory_0
    torch.cuda.set_per_process_memory_fraction(fraction_0, 0)
    print(f"GPU 0 Memory Usage Fraction: {fraction_0:.4f}")

# setup_gpu()

class Config:
    num_folds = 5
