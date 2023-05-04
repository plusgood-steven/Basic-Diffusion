import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os

def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_result(record,save_dir_path):
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    epochs = np.array([e + 1 for e in range(len(record["train_loss"]))])
    plt.plot(epochs, np.array(record["train_loss"]), label="train")
    plt.legend()
    plt.savefig(f"{save_dir_path}/loss.jpg")
    plt.close()

    plt.title("Accuracy Result")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    epochs = np.array([e + 1 for e in range(len(record["train_loss"]))])
    plt.plot(epochs, np.array(record["val_accu"]), label="val")
    plt.legend()
    plt.savefig(f"{save_dir_path}/accuracy.jpg")
    plt.close()

def set_num_workers(cpu_num=8):
    # Num of CPUs you want to use
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)