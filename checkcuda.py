import torch

def check_cuda():
    is_available = torch.cuda.is_available()
    print("CUDA Available:", is_available)
    if is_available:
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
        print("CUDA Device Count:", torch.cuda.device_count())
        print("CUDA Current Device:", torch.cuda.current_device())

if __name__ == "__main__":
    check_cuda()

