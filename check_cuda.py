import torch
print(torch.cuda.is_available(),)
if torch.cuda.is_available():
    print(
        torch.cuda.device_count(),
        torch.cuda.current_device(),
torch.cuda.device(0),
torch.cuda.get_device_name(0))

