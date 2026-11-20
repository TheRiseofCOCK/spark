import torch, platform, time
print("FL-COLAB WORKER HELLO from colab2")
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    print("matmul ok:", y.shape, float(y.sum()))
import os
os.makedirs("/content/outputs/hello_colab2", exist_ok=True)
with open("/content/outputs/hello_colab2/proof.txt", "w") as f:
    f.write("worker colab2 executed a GPU job at " + str(time.time()))
print("job complete")
