import torch, platform, time
print("FL-COLAB WORKER v2 hello from colab1")
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
import os
os.makedirs("/content/outputs/hello2_colab1", exist_ok=True)
with open("/content/outputs/hello2_colab1/proof.txt", "w") as f:
    f.write("worker colab1 v2 cell executed a GPU job at " + str(time.time()))
print("job complete")
