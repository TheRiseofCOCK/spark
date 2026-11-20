import os, time
os.makedirs("/content/outputs/diag_colab1", exist_ok=True)
with open("/content/outputs/diag_colab1/proof.txt", "w") as f:
    f.write("diag executed at " + str(time.time()))
print("diag print works")
