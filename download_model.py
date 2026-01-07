import os
from huggingface_hub import snapshot_download

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 替换成你的 HF Token，以 hf_ 开头
MY_TOKEN = "" 

print("开始下载模型...")
snapshot_download(
    repo_id="Qwen/Qwen2-1.5B-Instruct",
    local_dir="/home/xrrfolder/models/Qwen2-1.5B-Instruct",
    max_workers=8,
    token=MY_TOKEN  # <--- 加上这一行
)
print("下载完成！")
