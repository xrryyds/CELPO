import json
import shutil

# 备份原文件
# shutil.copy('your_file.json', 'your_file_backup.json')

# 读取 JSON 文件
with open('/Users/xiongrengrong/项目/CELPO/datasets/exam/exam.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 为每条数据添加 requestion_idx
for idx, item in enumerate(data, start=1):
    item['question_idx'] = idx

# 写回 JSON 文件
with open('/Users/xiongrengrong/项目/CELPO/datasets/exam/exam.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"已成功为 {len(data)} 条数据添加 requestion_idx")
print("原文件已备份为 your_file_backup.json")
