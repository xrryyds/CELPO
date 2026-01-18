import json

# 读取JSON文件
with open('/Users/xiongrengrong/项目/CELPO/datasets/exam/exam.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历列表中的每个字典
for item in data:
    if 'qustion_idx' in item:
        item['question_idx'] = item.pop('qustion_idx')

# 保存修改后的JSON文件
with open('/Users/xiongrengrong/项目/CELPO/datasets/exam/exam.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"键名替换完成！共处理 {len(data)} 条记录。")
