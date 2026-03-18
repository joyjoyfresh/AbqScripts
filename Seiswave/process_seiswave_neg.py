import os

# 输入文件路径
input_file = r"c:\Users\12462\Documents\Master\Abaqus\Scripts\Seiswave\El_Centro.txt"
# 输出文件路径
output_file = r"c:\Users\12462\Documents\Master\Abaqus\Scripts\Seiswave\El_Centro_neg.txt"

# 读取并处理文件
with open(input_file, 'r') as f:
    lines = f.readlines()

# 处理每一行数据
processed_lines = []
for line in lines:
    # 分割行数据
    parts = line.strip().split()
    if len(parts) == 2:
        time = parts[0]
        value = float(parts[1])
        # 将第二列乘以-1
        neg_value = value * -1
        # 格式化输出
        processed_line = f"{time}\t{neg_value:.6f}\n"
        processed_lines.append(processed_line)

# 写入输出文件
with open(output_file, 'w') as f:
    f.writelines(processed_lines)

print(f"处理完成！输出文件已保存至: {output_file}")
