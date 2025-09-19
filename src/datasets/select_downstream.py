import os
import shutil

source_folder = "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/raw_data/aiger"
destination_folder = "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/raw_data/bmc"
latch_threshold = 25

def filter_aiger_files_recursively(source_dir, dest_dir, threshold):
    """
    递归地读取源文件夹及其子文件夹中所有 AIGER 文件的第一行，
    将锁存器数量小于等于阈值的 AIGER 文件复制到目标文件夹。
    """
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"已创建目标文件夹: {dest_dir}")
    else:
        print(f"目标文件夹已存在: {dest_dir}")
        print("注意: 如果目标文件夹中已存在同名文件，它们将被覆盖。")

    processed_count = 0
    copied_count = 0

    # 使用 os.walk() 递归遍历所有文件夹
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            # 仅处理 .aig 文件
            if filename.endswith(".aig"):
                file_path = os.path.join(dirpath, filename)
                processed_count += 1

                try:
                    with open(file_path, "rb") as f:
                        first_line = f.readline().decode('utf-8').strip()

                    parts = first_line.split()
                    if len(parts) >= 6 and parts[0] == "aig":
                        # LATCH 的数量是第 4 个数字 (索引为 3)
                        latch_count = int(parts[3])

                        if latch_count <= threshold:
                            dest_path = os.path.join(dest_dir, filename)
                            shutil.copy2(file_path, dest_path)
                            copied_count += 1
                            print(f"已复制文件: {filename} (L = {latch_count})")
                        # else:
                        #     print(f"跳过文件: {filename} (L = {latch_count})")
                    else:
                        print(f"跳过文件: {filename} (无效的 AIGER 文件头)")

                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
    
    print("\n--- 结果摘要 ---")
    print(f"总共处理了 {processed_count} 个 AIGER 文件。")
    print(f"已复制 {copied_count} 个文件到 {destination_folder}。")

# 运行脚本
if __name__ == "__main__":
    filter_aiger_files_recursively(source_folder, destination_folder, latch_threshold)