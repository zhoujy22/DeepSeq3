import os
import shutil

def count_output_and_dff(file_path):
    output_count = 0
    dff_count = 0
    input_count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('OUTPUT('):
                output_count += 1
            if 'DFF' in line:
                dff_count += 1
            if line.strip().startswith('INPUT('):
                input_count += 1
    return output_count, dff_count, input_count

def filter_and_copy_bench_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith('.bench'):
            file_path = os.path.join(source_dir, filename)
            output_count, dff_count, input_count = count_output_and_dff(file_path)

            if output_count + dff_count > 10 and output_count + dff_count <= 15:
                shutil.copy(file_path, os.path.join(target_dir, filename))
                print(f"Copied: {filename} | OUTPUT: {output_count}, DFF: {dff_count}")

if __name__ == "__main__":
    source_folder = "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/raw_data/subbench"  # 替换为源文件夹路径
    target_folder = "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/raw_data/experimentC"  # 替换为目标文件夹路径
    filter_and_copy_bench_files(source_folder, target_folder)