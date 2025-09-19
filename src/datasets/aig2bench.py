import os
import subprocess

# 配置路径
ABC_BIN = "/home/jingyi/workspace/E-Syn/abc/abc"
AIG_DIR = "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/raw_data/bmc"
BENCH_DIR = "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/raw_data/bench_out"

# 创建输出目录
os.makedirs(BENCH_DIR, exist_ok=True)

# 遍历所有 .aig 文件
for fname in os.listdir(AIG_DIR):
    if not fname.endswith(".aig"):
        continue

    aig_path = os.path.join(AIG_DIR, fname)
    bench_path = os.path.join(BENCH_DIR, fname.replace(".aig", ".bench"))

    print(f"Converting {fname} -> {bench_path}")

    # 构造 ABC 命令
    abc_cmd = f"read_aiger {aig_path}; strash; short_names; write_bench -l -no_vdd {bench_path}"

    # 调用 ABC
    try:
        subprocess.run(
            [ABC_BIN, "-c", abc_cmd],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {fname} 转换失败: {e.stderr}")

print("✅ 全部转换完成！")