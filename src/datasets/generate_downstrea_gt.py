import os
import json
import subprocess
import time

# 配置
ABC_BIN = "/home/jingyi/workspace/E-Syn/abc/abc"
AIG_DIR = "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/raw_data/bmc"
OUTPUT_JSON = "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/raw_data/abc_results.json"

# 结果存储
results = {}

# 遍历目录下所有 .aig 文件
for fname in os.listdir(AIG_DIR):
    if not fname.endswith(".aig"):
        continue

    aig_path = os.path.join(AIG_DIR, fname)
    print(f"Running ABC on {fname} ...")

    abc_cmd = f'read_aiger {aig_path}; bmc2 -F 1000'

    start_time = time.time()
    try:
        # 调用 abc
        proc = subprocess.run(
            [ABC_BIN, "-c", abc_cmd],
            capture_output=True,
            text=True,
            timeout=300  
        )
        elapsed = time.time() - start_time

        # 保存结果
        results[fname] = {
            "status": "ok",
            "time_sec": round(elapsed, 2),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip()
        }

    except subprocess.TimeoutExpired:
        results[fname] = {
            "status": "timeout",
            "time_sec": 300
        }
    except Exception as e:
        results[fname] = {
            "status": "error",
            "error": str(e)
        }

# 保存到 JSON 文件
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)

print(f"All results saved to {OUTPUT_JSON}")