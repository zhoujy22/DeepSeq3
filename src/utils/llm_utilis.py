import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from accelerate import Accelerator
def get_embeddings(texts, save_path):
    accelerator = Accelerator()
    model_name = "01-ai/Yi-Coder-9B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True,device_map="auto")

    inputs = tokenizer(texts, return_tensors="pt")
    inputs = accelerator.prepare(inputs)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    penultimate_layer_embedding = hidden_states[-2]
    attention_mask = inputs.attention_mask
    sum_embeddings = torch.sum(penultimate_layer_embedding * attention_mask.unsqueeze(-1), dim=1)
    # 计算有效 token 数量
    num_tokens = torch.sum(attention_mask, dim=1)
    # 计算平均 embedding
    mean_pooled_embedding = sum_embeddings / num_tokens.unsqueeze(-1)
    
    mean_pooled_embedding_cpu = mean_pooled_embedding.cpu()
    print(f"Embedding 的维度: {mean_pooled_embedding.shape}")
    return mean_pooled_embedding_cpu
    # if not os.path.exists(os.path.dirname(save_path)):
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # torch.save(mean_pooled_embedding_cpu, save_path)
    
    # device = model.device
    # for k, v in inputs.items():
    #     if isinstance(v, torch.Tensor):
    #         inputs[k] = v.to(device)

    # generated_tokens = model.generate(
    #     **inputs,
    #     max_new_tokens=50,
    #     do_sample=True, # 允许模型生成更具创造性的文本
    #     top_k=50,
    #     top_p=0.95,
    #     temperature=0.7
    # )
    
    # # 将生成的 token ID 转换回人类可读的文本
    # generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    # print("\n--- 生成的文本输出 ---")
    # print(generated_text)
def generate_verilog_prompt(verilog_file_path, dff_name):
    """
    从 Verilog 文件中提取指定 DFF 的信息，并生成特定格式的 prompt。

    Args:
        verilog_file_path (str): Verilog 文件的路径。
        dff_name (str): 要查找的 DFF（触发器）名称。

    Returns:
        str: 生成的 prompt 字符串。如果文件或 DFF 不存在，则返回 None。
    """
    
    with open(verilog_file_path, 'r', encoding='utf-8') as f:
        rtl_content = f.read()

    if dff_name not in rtl_content:
        print(f"警告: 文件中未找到 DFF 名称 '{dff_name}'。")
        return None

    prompt = (
        f"DFF name: {dff_name};\n"
        f"Description:\n"
        f"{dff_name} in the RTL code,\n"
        f"all RTL code is: **{rtl_content}**"
        )
        
    return prompt
def generate_and_prompt():
    prompt = (
        f"module AND_gate (Y, A, B);\n"
        f"Output: Y;\n"
        f"Input A, B;\n"
        f"Y = A & B;\n"
        f"endmodule"
        )
        
    return prompt
def generate_not_prompt():
    prompt = (
        f"module NOT_gate (Y, A);\n"
        f"Output: Y;\n"
        f"Input A;\n"
        f"Y = ~A;\n"
        f"endmodule"
        )
        
    return prompt
    
if __name__ == "__main__":
    texts = [
        "This is a sample text.",
        "Another example text for embedding."
    ]
    save_path = "/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/src/embedding.pt"
    texts = generate_verilog_prompt("/home/jingyi/workspace/DeepSeq2-ICCAD/DeepSeq2-ICCAD/raw_data/experimentB_verilog/aes_core_001.v", "_000_")
    print(texts)
    embeds = get_embeddings(texts, save_path)
    print(f"Embeddings are {embeds}")