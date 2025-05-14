import os
# 设置镜像站点地址，确保从镜像站点下载数据
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# 设置镜像站点地址，确保从镜像站点下载数据
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def load_data(index_filename, data_filename):
    """
    从文件加载 FAISS 索引和数据
    """
    index = faiss.read_index(index_filename)
    data = np.load(data_filename, allow_pickle=True).item()
    return index, data['embeddings'], data['documents']

def search_index(query, index, documents, model, top_k=5):
    """
    在 FAISS 索引中进行搜索并返回最相关的文档
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [(documents[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results

def prepare_context_from_retrieved_docs(query, index, documents, embedding_model, top_k=5, max_length=512):
    """
    检索最相关的文档，并将内容控制在模型最大输入长度内
    """
    # 检索相关文档
    retrieved_docs = search_index(query, index, documents, embedding_model, top_k=top_k)

    # 拼接检索结果，控制总长度不超过max_length
    context = ""
    for doc, _ in retrieved_docs:
        if len(context) + len(doc) > max_length:
            break
        context += doc + "\n"
    return context

def generate_answer(query, model, tokenizer, index, documents, embedding_model, max_length=512):
    """
    生成给定查询的答案
    """
    # 从索引中检索相关内容，并构建上下文
    context = prepare_context_from_retrieved_docs(query, index, documents, embedding_model, top_k=5, max_length=max_length)

    # 将上下文与用户问题结合
    input_text = f"上下文信息:\n{context}\n\n问题: {query}"
    messages = [{"role": "user", "content": input_text}]

    # 使用生成模型生成答案
    response = model.chat(tokenizer, messages)
    return response

def calculate_similarity(generated_answer, ground_truth, similarity_model):
    """
    计算生成的回答与真实答案之间的相似度
    """
    generated_embedding = similarity_model.encode(generated_answer, convert_to_tensor=True)
    ground_truth_embedding = similarity_model.encode(ground_truth, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(generated_embedding, ground_truth_embedding).item()
    return similarity_score

def main(input_file, output_file):
    # 加载生成模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/afterSFT/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/afterSFT/Baichuan2-7B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained("/root/autodl-tmp/afterSFT/Baichuan2-7B-Chat")

    # 加载检索模型和数据
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 用于查询的向量化
    index, doc_embeddings, documents = load_data("tugraph_faiss_index.index", "tugraph_data.npy")

    # 加载相似度计算模型
    similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 处理输入文件并生成答案
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f]

    total_similarity = 0
    count = 0
    output_data = []
    
    for entry in tqdm(input_data, desc="Processing queries"):
        query = entry['input_field']
        ground_truth = entry['output_field']
        
        # 使用 RAG 模型生成答案
        generated_answer = generate_answer(query, model, tokenizer, index, documents, embedding_model)
        
        # 计算生成答案与真实答案的相似度
        similarity = calculate_similarity(generated_answer, ground_truth, similarity_model)
        total_similarity += similarity
        count += 1

        # 保存生成的答案和相似度
        output_data.append({
            "id": entry['id'],
            "input_field": query,
            "output_field": generated_answer,
            "ground_truth": ground_truth,
            "similarity_score": similarity
        })

    # 计算平均相似度
    average_similarity = total_similarity / count
    print(f"Average similarity score: {average_similarity:.4f}")

    # 保存结果到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main("val.jsonl", "val_answer.jsonl")

# Baichuan2-7B-Chat
# Average similarity score: 0.5751
