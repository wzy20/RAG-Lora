import os
# 设置镜像站点地址，确保从镜像站点下载数据
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


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
    retrieved_docs = search_index(query, index, documents, embedding_model, top_k=top_k)
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
    context = prepare_context_from_retrieved_docs(query, index, documents, embedding_model, top_k=5, max_length=max_length)
    input_text = f"上下文信息:\n{context}\n\n问题: {query}"
    messages = [{"role": "user", "content": input_text}]
    response = model.chat(tokenizer, messages)
    return response

def main(input_file, output_file="answer.jsonl"):
    # 加载生成模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/afterSFT/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/afterSFT/Baichuan2-7B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained("/root/autodl-tmp/afterSFT/Baichuan2-7B-Chat")

    # 加载检索模型和数据
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    index, doc_embeddings, documents = load_data("tugraph_faiss_index.index", "tugraph_data.npy")

    # 处理输入文件并生成答案
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f]

    output_data = []
    for entry in tqdm(input_data, desc="Processing queries"):
        query = entry['input_field']
        answer = generate_answer(query, model, tokenizer, index, documents, embedding_model)
        output_data.append({
            "id": entry['id'],
            "output_field": answer
        })

    # 保存结果到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main("test1.jsonl", "answer.jsonl")
