import os
import re
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

# 初始化用于扩展同义句的大模型
model_name = "baichuan-inc/Baichuan2-13B-Chat"
local_cache_dir = "../../my_models/Baichuan2-13B-Chat"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    revision="v2.0",
    use_fast=False,
    trust_remote_code=True,
    cache_dir=local_cache_dir
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    revision="v2.0",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    cache_dir=local_cache_dir
)
model.generation_config = GenerationConfig.from_pretrained(
    model_name,
    revision="v2.0",
    cache_dir=local_cache_dir
)
model = model.quantize(8).cuda()

def expand_query(query):
    messages = [{"role": "user", "content": f"请生成一个关于'{query}'的同义句。"}]
    expanded_query_1 = model.chat(tokenizer, messages)
    expanded_query_2 = model.chat(tokenizer, messages)
    return [query, expanded_query_1, expanded_query_2]

def parse_sql_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sql_content = file.read()
    insert_statements = re.findall(r'INSERT INTO\s+\w+\s*\(.+?\)\s*VALUES\s*\(.+?\);', sql_content, flags=re.IGNORECASE)
    keys_and_values = []
    for statement in insert_statements:
        key = re.search(r'INSERT INTO\s+\w+\s*\((\w+)', statement, flags=re.IGNORECASE)
        key_value = key.group(1) if key else "UnknownKey"
        keys_and_values.append((key_value, statement))
    return keys_and_values

def load_file_names(file_names_path):
    with open(file_names_path, 'r', encoding='utf-8') as file:
        file_names = [line.strip() for line in file.readlines() if line.strip()]
    return file_names

def load_documents(file_paths, finetune_path):
    documents = []
    keys = []
    for path in file_paths:
        keys_and_statements = parse_sql_file(path)
        for key, statement in keys_and_statements:
            keys.append(key)
            documents.append(statement)
    with open(finetune_path, 'r', encoding='utf-8') as finetune_file:
        for line in finetune_file:
            data = json.loads(line.strip())
            keys.append(data['prompt'])
            documents.append(f"问：{data['prompt']} 答：{data['completion']}")
    return keys, documents

def vectorize_documents(keys, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', batch_size=32):
    model = SentenceTransformer(model_name)
    doc_embeddings = []
    for batch_start in tqdm(range(0, len(keys), batch_size), desc="向量化键"):
        batch = keys[batch_start:batch_start + batch_size]
        embeddings = model.encode(batch, batch_size=batch_size, convert_to_numpy=True)
        doc_embeddings.append(embeddings)
    return np.vstack(doc_embeddings), model

def create_faiss_index(doc_embeddings):
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.add(np.array(doc_embeddings))
    return index

def save_data(doc_embeddings, documents, index, index_filename, data_filename):
    faiss.write_index(index, index_filename)
    print(f"FAISS 索引已保存到 {index_filename}")
    np.save(data_filename, {'embeddings': doc_embeddings, 'documents': documents})
    print(f"向量和文档内容已保存到 {data_filename}")

def load_data(index_filename, data_filename):
    index = faiss.read_index(index_filename)
    data = np.load(data_filename, allow_pickle=True).item()
    print(f"FAISS 索引和数据已从 {index_filename} 和 {data_filename} 加载")
    return index, data['embeddings'], data['documents']

def search_index(query, index, documents, model, top_k=8):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [(documents[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results

def extract_possible_filenames(query):
    possible_filenames = re.findall(r'\b[A-Za-z_]+\.[A-Za-z]+\b', query)
    return possible_filenames

def build_prompt(query, retrieved_chunks, matched_files, index, documents, model, top_k=1):
    prompt = f"Question: {query}\nPlease answer the above question based on the following information.\n"
    for i, (chunk, _) in enumerate(retrieved_chunks, 1):
        prompt += f"Chunk {i}: {chunk}\n"
    for i, filename in enumerate(matched_files, len(retrieved_chunks) + 1):
        file_search_results = search_index(filename, index, documents, model, top_k=top_k)
        for j, (content, _) in enumerate(file_search_results, 1):
            prompt += f"Chunk {i}.{j}: {filename} 文件相关内容：\n{content}\n"
    # prompt+="你的回答应该简明扼要10-30个字即可，中文，不要出现“根据您提供的信息”字样，直接回答即可，模范回答(注意此内容与本问题无关)：通过设置refresh_time和expire_time参数均为0;蚂蚁集团与清华大学联合研发的。"
    prompt+="请结合相关信息（用Chuck给出）回答Question"
    return prompt


import json


def run_inference_on_dataset(dataset_path, index, documents, rag_model, top_k=3):
    results = []
    with open(dataset_path, 'r', encoding='utf-8') as file:
        lines = list(file)
        for line in tqdm(lines, desc="Processing dataset"):
            data = json.loads(line.strip())
            query = data['input_field']
            query_id = data['id']

            # 扩展同义句
            expanded_queries = expand_query(query)
            all_retrieved_chunks = []
            for expanded_query in expanded_queries:
                retrieved_chunks = search_index(expanded_query, index, documents, rag_model, top_k=top_k)
                all_retrieved_chunks.extend(retrieved_chunks)

            # 构建 Prompt 并生成答案
            prompt = build_prompt(query, all_retrieved_chunks[:9], [], index, documents, rag_model)
            print("***********************************************************************************************")
            print('prompt:')
            print(prompt)
            answer = generate_answer(prompt)
            print('answer:')
            print(answer)
            print('------------------------------------------------------------------------------------------------')
            results.append({"id": query_id, "output_field": answer})
    return results

def generate_answer(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = model.chat(tokenizer, messages)
    return response

def save_answers(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        for result in results:
            json.dump(result, file, ensure_ascii=False)
            file.write('\n')

def calculate_similarity(answers_path, groundtruth_path):
    total_similarity = 0
    count = 0
    with open(answers_path, 'r', encoding='utf-8') as answers_file, \
         open(groundtruth_path, 'r', encoding='utf-8') as groundtruth_file:
        for answer_line, truth_line in zip(answers_file, groundtruth_file):
            answer_data = json.loads(answer_line.strip())
            truth_data = json.loads(truth_line.strip())
            similarity = fuzz.ratio(answer_data['output_field'], truth_data['output_field'])
            total_similarity += similarity
            count += 1
    avg_similarity = total_similarity / count if count > 0 else 0
    print(f"Average Similarity: {avg_similarity:.2f}%")
    return avg_similarity

def main():
    print("启动后半天没反应是因为连不到外网，下载不了计算语义相似度的模型，请关闭此程序，启动vpn后再启动此程序")
    sql_file_paths = [
        "J:/Brainstorms/RAG/TuGraph-CPP-Procedure-API.sql",
        "J:/Brainstorms/RAG/tugraph-db-master.sql",
        "J:/Brainstorms/RAG/TuGraph-Python-Procedure-API.sql"
    ]
    finetune_file_path = "J:/Brainstorms/RAG/finetune_dataset.txt"
    index_filename = "J:/Brainstorms/RAG/faiss_index_v2.index"
    data_filename = "J:/Brainstorms/RAG/document_data_v2.npy"

    if os.path.exists(index_filename) and os.path.exists(data_filename):
        index, doc_embeddings, documents = load_data(index_filename, data_filename)
        rag_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    else:
        keys, documents = load_documents(sql_file_paths, finetune_file_path)
        doc_embeddings, rag_model = vectorize_documents(keys)
        index = create_faiss_index(doc_embeddings)
        save_data(doc_embeddings, documents, index, index_filename, data_filename)

    # 验证集推理并计算相似度
    # val_path = "J:/Brainstorms/RAG/赛题数据/val.jsonl"
    # val_results = run_inference_on_dataset(val_path, index, documents, rag_model)
    # val_output_path = "J:/Brainstorms/RAG/answer_val_detail.jsonl"
    # save_answers(val_results, val_output_path)
    # calculate_similarity(val_output_path, val_path)

    # 测试集推理
    test_path = "J:/Brainstorms/RAG/赛题数据/test1.jsonl"
    test_results = run_inference_on_dataset(test_path, index, documents, rag_model)
    save_answers(test_results, "J:/Brainstorms/RAG/answer_test1_detail.jsonl")

if __name__ == "__main__":
    main()