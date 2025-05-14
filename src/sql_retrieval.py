import os
import re
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def parse_sql_file(file_path):
    """
    从SQL文件中读取内容，提取有效的文件名（带后缀）作为key，并返回文档段落列表
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        sql_content = file.read()

    # 正则表达式匹配文件名（带后缀），仅限字母、数字、下划线、横线、点号组合的文件名，且必须包含字母
    file_names = re.findall(r'\b[a-zA-Z0-9_-]+\.[a-zA-Z]+\b', sql_content)

    # 去重并过滤无意义的条目
    file_names = [name for name in set(file_names) if any(c.isalpha() for c in name)]

    # 提取独立文档段落，可以按 `CREATE PROCEDURE` 或 `CREATE FUNCTION` 等关键字分割
    documents = re.split(r'CREATE\s+(?:PROCEDURE|FUNCTION)', sql_content, flags=re.IGNORECASE)
    documents = ["CREATE PROCEDURE " + doc.strip() for doc in documents if doc.strip()]

    return file_names, documents


def load_documents(file_paths, finetune_path):
    """
    读取SQL文件和微调数据文件并组合文档，同时提取SQL文件中的文件名列表
    """
    documents = []
    file_keys = set()  # 使用集合存储文件名，确保唯一性

    # 处理 SQL 文件
    for path in file_paths:
        file_names, docs = parse_sql_file(path)
        documents.extend(docs)
        file_keys.update(file_names)  # 添加文件名到 file_keys 中

    # 处理 finetune_dataset.txt 文件
    with open(finetune_path, 'r', encoding='utf-8') as finetune_file:
        for line in finetune_file:
            data = json.loads(line.strip())
            combined_text = data['prompt'] + " " + data['completion']
            documents.append(combined_text)

    # 转换文件名集合为列表格式
    file_keys_list = list(file_keys)
    return file_keys_list, documents


def vectorize_documents(documents, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    doc_embeddings = [model.encode(doc) for doc in tqdm(documents, desc="向量化文档")]
    return np.array(doc_embeddings), model


def create_faiss_index(doc_embeddings):
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))
    return index


def search_index(query, index, documents, model, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [(documents[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results


def build_prompt(query, retrieved_chunks):
    prompt = f"Question: {query}\nPlease answer the above question based on the following information.\n"
    for i, (chunk, _) in enumerate(retrieved_chunks, 1):
        prompt += f"Chunk {i}: {chunk}\n"
    return prompt


def main():
    sql_file_paths = [
        "J:/Brainstorms/RAG/TuGraph-CPP-Procedure-API.sql",
        "J:/Brainstorms/RAG/tugraph-db-master.sql",
        "J:/Brainstorms/RAG/TuGraph-Python-Procedure-API.sql"
    ]
    finetune_file_path = "J:/Brainstorms/RAG/finetune_dataset.txt"

    # 读取并提取 SQL 文件中的文件名列表，并加载所有文档
    file_keys_list, documents = load_documents(sql_file_paths, finetune_file_path)
    print("Extracted File Keys:", file_keys_list)




if __name__ == "__main__":
    main()
