import os
# 设置镜像站点地址，确保从镜像站点下载数据
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def parse_sql_file(file_path):
    """
    从SQL文件中读取内容并分割成独立的文档段落
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        sql_content = file.read()
    
    # 分割文档，可以按 `CREATE PROCEDURE` 或 `CREATE FUNCTION` 等关键字分割
    documents = re.split(r'CREATE\s+(?:PROCEDURE|FUNCTION)', sql_content, flags=re.IGNORECASE)
    
    # 重新添加关键字并清理空格
    documents = ["CREATE PROCEDURE " + doc.strip() for doc in documents if doc.strip()]
    return documents

def load_documents(file_paths):
    """
    读取所有文件并组合文档
    """
    documents = []
    for path in file_paths:
        documents.extend(parse_sql_file(path))
    return documents

def vectorize_documents(documents, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    使用指定的模型将文档向量化
    """
    model = SentenceTransformer(model_name)
    # 使用 tqdm 包装文档向量化过程
    doc_embeddings = []
    for doc in tqdm(documents, desc="向量化文档"):
        embedding = model.encode(doc)
        doc_embeddings.append(embedding)
    return np.array(doc_embeddings), model

def create_faiss_index(doc_embeddings):
    """
    创建 FAISS 索引并添加向量
    """
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))
    return index

def save_data(doc_embeddings, documents, index, index_filename, data_filename):
    """
    保存 FAISS 索引到文件并保存向量和文本内容到单个 .npy 文件
    """
    # 保存 FAISS 索引
    faiss.write_index(index, index_filename)
    print(f"FAISS 索引已保存到 {index_filename}")
    
    # 将向量和文本内容一起保存到 .npy 文件
    np.save(data_filename, {'embeddings': doc_embeddings, 'documents': documents})
    print(f"向量和文档内容已保存到 {data_filename}")

def load_data(index_filename, data_filename):
    """
    从文件加载 FAISS 索引和数据
    """
    index = faiss.read_index(index_filename)
    data = np.load(data_filename, allow_pickle=True).item()
    print(f"FAISS 索引和数据已从 {index_filename} 和 {data_filename} 加载")
    return index, data['embeddings'], data['documents']

def search_index(query, index, documents, model, top_k=5):
    """
    在 FAISS 索引中进行搜索并返回最相关的文档
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [(documents[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results

def main():
    # 指定SQL文件路径
    file_paths = [
        "/root/autodl-tmp/RAGdata/TuGraph-CPP-Procedure-API.sql",
        "/root/autodl-tmp/RAGdata/tugraph-db-master.sql",
        "/root/autodl-tmp/RAGdata/TuGraph-Python-Procedure-API.sql"
    ]

    # 读取并向量化文档
    documents = load_documents(file_paths)
    doc_embeddings, model = vectorize_documents(documents)

    # 创建 FAISS 索引
    index = create_faiss_index(doc_embeddings)

    # 保存 FAISS 索引和数据到文件
    save_data(doc_embeddings, documents, index, "tugraph_faiss_index.index", "tugraph_data.npy")

    print("FAISS 索引和数据文件已成功创建并保存！")

    # 测试查询
    query = "如何在TuGraph-DB中创建一个新的图实例？"
    index, doc_embeddings, documents = load_data("tugraph_faiss_index.index", "tugraph_data.npy")
    results = search_index(query, index, documents, model)

    print("\n查询结果:")
    for i, (doc, score) in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Document: {doc[:200]}")  # 只显示前200字符
        print(f"Score: {score}\n")

if __name__ == "__main__":
    main()
