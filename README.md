# RAG-Lora
# 项目介绍

本项目旨在开发一个智能问答助手，提升TuGraph-DB的用户体验和技术支持能力。选择Baichuan2 +LoRA，并结合FAISS数据库进行RAG检索增强。

# 文件说明

sql_retrieval.py	获取有哪些文件后缀--结果保存至file_names.txt

rag.py			换了强力模型-处理faiss数据库并扩充prompt-跑验证集和测试集

# 流程说明

+----------------------------------------------------------+
|                    创建 FAISS 数据库                     |
+----------------------------------------------------------+
|                                                          |
| 1. 数据加载                                              |
|    - Load SQL Files                                      |
|      - 通过源码文件生成                                  |
|    - Load Finetune Dataset                               |
|      - 通过询问 gpt-4 生成                               |
|                                                          |
| 2. 向量化                                                |
|    - Vectorize Keys using SentenceTransformer            |
|      paraphrase-multilingual-mpnet-base-v2               |
|                                                          |
| 3. FAISS 索引创建                                        |
|    - Create HNSW Index                                   |
|    - Add Embeddings to Index                             |
|                                                          |
| 4. 保存数据                                              |
|    - Save Index to .index File                           |
|    - Save Embeddings and Documents to .npy File          |
+----------------------------------------------------------+
                        |
                        |
                        v
+----------------------------------------------------------+
|                        推理流程                          |
+----------------------------------------------------------+
|                                                          |
| 1. 用户查询输入                                          |
|    - User Query                                          |
|    - Expand Synonym Queries                              |
|                                                          |
| 2. 检索                                                  |
|    - Vectorize Query using SentenceTransformer           |
|    - Search FAISS Index                                  |
|                                                          |
| 3. 文件名匹配                                            |
|    - Extract Filenames using Regex                       |
|    - Search Index with Filenames                         |
|                                                          |
| 4. 生成 Prompt                                           |
|    - Integrate Query Results                             |
|    - Integrate Code Context                              |
|                                                          |
| 5. 生成答案                                              |
|    - Generate Response using Baichuan2-13B-Chat          |
+----------------------------------------------------------+

# 使用的微调框架说明

autodl  一张4090 24GB显存   +   https://github.com/hiyouga/LLaMA-Factory

