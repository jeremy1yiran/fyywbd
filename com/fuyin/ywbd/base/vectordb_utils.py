import chromadb
from chromadb.config import Settings
from openai_utils import get_embedding


class InMemoryVecDB:

    def __init__(self, name="demo"):
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.chroma_client.reset()
        self.name = name
        self.collection = self.chroma_client.get_or_create_collection(name=name)

    '''
    
    metadatas=[{"source": self.name} for _ in documents],
    这一行创建一个名为 metadatas 的列表，其长度与 documents 相同。
    对于 documents 列表中的每个文档，它创建一个包含单个键值对的字典。键是 "source"，值是当前对象的 name 属性。
    这意味着每个文档都与一个元数据字典相关联，该字典标识了文档的来源（self.name）。
    使用 _ 作为循环变量表示我们不关心循环的具体迭代项，只关心迭代的次数。
    metadatas  打印出的结果类似于
    [{"source": "demo"}, {"source": "demo"}, {"source": "demo"}]
    断点部分主要查看metadatas 和ids的内容。
   
    '''

    def add_documents(self, documents):
        self.collection.add(
            embeddings=[get_embedding(doc) for doc in documents],
            documents=documents,
            metadatas=[{"source": self.name} for _ in documents],
            ids=[f"id_{i}" for i in range(len(documents))]
        )

    def search(self, query, top_n):
        """检索向量数据库"""
        results = self.collection.query(
            query_embeddings=[get_embedding(query)],
            n_results=top_n   # top_n 是一个整数，表示用户想要从数据库中检索的结果数量。
        )
        return results['documents'][0]