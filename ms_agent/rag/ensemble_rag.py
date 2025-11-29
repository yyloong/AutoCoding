import os
import pickle
import shutil
from typing import Any, List, Optional, Dict
from omegaconf import DictConfig

from ms_agent.utils import get_logger

from modelscope import snapshot_download
from llama_index.core.base.llms.types import MessageRole
from ..llm import LLM, Message
from .base import RAG
import asyncio
from llama_index.core.llms import (
    CustomLLM,
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata
)

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import get_response_synthesizer
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.schema import QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core.tools import ToolMetadata
from llama_index.core.llms import CustomLLM
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.selectors import LLMSingleSelector

logger = get_logger()

class MultiLlamaIndexRAG(RAG):
    """LlamaIndexRAG class to implement the RAG of llama-index

    The configuration needed in the config yaml:
        - name: LlamaIndexRAG
        - embedding: An embedding model, required, default `Qwen/Qwen3-Embedding-0.6B`
        - chunk_size: The chunk_size of splitting, default `512`
        - chunk_overlap: The overlap of each chunk, default `50`
        - retrieve_only: retrieve only will stop using the llm, only use embedding model,
            thus, query methods will not be available. Default `False`
        - storage_dir: The directory to store and load index files, default `./llama_index`
        If not retrieve_only, the llm model will be the same with the model configured in the `llm` fields.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self._validate_config(config)
        self.embedding_model = getattr(
            config.rag, "embedding", "Qwen/Qwen3-Embedding-0.6B"
        )
        self.llm_model = getattr(config.rag, "llm", config.llm.model)
        self.chunk_size = getattr(config.rag, "chunk_size", 512)
        self.chunk_overlap = getattr(config.rag, "chunk_overlap", 50)
        self.use_storage = getattr(config.rag, "use_storage", False)
        self.required_exts = getattr(config.rag, "required_exts", [".txt",".csv"])
        documents_dirs = getattr(config.rag, "documents_dirs", {})
        self.rag_dirs = getattr(documents_dirs, "path_list", [])
        self.llama_debug = LlamaDebugHandler(print_trace_on_end=False)
        callback_manager = CallbackManager([self.llama_debug])
        Settings.callback_manager = callback_manager
        rag_names = getattr(documents_dirs, "name_list", [])
        rag_description = getattr(documents_dirs, "description_list", [])
        self.rags_metadata = [
            ToolMetadata(name=name, description=desc)
            for name, desc in zip(rag_names, rag_description)
        ]
        self.storage_dir = getattr(config.rag, "storage_dir", "./llama_index")
        self.top_k = getattr(config.rag, "top_k", 50)
        self.top_n = getattr(config.rag, "top_n", 5)
        self.num_queries = getattr(config.rag, "num_queries", 1)
        self._setup_embedding_model(config)
        # Set node parser
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        # If retrieve only, don't set LLM
        self._llm_instance = LLM.from_config(self.config)
        class MSCustomLLM(CustomLLM):

            @property
            def metadata(_self) -> LLMMetadata:
                return LLMMetadata(
                    context_window=65536,  # TODO temp value
                    num_output=4096,
                    model_name=self.config.llm.model,
                    is_function_calling_model=True
                )

            @llm_completion_callback()
            def complete(_self, prompt: str,
                            **kwargs) -> CompletionResponse:
                message: Message = self._llm_instance.generate(
                    messages=[Message(role='user', content=prompt)],
                    stream=False,
                    **kwargs)
                return CompletionResponse(text=message.content)

            @llm_completion_callback()
            def stream_complete(_self,
                                prompt: str,
                                formatted: bool = False,
                                **kwargs: Any):
                for message in self._llm_instance.generate(
                        messages=[Message(role='user', content=prompt)],
                        stream=True,
                        **kwargs):
                    yield CompletionResponse(text=message.content)
        Settings.llm = MSCustomLLM()
        self.index = None
        self.query_engine = None

    def _validate_config(self, config: DictConfig):
        """Validate configuration parameters"""
        if not hasattr(config, "rag") or not hasattr(config.rag, "embedding"):
            raise ValueError("Missing rag.embedding parameter in configuration")

        chunk_size = getattr(config.rag, "chunk_size", 512)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

    def _setup_embedding_model(self, config: DictConfig):
        try:
            use_hf = getattr(config, "use_huggingface", False)
            if not use_hf:
                self.embedding_model = snapshot_download(self.embedding_model)

            Settings.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model,
                device="cuda:2",
                embed_batch_size= 32,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")

    async def initialize_all_components(self):
        await self.initialize_index()
        await self.initialize_retriever()
        await self.initialize_postprocessor()
        await self.initialize_selector()
        await self.initialize_synthesis()

    async def initialize_index(self):
        if self.use_storage and os.path.exists(self.storage_dir):
            await self.load_index(self.storage_dir)
        else:
            if not self.rag_dirs:
                raise ValueError("rag_dirs must be specified to initialize index")
            await self.initialize_index_from_path()

    async def initialize_index_from_path(self):
        assert len(self.rag_dirs) == len(
            self.rags_metadata
        ), "documents_dirs.path_list and documents_dirs.name_list/documents_dirs.description_list must have the same length"
        self.index = []
        for dir in self.rag_dirs:
            reader = SimpleDirectoryReader(input_dir=dir, recursive=True, required_exts=self.required_exts)
            docs = reader.load_data()
            self.index.append(VectorStoreIndex.from_documents(documents=docs,show_progress=True))
        await self.save_index()

    async def initialize_retriever(self):
        retrievers = []
        for idx in range(len(self.index)):
            vector_retriever = VectorIndexRetriever(
                index=self.index[idx], similarity_top_k=self.top_k,distance_strategy=""
            )
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.index[idx].docstore, similarity_top_k=self.top_k,verbose=True
            )
            fusion_retriever = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=self.top_k,
                num_queries=self.num_queries,
                mode="reciprocal_rerank",
                verbose=True
            )
            retrievers.append(fusion_retriever)
        self.retriever = retrievers

    async def initialize_postprocessor(self):
        postprocessor = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=self.top_n
        )
        self.postprocessor = postprocessor
    
    async def initialize_selector(self):
        selector = LLMSingleSelector.from_defaults(llm=Settings.llm)
        self.selector = selector
    
    async def initialize_synthesis(self):
        self.synthesis = get_response_synthesizer(llm=Settings.llm,response_mode="compact",verbose=True)

    async def add_documents(self, documents: Dict[str,List[str]]):
        if not documents:
            raise ValueError("Document list cannot be empty")
        if not self.index:
            raise ValueError(
                "Index not initialized, please initialize index before adding documents"
            )
        if not self.query_engine:
            raise ValueError(
                "Query engine not initialized, please set LLM first before adding documents"
            )
        result = []
        for description, docs in documents.items():
            select_result = self.query_engine.selector.select(
                choices=self.rags_metadata.extend(
                    ToolMetadata(
                        description="If other tools are not relevant,choose this tool"
                    ),
                    name="no_relevant_tool",
                ),
                query=f"The description of documentsis {description}. Which rag should it be added to?",
            )
            index = select_result.index
            if index == len(self.rags_metadata):
                result.append(f"No relevant rag for the document: {doc}")
                continue
            for doc in docs:
                document = Document(text=doc)
                self.index[index].insert(document)
        if len(result) == 0:
            await self.initialize_query_engine()
            return f"Successfully added {len(documents)} documents."
        return "\n".join(result)

    async def query(self, query: str) -> str:
        if len(self.retriever) > 1:
            select_choices = self.selector.select(
                choices=self.rags_metadata,
                query=query,
            )
            tool_index = select_choices.selections[0].index
            reason = select_choices.selections[0].reason
            choice = self.rags_metadata[tool_index]
            logger.info(f"Selected rag: {choice.name} for the query.Reason: {reason}")
            retriever = self.retriever[tool_index]
        else:
            choice = self.rags_metadata[0]
            retriever = self.retriever[0]
            logger.info(f"Only one rag available, using rag: {choice.name} for the query.")
        retriever_result = retriever.retrieve(query)
        logger.info(f"Retrieved {len(retriever_result)} documents from rag: {choice.name}")
        logger.info("Score: " + ", ".join([str(node.score) for node in retriever_result]))
        if isinstance(query, str):
            query_bundle = QueryBundle(query)
        else:
            query_bundle = query

        post_processed_result = self.postprocessor.postprocess_nodes(
            retriever_result,query_bundle
        )
        logger.info(f"Post-processed to {len(post_processed_result)} documents after reranking")
        logger.info("Score: " + ", ".join([str(node.score) for node in post_processed_result]))
        if post_processed_result:
            response = self.synthesis.synthesize(
                query=query,
                nodes=post_processed_result,
            )
            logger.info("Synthesis completed.")
            return response.response
        else:
            logger.info("No relevant documents found after post-processing.")
            return "No relevant documents found."

    async def save_index(self, persist_dir: Optional[str] = None):
        """
        Save index using Pickle (Binary) for fast loading.
        """
        if self.index is None or len(self.index) == 0:
            raise ValueError("No index to save, please add documents first")

        save_dir = persist_dir or self.storage_dir
        os.makedirs(save_dir, exist_ok=True)

        for idx in range(len(self.index)):
            # 直接保存为 .pkl 文件，而不是文件夹
            file_name = f"{self.rags_metadata[idx].name}.pkl"
            save_path = os.path.join(save_dir, file_name)
            
            print(f"Saving index binary to {save_path}...")
            # protocol=pickle.HIGHEST_PROTOCOL 确保大文件性能最好
            with open(save_path, "wb") as f:
                pickle.dump(self.index[idx].storage_context, f, protocol=pickle.HIGHEST_PROTOCOL)

    async def load_index(self, persist_dir: Optional[str] = None):
        """
        加载逻辑：只尝试加载 .pkl。
        如果 .pkl 不存在，抛出错误提示用户需要重建，而不是尝试去加载那个会导致崩溃的 JSON。
        """
        print("Loading index from storage...")
        load_dir = persist_dir or self.storage_dir
        
        self.index = []
        for metadata in self.rags_metadata:
            # 构造 .pkl 文件路径
            pkl_path = os.path.join(load_dir, f"{metadata.name}.pkl")
            
            if os.path.exists(pkl_path):
                print(f"Loading binary index: {pkl_path}")
                with open(pkl_path, "rb") as f:
                    # 极速加载
                    loaded_storage_context = pickle.load(f)
                    self.index.append(load_index_from_storage(loaded_storage_context))
            else:
                # 关键修改：不要尝试加载 JSON，直接报错
                # 因为不管是这里加载，还是之前的 JSON 加载，都会 OOM
                error_msg = (
                    f"未找到二进制索引文件: {pkl_path}。\n"
                    f"请从原始文档重新构建索引，并使用新的 save_index 保存为二进制格式。"
                )
                raise FileNotFoundError(error_msg)

    async def get_index_info(self) -> dict:
        """Get index information"""
        if self.index is None:
            return {"status": "not_initialized"}

        doc_count = sum(len(idx.docstore.docs) for idx in self.index)
        return {
            "status": "initialized",
            "document_count": doc_count,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
        }
    
    async def add_index(self, description: str, name: str, documents: List[str]):
        """Add a new index with the given documents"""
        if not documents:
            raise ValueError("Document list cannot be empty")
        if name in self.rag_names:
            raise ValueError(f"Index with name {name} already exists")
        document_objs = [Document(text=doc) for doc in documents]
        new_index = VectorStoreIndex.from_documents(document_objs)
        self.index.append(new_index)
        self.rags_metadata.append(ToolMetadata(name=name, description=description))
        self.rag_names.append(name)
        self.initialize_query_engine()

    async def remove_all_documents(self):
        """Remove all documents from the index"""
        # Clear the index
        self.index = None

        # Clear the query engine
        self.query_engine = None

        # If storage directory exists, optionally clean it up
        if hasattr(self, "storage_dir") and os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir, ignore_errors=True)
            os.makedirs(self.storage_dir, exist_ok=True)

    async def clear_storage(self, persist_dir: Optional[str] = None):
        """Clear the persistent storage directory"""
        clear_dir = persist_dir or self.storage_dir
        if os.path.exists(clear_dir):
            shutil.rmtree(clear_dir)
            os.makedirs(clear_dir, exist_ok=True)
    
    async def llama_debug_info(self) -> str:
        """Get llama index debug information"""
        print("\n🔍 深入检查检索分数:")

# 获取所有 retrieve 类型的事件
# 这里会包含最外层的 FusionRetrieve 和 内部的 Vector/BM25 Retrieve
        events = self.llama_debug.get_event_pairs(event_type="retrieve")

        for i, (start_event, end_event) in enumerate(events):
            # payload 里的 nodes 包含该次检索的结果
            nodes = end_event.payload.get("nodes", [])
            
            print(f"\n--- 事件 #{i+1} ---")
            # 我们可以通过查看结果数量或特征来推断这是哪个 Retriever
            # 通常 Fusion 是最外层，它包含最终的 RRF 分数 (0.033...)
            # 内部的事件则是原始分数
            
            if nodes:
                first_score = nodes[0].score
                print(f"检索到节点数: {len(nodes)}")
                print(f"Top 1 分数: {first_score}")
                
                if first_score and first_score < 0.1: 
                    print("👉 这看起来像 RRF 融合后的分数 (FusionRetriever)")
                elif first_score and first_score > 1.0:
                    print("👉 这看起来像 BM25 的原始分数")
                else:
                    print("👉 这看起来像 Vector 的余弦相似度")
                    
                # 打印前3个详细信息
                for node in nodes[:3]:
                    print(f"   - Score: {node.score:.4f} | ID: {node.node_id[:8]}...")

if __name__ == "__main__":
    from ms_agent.config import Config
    config = Config.from_task("ms_agent/rag/model_config.yaml")
    rag = MultiLlamaIndexRAG(config)
    asyncio.run(rag.initialize_all_components())
    asyncio.run(rag.save_index())
    answer = asyncio.run(rag.query("best algorithm and code for Spaceship Titanic competition"))
    print(answer)
    asyncio.run(rag.llama_debug_info())
