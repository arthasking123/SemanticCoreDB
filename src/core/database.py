"""
SemanticCoreDB 主数据库类
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import uuid
from datetime import datetime

from loguru import logger

from .config import Config
from .event_store import EventStore
from .metadata_graph import MetadataGraph
from ..storage.chunked_object_store import ChunkedObjectStore
from ..storage.object_store import ObjectStore
from ..semantic.custom_vector_engine import CustomVectorEngine
from ..semantic.vector_index import VectorIndex
from ..semantic.embedding_service import EmbeddingService
from ..query.natural_language_parser import NaturalLanguageParser
from ..query.parser import QueryParser
from ..query.executor import QueryExecutor
from .event_sourcing import EventSourcing


class SemanticCoreDB:
    """
    基于 LLM 的语义驱动数据库主类
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化数据库
        
        Args:
            config: 配置对象，如果为 None 则使用默认配置
        """
        self.config = config or Config()
        self.db_id = str(uuid.uuid4())
        
        # 初始化核心组件
        self._init_components()
        
        # 启动后台任务
        self._start_background_tasks()
        
        logger.info(f"SemanticCoreDB 初始化完成，数据库 ID: {self.db_id}")
    
    def _init_components(self):
        """初始化核心组件"""
        # 创建数据目录
        self._create_data_directories()
        
        # 初始化存储组件
        self.chunked_object_store = ChunkedObjectStore(self.config.storage)
        self.object_store = ObjectStore(self.config.storage)
        self.event_store = EventStore(self.config.storage)
        self.event_sourcing = EventSourcing(self.config.storage)
        self.metadata_graph = MetadataGraph(self.config.storage)
        
        # 初始化语义组件
        self.embedding_service = EmbeddingService(self.config.semantic)
        self.custom_vector_engine = CustomVectorEngine(self.config.semantic, self.config.storage)
        self.vector_index = VectorIndex(self.config.semantic, self.config.storage)
        
        # 初始化 LLM 客户端
        self.llm_client = self._create_llm_client()
        
        # 初始化查询组件
        self.natural_language_parser = NaturalLanguageParser(self.config.query, self.llm_client)
        self.query_parser = QueryParser(self.config.query)
        self.query_executor = QueryExecutor(
            self.chunked_object_store,
            self.vector_index,
            self.metadata_graph,
            self.config.query,
            self.config.semantic,  # 传递语义配置
            self.embedding_service  # 传递嵌入服务
        )
        
        logger.info("核心组件初始化完成")
    
    def _create_data_directories(self):
        """创建数据目录"""
        directories = [
            self.config.storage.data_dir,
            self.config.storage.event_log_dir,
            self.config.storage.object_store_dir,
            self.config.storage.vector_index_dir,
            self.config.storage.metadata_graph_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _create_llm_client(self):
        """创建 LLM 客户端"""
        try:
            from ..semantic.llm_client import create_llm_client
            
            # 根据配置创建 LLM 客户端
            provider = getattr(self.config.semantic, 'llm_provider', 'mock')
            
            if provider == 'mock':
                return create_llm_client('mock')
            elif provider == 'openai':
                openai_config = self.config.semantic.openai
                return create_llm_client('openai', 
                                       api_key=openai_config.api_key, 
                                       model=openai_config.model,
                                       max_tokens=openai_config.max_tokens,
                                       temperature=openai_config.temperature,
                                       timeout=openai_config.timeout)
            elif provider == 'anthropic':
                anthropic_config = self.config.semantic.anthropic
                return create_llm_client('anthropic', 
                                       api_key=anthropic_config.api_key, 
                                       model=anthropic_config.model,
                                       max_tokens=anthropic_config.max_tokens,
                                       timeout=anthropic_config.timeout)
            elif provider == 'local':
                local_config = self.config.semantic.local
                return create_llm_client('local', 
                                       base_url=local_config.base_url, 
                                       model=local_config.model,
                                       timeout=local_config.timeout,
                                       max_tokens=local_config.max_tokens)
            else:
                logger.warning(f"未知的 LLM 提供商: {provider}，使用模拟客户端")
                return create_llm_client('mock')
                
        except Exception as e:
            logger.error(f"创建 LLM 客户端失败: {e}，使用模拟客户端")
            from ..semantic.llm_client import create_llm_client
            return create_llm_client('mock')
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 这里可以启动一些后台任务，比如：
        # - 向量索引重建
        # - 元数据图优化
        # - 冷数据归档
        pass
    
    async def insert(self, data: Dict[str, Any]) -> str:
        """
        插入数据
        
        Args:
            data: 要插入的数据，格式为：
                {
                    "type": "text|image|video|audio|iot",
                    "data": "实际数据或文件路径",
                    "metadata": {...},
                    "tags": [...]
                }
        
        Returns:
            数据对象 ID
        """
        try:
            # 生成对象 ID
            object_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            # 创建事件单元
            event_unit = {
                "event_id": str(uuid.uuid4()),
                "object_id": object_id,
                "event_type": "INSERT",
                "timestamp": timestamp,
                "data": data,
                "metadata": {
                    "db_id": self.db_id,
                    "version": "1.0"
                }
            }
            
            # 存储对象数据（使用 Chunked Object Store）
            await self.chunked_object_store.store_object(object_id, data)
            
            # 生成语义嵌入
            if data.get("type") in ["text", "image", "video", "audio"]:
                embedding = await self.embedding_service.generate_embedding(data)
                await self.custom_vector_engine.add_vector(object_id, embedding)
                await self.vector_index.add_vector(object_id, embedding)
            
            # 更新元数据图
            await self.metadata_graph.add_object(object_id, data, timestamp)
            
            # 记录事件溯源
            await self.event_sourcing.record_event(
                object_id, 
                "INSERT", 
                data, 
                {"db_id": self.db_id, "version": "1.0"}
            )
            
            # 记录事件
            await self.event_store.append_event(event_unit)
            
            logger.info(f"数据插入成功，对象 ID: {object_id}")
            return object_id
            
        except Exception as e:
            logger.error(f"数据插入失败: {e}")
            raise
    
    async def query(self, query: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行查询
        
        Args:
            query: 查询语句或查询对象
        
        Returns:
            查询结果列表
        """
        try:
            # 解析查询
            if isinstance(query, str):
                # 尝试自然语言解析
                nl_parsed = await self.natural_language_parser.parse_query(query)
                if nl_parsed.get('confidence', 0) > 0.7:
                    # 使用自然语言解析结果
                    parsed_query = nl_parsed
                    logger.info(f"使用自然语言解析，置信度: {nl_parsed.get('confidence', 0)}")
                else:
                    # 回退到传统解析
                    parsed_query = await self.query_parser.parse(query)
            else:
                parsed_query = query
            
            # 执行查询
            results = await self.query_executor.execute(parsed_query)
            
            logger.info(f"查询执行完成，返回 {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            raise
    
    async def update(self, object_id: str, data: Dict[str, Any]) -> bool:
        """
        更新数据
        
        Args:
            object_id: 对象 ID
            data: 更新的数据
        
        Returns:
            是否更新成功
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            # 创建更新事件
            event_unit = {
                "event_id": str(uuid.uuid4()),
                "object_id": object_id,
                "event_type": "UPDATE",
                "timestamp": timestamp,
                "data": data,
                "metadata": {
                    "db_id": self.db_id,
                    "version": "1.0"
                }
            }
            
            # 更新对象数据（使用 Chunked Object Store）
            await self.chunked_object_store.update_object(object_id, data)
            
            # 更新语义嵌入
            if data.get("type") in ["text", "image", "video", "audio"]:
                embedding = await self.embedding_service.generate_embedding(data)
                await self.custom_vector_engine.update_vector(object_id, embedding)
                await self.vector_index.update_vector(object_id, embedding)
            
            # 更新元数据图
            await self.metadata_graph.update_object(object_id, data, timestamp)
            
            # 记录事件溯源
            await self.event_sourcing.record_event(
                object_id, 
                "UPDATE", 
                data, 
                {"db_id": self.db_id, "version": "1.0"}
            )
            
            # 记录事件
            await self.event_store.append_event(event_unit)
            
            logger.info(f"数据更新成功，对象 ID: {object_id}")
            return True
            
        except Exception as e:
            logger.error(f"数据更新失败: {e}")
            raise
    
    async def delete(self, object_id: str) -> bool:
        """
        删除数据
        
        Args:
            object_id: 对象 ID
        
        Returns:
            是否删除成功
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            # 创建删除事件
            event_unit = {
                "event_id": str(uuid.uuid4()),
                "object_id": object_id,
                "event_type": "DELETE",
                "timestamp": timestamp,
                "data": {},
                "metadata": {
                    "db_id": self.db_id,
                    "version": "1.0"
                }
            }
            
            # 删除对象数据
            await self.chunked_object_store.delete_object(object_id)
            
            # 删除向量索引
            await self.vector_index.delete_vector(object_id)
            
            # 更新元数据图
            await self.metadata_graph.delete_object(object_id, timestamp)
            
            # 记录事件
            await self.event_store.append_event(event_unit)
            
            logger.info(f"数据删除成功，对象 ID: {object_id}")
            return True
            
        except Exception as e:
            logger.error(f"数据删除失败: {e}")
            raise
    
    async def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        获取单个对象
        
        Args:
            object_id: 对象 ID
        
        Returns:
            对象数据
        """
        try:
            # 使用 Chunked Object Store 获取对象
            object_data = await self.chunked_object_store.get_object(object_id)
            print(object_data)
            return object_data
        except Exception as e:
            logger.error(f"获取对象失败: {e}")
            return None
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            统计信息字典
        """
        try:
            stats = {
                "db_id": self.db_id,
                "total_objects": await self.chunked_object_store.get_object_count(),
                "total_vectors": await self.vector_index.get_vector_count(),
                "total_events": await self.event_store.get_event_count(),
                "metadata_graph_stats": await self.metadata_graph.get_statistics(),
                "storage_stats": await self.chunked_object_store.get_storage_stats(),
                "created_at": datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    async def backup(self, backup_path: str) -> bool:
        """
        备份数据库
        
        Args:
            backup_path: 备份路径
        
        Returns:
            是否备份成功
        """
        try:
            # 这里实现备份逻辑
            logger.info(f"数据库备份到: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"数据库备份失败: {e}")
            return False
    
    async def restore(self, backup_path: str) -> bool:
        """
        恢复数据库
        
        Args:
            backup_path: 备份路径
        
        Returns:
            是否恢复成功
        """
        try:
            # 这里实现恢复逻辑
            logger.info(f"从备份恢复数据库: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"数据库恢复失败: {e}")
            return False
    
    async def close(self):
        """关闭数据库连接"""
        try:
            # 关闭 LLM 客户端
            if hasattr(self, 'llm_client') and self.llm_client:
                await self.llm_client.close()
            
            # 关闭各个组件
            await self.chunked_object_store.close()
            await self.object_store.close()
            await self.vector_index.close()
            await self.metadata_graph.close()
            await self.event_store.close()
            await self.event_sourcing.close()
            await self.embedding_service.close()
            await self.custom_vector_engine.close()
            
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.run(self.close()) 