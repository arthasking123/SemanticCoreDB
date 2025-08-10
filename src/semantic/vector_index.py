"""
向量索引模块 - 管理向量数据的索引和检索
Windows 兼容版本，支持可选的向量库
"""

import asyncio
import numpy as np
import pickle
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from loguru import logger

# 尝试导入向量库
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss 不可用，将使用基础向量搜索")

try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False
    logger.warning("annoy 不可用，将使用基础向量搜索")

from ..core.config import SemanticConfig


class VectorIndex:
    """
    向量索引引擎 - 管理向量数据的索引和检索
    """
    
    def __init__(self, config: SemanticConfig, storage_config=None):
        """
        初始化向量索引
        
        Args:
            config: 语义配置
        """
        self.config = config
        self.embedding_dimension = config.embedding_dimension
        self.index_type = config.vector_index_type
        
        # 索引目录
        if storage_config and hasattr(storage_config, 'vector_index_dir'):
            self.index_dir = Path(storage_config.vector_index_dir)
        else:
            # 默认值
            self.index_dir = Path("data/vectors")
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 向量存储
        self.vectors = {}  # object_id -> vector
        self.vector_count = 0
        
        # 初始化索引
        self._init_index()
        
        # 加载现有索引
        self._load_index()
        
        logger.info(f"向量索引初始化完成，类型: {self.index_type}")
    
    def _init_index(self):
        """初始化索引"""
        try:
            if self.index_type == "faiss" and FAISS_AVAILABLE:
                # FAISS 索引
                self.index = faiss.IndexFlatIP(self.embedding_dimension)
                self.index_id_map = faiss.IndexIDMap(self.index)
                
            elif self.index_type == "annoy" and ANNOY_AVAILABLE:
                # Annoy 索引
                self.index = AnnoyIndex(self.embedding_dimension, 'angular')
                self.index_id_map = {}
                
            else:
                # 使用基础向量搜索
                self.index = None
                self.index_id_map = {}
                self.index_type = "basic"
                
            logger.info(f"索引初始化完成: {self.index_type}")
            
        except Exception as e:
            logger.error(f"初始化索引失败: {e}")
            # 回退到基础搜索
            self.index = None
            self.index_id_map = {}
            self.index_type = "basic"
            logger.info("回退到基础向量搜索")
    
    def _load_index(self):
        """加载现有索引"""
        try:
            # 加载向量数据
            vectors_file = self.index_dir / "vectors.pkl"
            if vectors_file.exists():
                with open(vectors_file, 'rb') as f:
                    self.vectors = pickle.load(f)
                    self.vector_count = len(self.vectors)
                
                # 重建索引
                self._rebuild_index()
                
                logger.info(f"加载向量索引，向量数: {self.vector_count}")
                
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
    
    def _rebuild_index(self):
        """重建索引"""
        try:
            if self.index_type == "faiss" and FAISS_AVAILABLE:
                # 清空 FAISS 索引
                self.index_id_map = faiss.IndexIDMap(self.index)
                
                # 重新添加向量
                if self.vectors:
                    vectors_array = np.array(list(self.vectors.values()))
                    object_ids = list(self.vectors.keys())
                    
                    # 添加向量到索引 - FAISS 需要整数 ID
                    index_ids = np.arange(len(object_ids))
                    self.index_id_map.add_with_ids(vectors_array, index_ids)
                    
                    # 维护从整数索引到字符串 object_id 的映射
                    if not hasattr(self, 'id_to_object_map'):
                        self.id_to_object_map = {}
                    for i, object_id in enumerate(object_ids):
                        self.id_to_object_map[i] = object_id
                    
            elif self.index_type == "annoy":
                # 清空 Annoy 索引
                self.index = AnnoyIndex(self.embedding_dimension, 'angular')
                self.index_id_map = {}
                
                # 重新添加向量
                for i, (object_id, vector) in enumerate(self.vectors.items()):
                    self.index.add_item(i, vector)
                    self.index_id_map[i] = object_id
                
                # 构建索引
                self.index.build(10)  # 10 棵树
                
        except Exception as e:
            logger.error(f"重建索引失败: {e}")
    
    async def add_vector(self, object_id: str, vector: np.ndarray) -> bool:
        """
        添加向量到索引
        
        Args:
            object_id: 对象 ID
            vector: 向量数据
        
        Returns:
            是否添加成功
        """
        try:
            # 确保向量维度正确
            if len(vector) != self.embedding_dimension:
                if len(vector) > self.embedding_dimension:
                    vector = vector[:self.embedding_dimension]
                else:
                    vector = np.pad(vector, (0, self.embedding_dimension - len(vector)))
            
            # 存储向量
            self.vectors[object_id] = vector
            self.vector_count += 1
            
            # 添加到索引
            if self.index_type == "faiss":
                # FAISS 需要整数 ID，我们维护一个映射
                index_id = self.vector_count - 1  # 因为我们已经增加了 vector_count
                self.index_id_map.add_with_ids(
                    vector.reshape(1, -1), 
                    np.array([index_id])
                )
                # 维护从整数索引到字符串 object_id 的映射
                if not hasattr(self, 'id_to_object_map'):
                    self.id_to_object_map = {}
                self.id_to_object_map[index_id] = object_id
                
            elif self.index_type == "annoy":
                index_id = len(self.index_id_map)
                self.index.add_item(index_id, vector)
                self.index_id_map[index_id] = object_id
                
                # 定期重建索引
                if index_id % 1000 == 0:
                    self.index.build(10)
            
            # 保存索引
            await self._save_index()
            
            logger.debug(f"向量已添加到索引: {object_id}")
            return True
            
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            return False
    
    async def update_vector(self, object_id: str, vector: np.ndarray) -> bool:
        """
        更新向量
        
        Args:
            object_id: 对象 ID
            vector: 新向量数据
        
        Returns:
            是否更新成功
        """
        try:
            # 删除旧向量
            await self.delete_vector(object_id)
            
            # 添加新向量
            return await self.add_vector(object_id, vector)
            
        except Exception as e:
            logger.error(f"更新向量失败: {e}")
            return False
    
    async def delete_vector(self, object_id: str) -> bool:
        """
        删除向量
        
        Args:
            object_id: 对象 ID
        
        Returns:
            是否删除成功
        """
        try:
            if object_id not in self.vectors:
                return True
            
            # 从存储中删除
            del self.vectors[object_id]
            self.vector_count -= 1
            
            # 重建索引（因为 FAISS 和 Annoy 都不支持直接删除）
            self._rebuild_index()
            
            # 保存索引
            await self._save_index()
            
            logger.debug(f"向量已删除: {object_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False
    
    async def search_similar(self, 
                           query_vector: np.ndarray, 
                           k: int = 10, 
                           threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            k: 返回结果数量
            threshold: 相似度阈值
        
        Returns:
            相似对象列表 (object_id, similarity_score)
        """
        try:
            # 确保查询向量维度正确
            if len(query_vector) != self.embedding_dimension:
                if len(query_vector) > self.embedding_dimension:
                    query_vector = query_vector[:self.embedding_dimension]
                else:
                    query_vector = np.pad(query_vector, (0, self.embedding_dimension - len(query_vector)))
            
            results = []
            
            if self.index_type == "faiss":
                # FAISS 搜索
                query_vector = query_vector.reshape(1, -1)
                similarities, index_ids = self.index_id_map.search(query_vector, k)
                
                for i in range(len(similarities[0])):
                    index_id = int(index_ids[0][i])
                    similarity = float(similarities[0][i])
                    
                    # 使用映射转换为真实的字符串对象ID
                    if hasattr(self, 'id_to_object_map') and index_id in self.id_to_object_map:
                        object_id = self.id_to_object_map[index_id]
                        if similarity >= threshold:
                            results.append((object_id, similarity))
                    else:
                        logger.warning(f"⚠️ 索引ID {index_id} 没有对应的对象ID映射")
                        
            elif self.index_type == "annoy":
                # Annoy 搜索
                indices = self.index.get_nns_by_vector(query_vector, k, include_distances=True)
                
                for i, (index_id, distance) in enumerate(zip(indices[0], indices[1])):
                    if index_id in self.index_id_map:
                        object_id = self.index_id_map[index_id]
                        # 将距离转换为相似度
                        similarity = 1.0 / (1.0 + distance)
                        
                        if similarity >= threshold:
                            results.append((object_id, similarity))
            
            else:
                # 基础向量搜索（当没有高级索引时）
                logger.debug("使用基础向量搜索")
                for object_id, vector in self.vectors.items():
                    # 计算余弦相似度
                    dot_product = np.dot(query_vector, vector)
                    norm_query = np.linalg.norm(query_vector)
                    norm_vector = np.linalg.norm(vector)
                    
                    if norm_query > 0 and norm_vector > 0:
                        similarity = dot_product / (norm_query * norm_vector)
                        if similarity >= threshold:
                            results.append((object_id, similarity))
            
            # 按相似度排序
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索相似向量失败: {e}")
            return []
    
    async def batch_search(self, 
                          query_vectors: List[np.ndarray], 
                          k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        批量搜索
        
        Args:
            query_vectors: 查询向量列表
            k: 返回结果数量
        
        Returns:
            搜索结果列表
        """
        try:
            results = []
            
            for query_vector in query_vectors:
                result = await self.search_similar(query_vector, k)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"批量搜索失败: {e}")
            return []
    
    async def get_vector(self, object_id: str) -> Optional[np.ndarray]:
        """
        获取向量
        
        Args:
            object_id: 对象 ID
        
        Returns:
            向量数据
        """
        try:
            return self.vectors.get(object_id)
        except Exception as e:
            logger.error(f"获取向量失败: {e}")
            return None
    
    async def get_vector_count(self) -> int:
        """获取向量总数"""
        return self.vector_count
    
    async def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        try:
            stats = {
                'vector_count': self.vector_count,
                'index_type': self.index_type,
                'embedding_dimension': self.embedding_dimension,
                'index_size': len(self.vectors)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取索引统计信息失败: {e}")
            return {}
    
    async def _save_index(self):
        """保存索引"""
        try:
            # 保存向量数据
            vectors_file = self.index_dir / "vectors.pkl"
            with open(vectors_file, 'wb') as f:
                pickle.dump(self.vectors, f)
            
            # 保存索引文件
            if self.index_type == "faiss":
                index_file = self.index_dir / "faiss.index"
                faiss.write_index(self.index_id_map, str(index_file))
                
            elif self.index_type == "annoy":
                index_file = self.index_dir / "annoy.index"
                self.index.save(str(index_file))
                
                # 保存 ID 映射
                id_map_file = self.index_dir / "id_map.pkl"
                with open(id_map_file, 'wb') as f:
                    pickle.dump(self.index_id_map, f)
                    
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
    
    async def optimize_index(self):
        """优化索引"""
        try:
            if self.index_type == "annoy":
                # 重建 Annoy 索引
                self.index.build(10)
                
            logger.info("索引优化完成")
            
        except Exception as e:
            logger.error(f"优化索引失败: {e}")
    
    async def close(self):
        """关闭向量索引"""
        try:
            await self._save_index()
            logger.info("向量索引已关闭")
        except Exception as e:
            logger.error(f"关闭向量索引失败: {e}")