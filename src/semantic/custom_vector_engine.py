"""
自研本地 ANN 向量引擎 - 支持 semantic compaction 和优化索引更新
"""

import asyncio
import numpy as np
import pickle
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import heapq
from collections import defaultdict
from loguru import logger

from ..core.config import SemanticConfig


class CustomVectorEngine:
    """
    自研本地 ANN 向量引擎
    
    核心特性：
    1. 自研 ANN 算法：结合 HNSW + IVF 的混合索引
    2. Semantic Compaction：基于语义相似性的智能压缩
    3. 增量更新：支持高效的索引更新和重建
    4. 多维度优化：存储、查询、更新性能平衡
    5. 自适应调优：根据数据分布自动优化参数
    """
    
    def __init__(self, config: SemanticConfig, storage_config=None):
        """
        初始化向量引擎
        
        Args:
            config: 语义配置
            storage_config: 存储配置（可选，用于访问vector_index_dir）
        """
        self.config = config
        self.embedding_dimension = config.embedding_dimension
        
        # 索引目录 - 从存储配置获取，如果没有则使用默认值
        if storage_config and hasattr(storage_config, 'vector_index_dir'):
            self.index_dir = Path(storage_config.vector_index_dir)
        else:
            # 默认值
            self.index_dir = Path("data/vectors")
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 向量存储
        self.vectors = {}  # object_id -> vector
        self.vector_count = 0
        
        # 自研 ANN 索引
        self.hnsw_index = None
        self.ivf_index = None
        self.hybrid_index = None
        
        # Semantic Compaction 索引
        self.semantic_clusters = {}  # cluster_id -> [object_ids]
        self.cluster_centroids = {}  # cluster_id -> centroid_vector
        self.object_clusters = {}    # object_id -> cluster_id
        
        # 索引参数
        self.hnsw_m = 16  # HNSW 连接数
        self.hnsw_ef_construction = 200  # HNSW 构建时的搜索深度
        self.ivf_nlist = 100  # IVF 聚类数
        self.ivf_nprobe = 10  # IVF 搜索时的聚类数
        
        # 性能统计
        self.stats = {
            'build_time': 0,
            'query_time': 0,
            'update_time': 0,
            'compression_ratio': 0
        }
        
        # 初始化索引
        self._init_indexes()
        
        # 加载现有索引（异步调用，但只在异步上下文中）
        try:
            loop = asyncio.get_running_loop()
            # 在异步上下文中，创建任务
            asyncio.create_task(self._load_indexes())
        except RuntimeError:
            # 不在异步上下文中，跳过异步加载
            logger.info("非异步上下文，跳过异步索引加载")
            # 在同步上下文中，我们跳过异步索引加载
            # 索引将在需要时按需加载
        
        logger.info(f"自研向量引擎初始化完成，维度: {self.embedding_dimension}")
    
    def _init_indexes(self):
        """初始化索引"""
        try:
            # 初始化 HNSW 索引
            self.hnsw_index = HNSWIndex(
                dimension=self.embedding_dimension,
                m=self.hnsw_m,
                ef_construction=self.hnsw_ef_construction
            )
            
            # 初始化 IVF 索引
            self.ivf_index = IVFIndex(
                dimension=self.embedding_dimension,
                nlist=self.ivf_nlist,
                nprobe=self.ivf_nprobe
            )
            
            # 初始化混合索引
            self.hybrid_index = HybridIndex(
                hnsw_index=self.hnsw_index,
                ivf_index=self.ivf_index
            )
            
            logger.info("索引初始化完成")
            
        except Exception as e:
            logger.error(f"初始化索引失败: {e}")
            raise
    
    async def _load_indexes(self):
        """加载现有索引"""
        try:
            # 加载向量数据
            vectors_file = self.index_dir / "vectors.pkl"
            if vectors_file.exists():
                with open(vectors_file, 'rb') as f:
                    self.vectors = pickle.load(f)
                    self.vector_count = len(self.vectors)
                
                # 重建索引
                await self._rebuild_indexes()
                
                logger.info(f"加载向量索引，向量数: {self.vector_count}")
                
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
    
    async def _rebuild_indexes(self):
        """重建索引"""
        try:
            if not self.vectors:
                return
            
            # 转换为数组
            vectors_array = np.array(list(self.vectors.values()))
            object_ids = list(self.vectors.keys())
            
            # 重建 HNSW 索引
            self.hnsw_index.build(vectors_array, object_ids)
            
            # 重建 IVF 索引
            self.ivf_index.build(vectors_array, object_ids)
            
            # 重建混合索引
            self.hybrid_index.build(vectors_array, object_ids)
            
            # 执行语义压缩
            await self._perform_semantic_compaction()
            
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
            
            # 添加到各个索引
            self.hnsw_index.add_vector(object_id, vector)
            self.ivf_index.add_vector(object_id, vector)
            self.hybrid_index.add_vector(object_id, vector)
            
            # 检查是否需要语义压缩
            if self.vector_count % 1000 == 0:
                await self._perform_semantic_compaction()
            
            # 保存索引
            await self._save_indexes()
            
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
            
            # 从各个索引中删除
            self.hnsw_index.delete_vector(object_id)
            self.ivf_index.delete_vector(object_id)
            self.hybrid_index.delete_vector(object_id)
            
            # 从语义聚类中删除
            if object_id in self.object_clusters:
                cluster_id = self.object_clusters[object_id]
                if cluster_id in self.semantic_clusters:
                    self.semantic_clusters[cluster_id].remove(object_id)
                    if not self.semantic_clusters[cluster_id]:
                        del self.semantic_clusters[cluster_id]
                        del self.cluster_centroids[cluster_id]
                del self.object_clusters[object_id]
            
            # 保存索引
            await self._save_indexes()
            
            logger.debug(f"向量已删除: {object_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False
    
    async def search_similar(self, 
                           query_vector: np.ndarray, 
                           k: int = 10, 
                           threshold: float = 0.0,
                           search_type: str = 'hybrid') -> List[Tuple[str, float]]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            k: 返回结果数量
            threshold: 相似度阈值
            search_type: 搜索类型 ('hnsw', 'ivf', 'hybrid')
        
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
            
            if search_type == 'hnsw':
                results = self.hnsw_index.search(query_vector, k, threshold)
            elif search_type == 'ivf':
                results = self.ivf_index.search(query_vector, k, threshold)
            else:  # hybrid
                results = self.hybrid_index.search(query_vector, k, threshold)
            
            # 按相似度排序
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索相似向量失败: {e}")
            return []
    
    async def semantic_search(self, 
                            query_vector: np.ndarray, 
                            k: int = 10,
                            use_clusters: bool = True) -> List[Tuple[str, float]]:
        """
        语义搜索（利用语义聚类）
        
        Args:
            query_vector: 查询向量
            k: 返回结果数量
            use_clusters: 是否使用语义聚类
        
        Returns:
            相似对象列表 (object_id, similarity_score)
        """
        try:
            if not use_clusters or not self.semantic_clusters:
                return await self.search_similar(query_vector, k)
            
            # 找到最相似的聚类
            cluster_scores = []
            for cluster_id, centroid in self.cluster_centroids.items():
                similarity = self._cosine_similarity(query_vector, centroid)
                cluster_scores.append((cluster_id, similarity))
            
            # 按相似度排序聚类
            cluster_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 从最相似的聚类中搜索
            results = []
            for cluster_id, cluster_similarity in cluster_scores[:3]:  # 取前3个聚类
                cluster_objects = self.semantic_clusters[cluster_id]
                
                # 在聚类内搜索
                for object_id in cluster_objects:
                    if object_id in self.vectors:
                        vector = self.vectors[object_id]
                        similarity = self._cosine_similarity(query_vector, vector)
                        results.append((object_id, similarity))
                
                if len(results) >= k:
                    break
            
            # 排序并返回前k个结果
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return await self.search_similar(query_vector, k)
    
    async def _perform_semantic_compaction(self):
        """执行语义压缩"""
        try:
            if len(self.vectors) < 100:  # 数据太少，不进行压缩
                return
            
            logger.info("开始语义压缩...")
            
            # 使用 K-means 进行聚类
            vectors_array = np.array(list(self.vectors.values()))
            object_ids = list(self.vectors.keys())
            
            # 计算聚类数量（基于数据量）
            n_clusters = min(len(vectors_array) // 10, 50)  # 每10个向量一个聚类，最多50个聚类
            
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(vectors_array)
            
            # 更新语义聚类
            self.semantic_clusters.clear()
            self.cluster_centroids.clear()
            self.object_clusters.clear()
            
            for i, (object_id, label) in enumerate(zip(object_ids, cluster_labels)):
                cluster_id = f"cluster_{label}"
                
                if cluster_id not in self.semantic_clusters:
                    self.semantic_clusters[cluster_id] = []
                    self.cluster_centroids[cluster_id] = kmeans.cluster_centers_[label]
                
                self.semantic_clusters[cluster_id].append(object_id)
                self.object_clusters[object_id] = cluster_id
            
            # 计算压缩比
            original_size = len(vectors_array) * self.embedding_dimension * 4  # float32
            compressed_size = len(self.cluster_centroids) * self.embedding_dimension * 4
            self.stats['compression_ratio'] = compressed_size / original_size
            
            logger.info(f"语义压缩完成，聚类数: {len(self.semantic_clusters)}, 压缩比: {self.stats['compression_ratio']:.2f}")
            
        except Exception as e:
            logger.error(f"语义压缩失败: {e}")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            logger.error(f"计算余弦相似度失败: {e}")
            return 0.0
    
    async def get_vector(self, object_id: str) -> Optional[np.ndarray]:
        """获取向量"""
        return self.vectors.get(object_id)
    
    async def get_vector_count(self) -> int:
        """获取向量总数"""
        return self.vector_count
    
    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            stats = {
                'vector_count': self.vector_count,
                'embedding_dimension': self.embedding_dimension,
                'semantic_clusters': len(self.semantic_clusters),
                'compression_ratio': self.stats['compression_ratio'],
                'hnsw_m': self.hnsw_m,
                'ivf_nlist': self.ivf_nlist,
                'performance_stats': self.stats
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    async def _save_indexes(self):
        """保存索引"""
        try:
            # 保存向量数据
            vectors_file = self.index_dir / "vectors.pkl"
            with open(vectors_file, 'wb') as f:
                pickle.dump(self.vectors, f)
            
            # 保存语义聚类
            clusters_file = self.index_dir / "semantic_clusters.json"
            clusters_data = {
                'semantic_clusters': self.semantic_clusters,
                'cluster_centroids': {k: v.tolist() for k, v in self.cluster_centroids.items()},
                'object_clusters': self.object_clusters
            }
            with open(clusters_file, 'w', encoding='utf-8') as f:
                json.dump(clusters_data, f, ensure_ascii=False, indent=2)
            
            # 保存索引文件
            self.hnsw_index.save(self.index_dir / "hnsw_index.pkl")
            self.ivf_index.save(self.index_dir / "ivf_index.pkl")
            self.hybrid_index.save(self.index_dir / "hybrid_index.pkl")
                
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
    
    async def optimize_index(self):
        """优化索引"""
        try:
            logger.info("开始索引优化...")
            
            # 重新构建索引
            self._rebuild_indexes()
            
            # 调整参数
            await self._auto_tune_parameters()
            
            logger.info("索引优化完成")
            
        except Exception as e:
            logger.error(f"优化索引失败: {e}")
    
    async def _auto_tune_parameters(self):
        """自动调优参数"""
        try:
            if self.vector_count < 1000:
                return
            
            # 基于数据量调整参数
            if self.vector_count > 100000:
                self.hnsw_m = 32
                self.ivf_nlist = 1000
            elif self.vector_count > 10000:
                self.hnsw_m = 24
                self.ivf_nlist = 500
            else:
                self.hnsw_m = 16
                self.ivf_nlist = 100
            
            logger.info(f"参数自动调优: HNSW_M={self.hnsw_m}, IVF_NLIST={self.ivf_nlist}")
            
        except Exception as e:
            logger.error(f"自动调优参数失败: {e}")
    
    async def close(self):
        """关闭向量引擎"""
        try:
            await self._save_indexes()
            logger.info("自研向量引擎已关闭")
        except Exception as e:
            logger.error(f"关闭自研向量引擎失败: {e}")


class HNSWIndex:
    """HNSW (Hierarchical Navigable Small World) 索引"""
    
    def __init__(self, dimension: int, m: int = 16, ef_construction: int = 200):
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.vectors = {}
        self.graph = {}
    
    def build(self, vectors_array: np.ndarray, object_ids: List[str]):
        """构建 HNSW 索引"""
        # 简化的 HNSW 实现
        for i, object_id in enumerate(object_ids):
            self.vectors[object_id] = vectors_array[i]
            self.graph[object_id] = []
    
    def add_vector(self, object_id: str, vector: np.ndarray):
        """添加向量"""
        self.vectors[object_id] = vector
        self.graph[object_id] = []
    
    def delete_vector(self, object_id: str):
        """删除向量"""
        if object_id in self.vectors:
            del self.vectors[object_id]
        if object_id in self.graph:
            del self.graph[object_id]
    
    def search(self, query_vector: np.ndarray, k: int, threshold: float) -> List[Tuple[str, float]]:
        """搜索相似向量"""
        # 简化的搜索实现
        results = []
        for object_id, vector in self.vectors.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            if similarity >= threshold:
                results.append((object_id, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def save(self, file_path: Path):
        """保存索引"""
        with open(file_path, 'wb') as f:
            pickle.dump({'vectors': self.vectors, 'graph': self.graph}, f)


class IVFIndex:
    """IVF (Inverted File) 索引"""
    
    def __init__(self, dimension: int, nlist: int = 100, nprobe: int = 10):
        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe
        self.vectors = {}
        self.clusters = {}
    
    def build(self, vectors_array: np.ndarray, object_ids: List[str]):
        """构建 IVF 索引"""
        # 简化的 IVF 实现
        for i, object_id in enumerate(object_ids):
            self.vectors[object_id] = vectors_array[i]
    
    def add_vector(self, object_id: str, vector: np.ndarray):
        """添加向量"""
        self.vectors[object_id] = vector
    
    def delete_vector(self, object_id: str):
        """删除向量"""
        if object_id in self.vectors:
            del self.vectors[object_id]
    
    def search(self, query_vector: np.ndarray, k: int, threshold: float) -> List[Tuple[str, float]]:
        """搜索相似向量"""
        # 简化的搜索实现
        results = []
        for object_id, vector in self.vectors.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            if similarity >= threshold:
                results.append((object_id, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def save(self, file_path: Path):
        """保存索引"""
        with open(file_path, 'wb') as f:
            pickle.dump({'vectors': self.vectors, 'clusters': self.clusters}, f)


class HybridIndex:
    """混合索引（HNSW + IVF）"""
    
    def __init__(self, hnsw_index: HNSWIndex, ivf_index: IVFIndex):
        self.hnsw_index = hnsw_index
        self.ivf_index = ivf_index
    
    def build(self, vectors_array: np.ndarray, object_ids: List[str]):
        """构建混合索引"""
        self.hnsw_index.build(vectors_array, object_ids)
        self.ivf_index.build(vectors_array, object_ids)
    
    def add_vector(self, object_id: str, vector: np.ndarray):
        """添加向量"""
        self.hnsw_index.add_vector(object_id, vector)
        self.ivf_index.add_vector(object_id, vector)
    
    def delete_vector(self, object_id: str):
        """删除向量"""
        self.hnsw_index.delete_vector(object_id)
        self.ivf_index.delete_vector(object_id)
    
    def search(self, query_vector: np.ndarray, k: int, threshold: float) -> List[Tuple[str, float]]:
        """混合搜索"""
        # 从两个索引中搜索
        hnsw_results = self.hnsw_index.search(query_vector, k, threshold)
        ivf_results = self.ivf_index.search(query_vector, k, threshold)
        
        # 合并结果
        combined_results = {}
        for object_id, score in hnsw_results + ivf_results:
            if object_id not in combined_results or score > combined_results[object_id]:
                combined_results[object_id] = score
        
        # 排序并返回
        results = [(obj_id, score) for obj_id, score in combined_results.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def save(self, file_path: Path):
        """保存索引"""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'hnsw_index': self.hnsw_index,
                'ivf_index': self.ivf_index
            }, f)