"""
元数据图模块 - 管理对象关系和语义标签
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
import networkx as nx
from loguru import logger

from .config import StorageConfig


class MetadataGraph:
    """
    元数据图引擎 - 管理对象关系和语义标签
    """
    
    def __init__(self, config: StorageConfig):
        """
        初始化元数据图
        
        Args:
            config: 存储配置
        """
        self.config = config
        self.metadata_dir = Path(config.metadata_graph_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # 图结构
        self.graph = nx.DiGraph()
        
        # 索引
        self.object_index = {}  # object_id -> node_data
        self.tag_index = {}     # tag -> set of object_ids
        self.type_index = {}    # type -> set of object_ids
        self.time_index = {}    # timestamp -> set of object_ids
        
        # 加载现有图数据
        self._load_graph()
        
        logger.info(f"元数据图初始化完成，节点数: {len(self.graph.nodes)}")
    
    def _load_graph(self):
        """加载现有图数据"""
        try:
            graph_file = self.metadata_dir / "metadata_graph.json"
            if graph_file.exists():
                with open(graph_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                
                # 重建图结构
                self.graph = nx.node_link_graph(graph_data, directed=True)
                
                # 重建索引
                self._rebuild_indexes()
                
                logger.info(f"加载元数据图，节点数: {len(self.graph.nodes)}")
                
        except Exception as e:
            logger.error(f"加载元数据图失败: {e}")
    
    def _rebuild_indexes(self):
        """重建索引"""
        try:
            self.object_index.clear()
            self.tag_index.clear()
            self.type_index.clear()
            self.time_index.clear()
            
            for node_id, node_data in self.graph.nodes(data=True):
                # 对象索引
                if 'object_id' in node_data:
                    self.object_index[node_data['object_id']] = node_data
                
                # 标签索引
                if 'tags' in node_data:
                    for tag in node_data['tags']:
                        if tag not in self.tag_index:
                            self.tag_index[tag] = set()
                        self.tag_index[tag].add(node_id)
                
                # 类型索引
                if 'type' in node_data:
                    obj_type = node_data['type']
                    if obj_type not in self.type_index:
                        self.type_index[obj_type] = set()
                    self.type_index[obj_type].add(node_id)
                
                # 时间索引
                if 'timestamp' in node_data:
                    timestamp = node_data['timestamp']
                    if timestamp not in self.time_index:
                        self.time_index[timestamp] = set()
                    self.time_index[timestamp].add(node_id)
                    
        except Exception as e:
            logger.error(f"重建索引失败: {e}")
    
    async def add_object(self, object_id: str, data: Dict[str, Any], timestamp: str):
        """
        添加对象到图
        
        Args:
            object_id: 对象 ID
            data: 对象数据
            timestamp: 时间戳
        """
        try:
            # 创建节点数据
            node_data = {
                'object_id': object_id,
                'type': data.get('type', 'unknown'),
                'timestamp': timestamp,
                'metadata': data.get('metadata', {}),
                'tags': data.get('tags', []),
                'created_at': datetime.utcnow().isoformat()
            }
            
            # 添加到图
            self.graph.add_node(object_id, **node_data)
            
            # 更新索引
            self.object_index[object_id] = node_data
            
            # 标签索引
            for tag in node_data['tags']:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(object_id)
            
            # 类型索引
            obj_type = node_data['type']
            if obj_type not in self.type_index:
                self.type_index[obj_type] = set()
            self.type_index[obj_type].add(object_id)
            
            # 时间索引
            if timestamp not in self.time_index:
                self.time_index[timestamp] = set()
            self.time_index[timestamp].add(object_id)
            
            # 保存图
            await self._save_graph()
            
            logger.debug(f"对象已添加到图: {object_id}")
            
        except Exception as e:
            logger.error(f"添加对象到图失败: {e}")
    
    async def update_object(self, object_id: str, data: Dict[str, Any], timestamp: str):
        """
        更新对象
        
        Args:
            object_id: 对象 ID
            data: 更新数据
            timestamp: 时间戳
        """
        try:
            if object_id not in self.graph.nodes:
                await self.add_object(object_id, data, timestamp)
                return
            
            # 获取现有节点数据
            node_data = self.graph.nodes[object_id]
            
            # 更新节点数据
            node_data.update({
                'type': data.get('type', node_data.get('type', 'unknown')),
                'timestamp': timestamp,
                'metadata': {**node_data.get('metadata', {}), **data.get('metadata', {})},
                'tags': list(set(node_data.get('tags', []) + data.get('tags', []))),
                'updated_at': datetime.utcnow().isoformat()
            })
            
            # 更新索引
            self.object_index[object_id] = node_data
            
            # 更新标签索引
            old_tags = set(self.graph.nodes[object_id].get('tags', []))
            new_tags = set(node_data['tags'])
            
            # 移除旧标签
            for tag in old_tags - new_tags:
                if tag in self.tag_index and object_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(object_id)
            
            # 添加新标签
            for tag in new_tags - old_tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(object_id)
            
            # 保存图
            await self._save_graph()
            
            logger.debug(f"对象已更新: {object_id}")
            
        except Exception as e:
            logger.error(f"更新对象失败: {e}")
    
    async def delete_object(self, object_id: str, timestamp: str):
        """
        删除对象
        
        Args:
            object_id: 对象 ID
            timestamp: 时间戳
        """
        try:
            if object_id not in self.graph.nodes:
                return
            
            # 获取节点数据
            node_data = self.graph.nodes[object_id]
            
            # 从索引中移除
            if object_id in self.object_index:
                del self.object_index[object_id]
            
            # 从标签索引中移除
            for tag in node_data.get('tags', []):
                if tag in self.tag_index and object_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(object_id)
            
            # 从类型索引中移除
            obj_type = node_data.get('type')
            if obj_type in self.type_index and object_id in self.type_index[obj_type]:
                self.type_index[obj_type].remove(object_id)
            
            # 从时间索引中移除
            if timestamp in self.time_index and object_id in self.time_index[timestamp]:
                self.time_index[timestamp].remove(object_id)
            
            # 从图中移除节点
            self.graph.remove_node(object_id)
            
            # 保存图
            await self._save_graph()
            
            logger.debug(f"对象已删除: {object_id}")
            
        except Exception as e:
            logger.error(f"删除对象失败: {e}")
    
    async def add_relationship(self, source_id: str, target_id: str, relationship_type: str, properties: Dict[str, Any] = None):
        """
        添加关系
        
        Args:
            source_id: 源对象 ID
            target_id: 目标对象 ID
            relationship_type: 关系类型
            properties: 关系属性
        """
        try:
            if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
                logger.warning(f"关系节点不存在: {source_id} -> {target_id}")
                return
            
            edge_data = {
                'type': relationship_type,
                'properties': properties or {},
                'created_at': datetime.utcnow().isoformat()
            }
            
            self.graph.add_edge(source_id, target_id, **edge_data)
            
            # 保存图
            await self._save_graph()
            
            logger.debug(f"关系已添加: {source_id} -> {target_id}")
            
        except Exception as e:
            logger.error(f"添加关系失败: {e}")
    
    async def find_objects_by_tag(self, tag: str) -> List[str]:
        """
        根据标签查找对象
        
        Args:
            tag: 标签
        
        Returns:
            对象 ID 列表
        """
        try:
            if tag in self.tag_index:
                return list(self.tag_index[tag])
            return []
        except Exception as e:
            logger.error(f"根据标签查找对象失败: {e}")
            return []
    
    async def find_objects_by_type(self, obj_type: str) -> List[str]:
        """
        根据类型查找对象
        
        Args:
            obj_type: 对象类型
        
        Returns:
            对象 ID 列表
        """
        try:
            if obj_type in self.type_index:
                return list(self.type_index[obj_type])
            return []
        except Exception as e:
            logger.error(f"根据类型查找对象失败: {e}")
            return []
    
    async def find_objects_by_time_range(self, start_time: str, end_time: str) -> List[str]:
        """
        根据时间范围查找对象
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            对象 ID 列表
        """
        try:
            result = set()
            for timestamp, object_ids in self.time_index.items():
                if start_time <= timestamp <= end_time:
                    result.update(object_ids)
            return list(result)
        except Exception as e:
            logger.error(f"根据时间范围查找对象失败: {e}")
            return []
    
    async def find_related_objects(self, object_id: str, max_depth: int = 2) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        查找相关对象
        
        Args:
            object_id: 对象 ID
            max_depth: 最大深度
        
        Returns:
            相关对象列表 (source_id, target_id, edge_data)
        """
        try:
            if object_id not in self.graph.nodes:
                return []
            
            related = []
            visited = set()
            
            def dfs(node_id: str, depth: int):
                if depth > max_depth or node_id in visited:
                    return
                
                visited.add(node_id)
                
                for neighbor in self.graph.neighbors(node_id):
                    edge_data = self.graph.edges[node_id, neighbor]
                    related.append((node_id, neighbor, edge_data))
                    
                    if depth < max_depth:
                        dfs(neighbor, depth + 1)
            
            dfs(object_id, 0)
            return related
            
        except Exception as e:
            logger.error(f"查找相关对象失败: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        try:
            stats = {
                'node_count': len(self.graph.nodes),
                'edge_count': len(self.graph.edges),
                'tag_count': len(self.tag_index),
                'type_count': len(self.type_index),
                'time_range_count': len(self.time_index),
                'graph_density': nx.density(self.graph),
                'is_connected': nx.is_weakly_connected(self.graph) if len(self.graph.nodes) > 1 else True
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取图统计信息失败: {e}")
            return {}
    
    async def _save_graph(self):
        """保存图到文件"""
        try:
            graph_file = self.metadata_dir / "metadata_graph.json"
            
            # 转换为可序列化的格式
            graph_data = nx.node_link_data(self.graph)
            
            with open(graph_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存图失败: {e}")
    
    async def close(self):
        """关闭元数据图"""
        try:
            await self._save_graph()
            logger.info("元数据图已关闭")
        except Exception as e:
            logger.error(f"关闭元数据图失败: {e}")