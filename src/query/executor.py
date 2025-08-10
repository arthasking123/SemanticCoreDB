"""
查询执行器模块 - 执行解析后的查询
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger

from ..storage.object_store import ObjectStore
from ..semantic.vector_index import VectorIndex
from ..core.metadata_graph import MetadataGraph
from ..core.config import QueryConfig


class QueryExecutor:
    """
    查询执行器 - 执行解析后的查询
    """
    
    def __init__(self, 
                 object_store: ObjectStore,
                 vector_index: VectorIndex,
                 metadata_graph: MetadataGraph,
                 query_config: QueryConfig,
                 semantic_config=None,
                 embedding_service=None):
        """
        初始化查询执行器
        
        Args:
            object_store: 对象存储
            vector_index: 向量索引
            metadata_graph: 元数据图
            query_config: 查询配置
            semantic_config: 语义配置
            embedding_service: 嵌入服务
        """
        self.object_store = object_store
        self.vector_index = vector_index
        self.metadata_graph = metadata_graph
        self.config = query_config
        self.semantic_config = semantic_config
        self.embedding_service = embedding_service
        
        logger.info("查询执行器初始化完成")
    
    async def execute(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        执行查询
        
        Args:
            parsed_query: 解析后的查询对象
        
        Returns:
            查询结果列表
        """
        try:
            query_type = parsed_query.get('type', 'natural_language')
            
            if query_type == 'sql_plus':
                return await self._execute_sql_plus(parsed_query)
            else:
                return await self._execute_natural_language(parsed_query)
                
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            return []
    
    async def _execute_sql_plus(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行 SQL++ 查询"""
        try:
            # 获取基础数据
            base_objects = await self._get_base_objects(parsed_query)
            
            # 应用 WHERE 条件
            filtered_objects = await self._apply_where_conditions(base_objects, parsed_query)
            
            # 应用语义条件
            semantic_objects = await self._apply_semantic_conditions(filtered_objects, parsed_query)
            
            # 应用 ORDER BY
            ordered_objects = await self._apply_order_by(semantic_objects, parsed_query)
            
            # 应用 LIMIT
            limited_objects = await self._apply_limit(ordered_objects, parsed_query)
            
            # 选择字段
            results = await self._select_fields(limited_objects, parsed_query)
            
            return results
            
        except Exception as e:
            logger.error(f"执行 SQL++ 查询失败: {e}")
            return []
    
    async def _execute_natural_language(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行自然语言查询"""
        try:
            # 获取候选对象
            candidate_objects = await self._get_candidate_objects(parsed_query)
            
            # 应用语义搜索
            semantic_results = await self._apply_semantic_search(candidate_objects, parsed_query)
            
            # 应用过滤条件
            filtered_results = await self._apply_filters(semantic_results, parsed_query)
            
            # 应用时间范围
            time_filtered_results = await self._apply_time_range(filtered_results, parsed_query)
            
            # 应用限制
            limited_results = await self._apply_limit(time_filtered_results, parsed_query)
            
            return limited_results
            
        except Exception as e:
            logger.error(f"执行自然语言查询失败: {e}")
            return []
    
    async def _get_base_objects(self, parsed_query: Dict[str, Any]) -> List[str]:
        """获取基础对象列表"""
        try:
            from_table = parsed_query.get('from_table', 'multimodal_data')
            
            # 根据表名获取对象
            if from_table == 'multimodal_data':
                # 获取所有对象
                return list(self.object_store.vectors.keys())
            else:
                # 根据类型过滤
                return await self.metadata_graph.find_objects_by_type(from_table)
                
        except Exception as e:
            logger.error(f"获取基础对象失败: {e}")
            return []
    
    async def _get_candidate_objects(self, parsed_query: Dict[str, Any]) -> List[str]:
        """获取候选对象列表"""
        try:
            # 尝试从 entities 字段中提取数据类型
            entities = parsed_query.get('entities', [])
            data_types = []
            
            for entity in entities:
                if entity.get('type') == 'data_type':
                    data_types.append(entity.get('value'))
            
            # 如果没有找到数据类型，尝试从 data_types 字段获取
            if not data_types:
                data_types = parsed_query.get('data_types', [])
            
            candidate_objects = []
            
            if data_types:
                # 根据数据类型过滤对象
                for data_type in data_types:
                    objects = await self.metadata_graph.find_objects_by_type(data_type)
                    candidate_objects.extend(objects)
            else:
                # 如果没有指定数据类型，获取所有对象
                # 优先从向量索引中获取所有对象ID
                if hasattr(self.vector_index, 'vectors') and self.vector_index.vectors:
                    candidate_objects = list(self.vector_index.vectors.keys())
                else:
                    # 如果没有向量索引，从元数据图中获取所有对象
                    candidate_objects = await self.metadata_graph.get_all_objects()
                
                # 如果还是没有找到对象，尝试从对象存储获取
                if not candidate_objects and hasattr(self.object_store, 'object_index'):
                    candidate_objects = list(self.object_store.object_index.keys())
            
            result = list(set(candidate_objects))  # 去重
            return result
            
        except Exception as e:
            logger.error(f"获取候选对象失败: {e}")
            return []
    
    async def _apply_where_conditions(self, objects: List[str], parsed_query: Dict[str, Any]) -> List[str]:
        """应用 WHERE 条件"""
        try:
            conditions = parsed_query.get('where_conditions', [])
            filtered_objects = objects
            
            for condition in conditions:
                filtered_objects = await self._apply_single_condition(filtered_objects, condition)
            
            return filtered_objects
            
        except Exception as e:
            logger.error(f"应用 WHERE 条件失败: {e}")
            return objects
    
    async def _apply_single_condition(self, objects: List[str], condition: Dict[str, Any]) -> List[str]:
        """应用单个条件"""
        try:
            condition_type = condition.get('type')
            
            if condition_type == 'equals':
                return await self._apply_equals_condition(objects, condition)
            elif condition_type == 'greater_than':
                return await self._apply_greater_than_condition(objects, condition)
            elif condition_type == 'less_than':
                return await self._apply_less_than_condition(objects, condition)
            else:
                return objects
                
        except Exception as e:
            logger.error(f"应用单个条件失败: {e}")
            return objects
    
    async def _apply_equals_condition(self, objects: List[str], condition: Dict[str, Any]) -> List[str]:
        """应用等于条件"""
        try:
            field = condition.get('field')
            value = condition.get('value')
            
            filtered_objects = []
            for object_id in objects:
                object_data = await self.object_store.get_object(object_id)
                if object_data and object_data.get('metadata', {}).get(field) == value:
                    filtered_objects.append(object_id)
            
            return filtered_objects
            
        except Exception as e:
            logger.error(f"应用等于条件失败: {e}")
            return objects
    
    async def _apply_greater_than_condition(self, objects: List[str], condition: Dict[str, Any]) -> List[str]:
        """应用大于条件"""
        try:
            field = condition.get('field')
            value = condition.get('value')
            
            filtered_objects = []
            for object_id in objects:
                object_data = await self.object_store.get_object(object_id)
                if object_data:
                    field_value = object_data.get('metadata', {}).get(field)
                    if field_value and field_value > value:
                        filtered_objects.append(object_id)
            
            return filtered_objects
            
        except Exception as e:
            logger.error(f"应用大于条件失败: {e}")
            return objects
    
    async def _apply_less_than_condition(self, objects: List[str], condition: Dict[str, Any]) -> List[str]:
        """应用小于条件"""
        try:
            field = condition.get('field')
            value = condition.get('value')
            
            filtered_objects = []
            for object_id in objects:
                object_data = await self.object_store.get_object(object_id)
                if object_data:
                    field_value = object_data.get('metadata', {}).get(field)
                    if field_value and field_value < value:
                        filtered_objects.append(object_id)
            
            return filtered_objects
            
        except Exception as e:
            logger.error(f"应用小于条件失败: {e}")
            return objects
    
    async def _apply_semantic_conditions(self, objects: List[str], parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用语义条件"""
        try:
            semantic_conditions = parsed_query.get('semantic_conditions', [])
            results = []
            
            for object_id in objects:
                object_data = await self.object_store.get_object(object_id)
                if not object_data:
                    continue
                
                # 检查语义条件
                semantic_score = await self._calculate_semantic_score(object_data, semantic_conditions)
                
                if semantic_score > 0:
                    results.append({
                        'object_id': object_id,
                        'object_data': object_data,
                        'semantic_score': semantic_score
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"应用语义条件失败: {e}")
            return []
    
    async def _apply_semantic_search(self, objects: List[str], parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用语义搜索"""
        try:
            results = []
            
            # 检查是否有语义条件
            semantic_conditions = parsed_query.get('semantic_conditions', [])
            
            # 检查是否有内容过滤条件（用于语义搜索）
            filters = parsed_query.get('filters', [])
            content_filters = []
            
            for filter_condition in filters:
                if isinstance(filter_condition, dict):
                    filter_type = filter_condition.get('type', '')
                    filter_value = filter_condition.get('value', '')
                    if filter_type == 'content' and filter_value:
                        content_filters.append(filter_value)
            
            # 合并所有需要语义搜索的条件
            all_search_terms = semantic_conditions + content_filters
            
            if not all_search_terms:
                # 如果没有搜索条件，返回所有对象
                for object_id in objects:
                    object_data = await self.object_store.get_object(object_id)
                    if object_data:
                        results.append({
                            'object_id': object_id,
                            'object_data': object_data,
                            'similarity': 1.0
                        })
                return results
            
            # 对每个搜索词进行语义搜索
            for search_term in all_search_terms:
                # 生成查询向量
                query_vector = await self._generate_query_vector(search_term)
                
                # 搜索相似向量
                threshold = self.semantic_config.semantic_threshold if self.semantic_config else 0.5
                
                similar_objects = await self.vector_index.search_similar(
                    query_vector, 
                    k=50, 
                    threshold=threshold
                )
                
                for object_id, similarity in similar_objects:
                    # 检查对象是否在候选列表中
                    if object_id in objects:
                        # 检查是否已经添加过这个对象
                        existing_result = next((r for r in results if r['object_id'] == object_id), None)
                        if existing_result:
                            # 如果已存在，更新相似度分数（取最高分）
                            existing_result['similarity'] = max(existing_result['similarity'], similarity)
                        else:
                            # 如果不存在，添加新结果
                            object_data = await self.object_store.get_object(object_id)
                            if object_data:
                                results.append({
                                    'object_id': object_id,
                                    'object_data': object_data,
                                    'similarity': similarity
                                })
                    else:
                        # 如果候选对象列表为空，可能是获取候选对象时出现了问题
                        if not objects:
                            # 在这种情况下，我们可以考虑将搜索结果直接添加到结果中
                            # 但需要先验证对象是否存在
                            object_data = await self.object_store.get_object(object_id)
                            if object_data:
                                results.append({
                                    'object_id': object_id,
                                    'object_data': object_data,
                                    'similarity': similarity
                                })
            
            return results
            
        except Exception as e:
            logger.error(f"应用语义搜索失败: {e}")
            return []
    
    async def _generate_query_vector(self, query_text: str) -> np.ndarray:
        """生成查询向量"""
        try:
            # 使用嵌入服务生成向量
            if hasattr(self, 'embedding_service') and self.embedding_service:
                # 创建临时的数据对象来生成嵌入
                temp_data = {
                    "type": "text",
                    "data": query_text
                }
                vector = await self.embedding_service.generate_embedding(temp_data)
                return vector
            else:
                # 如果没有嵌入服务，返回随机向量
                import numpy as np
                vector = np.random.rand(384)  # 默认维度
                return vector
                
        except Exception as e:
            logger.error(f"生成查询向量失败: {e}")
            # 返回随机向量作为后备
            import numpy as np
            vector = np.random.rand(384)
            return vector
            
        except Exception as e:
            logger.error(f"生成查询向量失败: {e}")
            vector = np.zeros(self.vector_index.embedding_dimension)
            return vector
    
    async def _calculate_semantic_score(self, object_data: Dict[str, Any], conditions: List[str]) -> float:
        """计算语义分数"""
        try:
            # 简单的关键词匹配
            object_text = str(object_data.get('data', ''))
            score = 0.0
            
            for condition in conditions:
                if condition.lower() in object_text.lower():
                    score += 1.0
            
            return score / len(conditions) if conditions else 0.0
            
        except Exception as e:
            logger.error(f"计算语义分数失败: {e}")
            return 0.0
    
    async def _apply_filters(self, results: List[Dict[str, Any]], parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用过滤条件"""
        try:
            filters = parsed_query.get('filters', [])
            filtered_results = results
            
            # 处理不同类型的过滤条件
            if isinstance(filters, dict):
                # 如果是字典格式，按原来的方式处理
                for filter_type, filter_value in filters.items():
                    filtered_results = await self._apply_single_filter(filtered_results, filter_type, filter_value)
            elif isinstance(filters, list):
                # 如果是列表格式，处理每个过滤条件
                for filter_condition in filters:
                    if isinstance(filter_condition, dict):
                        filter_type = filter_condition.get('type', '')
                        filter_value = filter_condition.get('value', '')
                        if filter_type and filter_value:
                            filtered_results = await self._apply_single_filter(filtered_results, filter_type, filter_value)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"应用过滤条件失败: {e}")
            return results
    
    async def _apply_single_filter(self, results: List[Dict[str, Any]], filter_type: str, filter_value: str) -> List[Dict[str, Any]]:
        """应用单个过滤条件"""
        try:
            filtered_results = []
            
            for result in results:
                object_data = result['object_data']
                
                if filter_type == 'location':
                    if filter_value in str(object_data.get('metadata', {}).get('location', '')):
                        filtered_results.append(result)
                        
                elif filter_type == 'color':
                    if filter_value in str(object_data.get('data', '')):
                        filtered_results.append(result)
                        
                elif filter_type == 'object':
                    if filter_value in str(object_data.get('data', '')):
                        filtered_results.append(result)
                        
                else:
                    # 默认包含
                    filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"应用单个过滤条件失败: {e}")
            return results
    
    async def _apply_time_range(self, results: List[Dict[str, Any]], parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用时间范围过滤"""
        try:
            time_range = parsed_query.get('time_range', {})
            if not time_range:
                return results
            
            filtered_results = []
            
            for result in results:
                object_data = result['object_data']
                created_at = object_data.get('created_at', '')
                
                if await self._is_in_time_range(created_at, time_range):
                    filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"应用时间范围失败: {e}")
            return results
    
    async def _is_in_time_range(self, created_at: str, time_range: Dict[str, Any]) -> bool:
        """检查是否在时间范围内"""
        try:
            # 这里实现时间范围检查逻辑
            # 暂时返回 True
            return True
            
        except Exception as e:
            logger.error(f"检查时间范围失败: {e}")
            return True
    
    async def _apply_order_by(self, results: List[Dict[str, Any]], parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用排序"""
        try:
            order_by = parsed_query.get('order_by', [])
            
            if not order_by:
                # 默认按相似度排序
                return sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)
            
            # 这里实现多字段排序逻辑
            return results
            
        except Exception as e:
            logger.error(f"应用排序失败: {e}")
            return results
    
    async def _apply_limit(self, results: List[Dict[str, Any]], parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用限制"""
        try:
            limit = parsed_query.get('limit', self.config.default_limit)
            
            if limit and len(results) > limit:
                return results[:limit]
            
            return results
            
        except Exception as e:
            logger.error(f"应用限制失败: {e}")
            return results
    
    async def _select_fields(self, results: List[Dict[str, Any]], parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """选择字段"""
        try:
            select_fields = parsed_query.get('select_fields', ['*'])
            
            if '*' in select_fields:
                return results
            
            # 这里实现字段选择逻辑
            return results
            
        except Exception as e:
            logger.error(f"选择字段失败: {e}")
            return results
    
    async def close(self):
        """关闭查询执行器"""
        try:
            logger.info("查询执行器已关闭")
        except Exception as e:
            logger.error(f"关闭查询执行器失败: {e}")