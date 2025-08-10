"""
查询解析器模块 - 解析自然语言和 SQL++ 查询
"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Union
from loguru import logger

from ..core.config import QueryConfig


class QueryParser:
    """
    查询解析器 - 解析自然语言和 SQL++ 查询
    """
    
    def __init__(self, config: QueryConfig):
        """
        初始化查询解析器
        
        Args:
            config: 查询配置
        """
        self.config = config
        self.llm_client = None  # 这里可以集成 LLM 客户端
        
        logger.info("查询解析器初始化完成")
    
    async def parse(self, query: str) -> Dict[str, Any]:
        """
        解析查询
        
        Args:
            query: 查询字符串
        
        Returns:
            解析后的查询对象
        """
        try:
            # 判断查询类型
            if self._is_sql_plus_query(query):
                return await self._parse_sql_plus(query)
            elif self._is_natural_language_query(query):
                return await self._parse_natural_language(query)
            else:
                # 默认作为自然语言查询处理
                return await self._parse_natural_language(query)
                
        except Exception as e:
            logger.error(f"解析查询失败: {e}")
            return self._create_default_query(query)
    
    def _is_sql_plus_query(self, query: str) -> bool:
        """判断是否为 SQL++ 查询"""
        # 检查是否包含 SQL 关键字
        sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'ORDER BY', 'LIMIT',
            'SEMANTIC_MATCH', 'CAPTION_CONTAINS', 'VECTOR_SIMILARITY'
        ]
        
        query_upper = query.upper()
        return any(keyword in query_upper for keyword in sql_keywords)
    
    def _is_natural_language_query(self, query: str) -> bool:
        """判断是否为自然语言查询"""
        # 检查是否包含中文或英文自然语言特征
        chinese_pattern = r'[\u4e00-\u9fff]'
        english_natural_pattern = r'\b(find|search|show|get|retrieve|display)\b'
        
        has_chinese = bool(re.search(chinese_pattern, query))
        has_english_natural = bool(re.search(english_natural_pattern, query, re.IGNORECASE))
        
        return has_chinese or has_english_natural
    
    async def _parse_sql_plus(self, query: str) -> Dict[str, Any]:
        """解析 SQL++ 查询"""
        try:
            # 这里实现 SQL++ 解析逻辑
            parsed_query = {
                'type': 'sql_plus',
                'original_query': query,
                'select_fields': self._extract_select_fields(query),
                'from_table': self._extract_from_table(query),
                'where_conditions': self._extract_where_conditions(query),
                'order_by': self._extract_order_by(query),
                'limit': self._extract_limit(query),
                'semantic_conditions': self._extract_semantic_conditions(query)
            }
            
            return parsed_query
            
        except Exception as e:
            logger.error(f"解析 SQL++ 查询失败: {e}")
            return self._create_default_query(query)
    
    async def _parse_natural_language(self, query: str) -> Dict[str, Any]:
        """解析自然语言查询"""
        try:
            # 使用 LLM 解析自然语言查询
            if self.llm_client:
                return await self._parse_with_llm(query)
            else:
                return await self._parse_with_rules(query)
                
        except Exception as e:
            logger.error(f"解析自然语言查询失败: {e}")
            return self._create_default_query(query)
    
    async def _parse_with_llm(self, query: str) -> Dict[str, Any]:
        """使用 LLM 解析查询"""
        try:
            # 构建 LLM 提示
            prompt = f"""
            请将以下自然语言查询转换为结构化的查询对象：
            
            查询：{query}
            
            请返回 JSON 格式的查询对象，包含以下字段：
            - type: 查询类型
            - data_types: 涉及的数据类型
            - semantic_conditions: 语义条件
            - filters: 过滤条件
            - time_range: 时间范围
            - limit: 结果数量限制
            """
            
            # 调用 LLM（这里需要集成具体的 LLM 客户端）
            # response = await self.llm_client.generate(prompt)
            # parsed_query = json.loads(response)
            
            # 暂时返回规则解析结果
            return await self._parse_with_rules(query)
            
        except Exception as e:
            logger.error(f"LLM 解析失败: {e}")
            return await self._parse_with_rules(query)
    
    async def _parse_with_rules(self, query: str) -> Dict[str, Any]:
        """使用规则解析查询"""
        try:
            parsed_query = {
                'type': 'natural_language',
                'original_query': query,
                'data_types': self._extract_data_types(query),
                'semantic_conditions': self._extract_semantic_conditions_nl(query),
                'filters': self._extract_filters(query),
                'time_range': self._extract_time_range(query),
                'limit': self._extract_limit_nl(query)
            }
            
            return parsed_query
            
        except Exception as e:
            logger.error(f"规则解析失败: {e}")
            return self._create_default_query(query)
    
    def _extract_select_fields(self, query: str) -> List[str]:
        """提取 SELECT 字段"""
        try:
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE)
            if select_match:
                fields_str = select_match.group(1)
                return [field.strip() for field in fields_str.split(',')]
            return ['*']
        except Exception:
            return ['*']
    
    def _extract_from_table(self, query: str) -> str:
        """提取 FROM 表名"""
        try:
            from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
            if from_match:
                return from_match.group(1)
            return 'multimodal_data'
        except Exception:
            return 'multimodal_data'
    
    def _extract_where_conditions(self, query: str) -> List[Dict[str, Any]]:
        """提取 WHERE 条件"""
        try:
            where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER BY|\s+LIMIT|$)', query, re.IGNORECASE)
            if where_match:
                conditions_str = where_match.group(1)
                return self._parse_conditions(conditions_str)
            return []
        except Exception:
            return []
    
    def _extract_order_by(self, query: str) -> List[str]:
        """提取 ORDER BY"""
        try:
            order_match = re.search(r'ORDER BY\s+(.*?)(?:\s+LIMIT|$)', query, re.IGNORECASE)
            if order_match:
                order_str = order_match.group(1)
                return [field.strip() for field in order_str.split(',')]
            return []
        except Exception:
            return []
    
    def _extract_limit(self, query: str) -> Optional[int]:
        """提取 LIMIT"""
        try:
            limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
            if limit_match:
                return int(limit_match.group(1))
            return None
        except Exception:
            return None
    
    def _extract_semantic_conditions(self, query: str) -> List[Dict[str, Any]]:
        """提取语义条件"""
        try:
            conditions = []
            
            # 提取 SEMANTIC_MATCH
            semantic_matches = re.findall(r'SEMANTIC_MATCH\s*\(\s*([^)]+)\s*,\s*"([^"]+)"\s*\)', query, re.IGNORECASE)
            for field, value in semantic_matches:
                conditions.append({
                    'type': 'semantic_match',
                    'field': field.strip(),
                    'value': value.strip()
                })
            
            # 提取 CAPTION_CONTAINS
            caption_matches = re.findall(r'CAPTION_CONTAINS\s*\(\s*"([^"]+)"\s*\)', query, re.IGNORECASE)
            for value in caption_matches:
                conditions.append({
                    'type': 'caption_contains',
                    'value': value.strip()
                })
            
            return conditions
            
        except Exception:
            return []
    
    def _extract_data_types(self, query: str) -> List[str]:
        """提取数据类型"""
        data_types = []
        
        if re.search(r'图片|照片|image', query, re.IGNORECASE):
            data_types.append('image')
        if re.search(r'视频|video', query, re.IGNORECASE):
            data_types.append('video')
        if re.search(r'音频|audio', query, re.IGNORECASE):
            data_types.append('audio')
        if re.search(r'文本|text', query, re.IGNORECASE):
            data_types.append('text')
        if re.search(r'传感器|iot', query, re.IGNORECASE):
            data_types.append('iot')
        
        return data_types if data_types else ['text', 'image', 'video', 'audio', 'iot']
    
    def _extract_semantic_conditions_nl(self, query: str) -> List[str]:
        """提取自然语言语义条件"""
        conditions = []
        
        # 提取关键词
        keywords = re.findall(r'包含|包含|有|显示|找出|搜索|查找', query)
        for keyword in keywords:
            # 提取关键词后面的内容
            pattern = f"{keyword}([^，。！？\s]+)"
            matches = re.findall(pattern, query)
            conditions.extend(matches)
        
        return conditions
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """提取过滤条件"""
        filters = {}
        
        # 提取位置信息
        location_match = re.search(r'在([^，。！？\s]+)(?:拍摄|拍摄的|的)', query)
        if location_match:
            filters['location'] = location_match.group(1)
        
        # 提取颜色信息
        color_match = re.search(r'([红黄蓝绿紫黑白灰棕]+)色', query)
        if color_match:
            filters['color'] = color_match.group(1)
        
        # 提取对象信息
        object_match = re.search(r'(汽车|狗|猫|人|建筑|动物)', query)
        if object_match:
            filters['object'] = object_match.group(1)
        
        return filters
    
    def _extract_time_range(self, query: str) -> Dict[str, str]:
        """提取时间范围"""
        time_range = {}
        
        # 提取时间信息
        time_patterns = [
            (r'上个月', 'last_month'),
            (r'上星期', 'last_week'),
            (r'昨天', 'yesterday'),
            (r'今天', 'today'),
            (r'(\d+)月(\d+)日', 'specific_date'),
            (r'(\d+)年(\d+)月', 'year_month')
        ]
        
        for pattern, time_type in time_patterns:
            match = re.search(pattern, query)
            if match:
                time_range['type'] = time_type
                time_range['value'] = match.groups() if match.groups() else match.group(0)
                break
        
        return time_range
    
    def _extract_limit_nl(self, query: str) -> Optional[int]:
        """提取自然语言查询的限制"""
        # 检查是否有数量限制
        limit_match = re.search(r'(\d+)个|(\d+)张|(\d+)条', query)
        if limit_match:
            for group in limit_match.groups():
                if group:
                    return int(group)
        
        return self.config.default_limit
    
    def _parse_conditions(self, conditions_str: str) -> List[Dict[str, Any]]:
        """解析条件字符串"""
        conditions = []
        
        # 分割条件
        condition_parts = re.split(r'\s+AND\s+|\s+OR\s+', conditions_str, flags=re.IGNORECASE)
        
        for part in condition_parts:
            part = part.strip()
            if part:
                condition = self._parse_single_condition(part)
                if condition:
                    conditions.append(condition)
        
        return conditions
    
    def _parse_single_condition(self, condition_str: str) -> Optional[Dict[str, Any]]:
        """解析单个条件"""
        try:
            # 解析各种条件类型
            if '=' in condition_str:
                field, value = condition_str.split('=', 1)
                return {
                    'type': 'equals',
                    'field': field.strip(),
                    'value': value.strip().strip('"\'')
                }
            elif '>' in condition_str:
                field, value = condition_str.split('>', 1)
                return {
                    'type': 'greater_than',
                    'field': field.strip(),
                    'value': value.strip()
                }
            elif '<' in condition_str:
                field, value = condition_str.split('<', 1)
                return {
                    'type': 'less_than',
                    'field': field.strip(),
                    'value': value.strip()
                }
            else:
                return None
        except Exception:
            return None
    
    def _create_default_query(self, query: str) -> Dict[str, Any]:
        """创建默认查询对象"""
        return {
            'type': 'natural_language',
            'original_query': query,
            'data_types': ['text', 'image', 'video', 'audio', 'iot'],
            'semantic_conditions': [query],
            'filters': {},
            'time_range': {},
            'limit': self.config.default_limit
        }
    
    async def close(self):
        """关闭查询解析器"""
        try:
            if self.llm_client:
                await self.llm_client.close()
            logger.info("查询解析器已关闭")
        except Exception as e:
            logger.error(f"关闭查询解析器失败: {e}")