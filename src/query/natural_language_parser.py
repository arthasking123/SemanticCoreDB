"""
基于模型的自然语言查询解析器 - 支持 NLQ → SQL++ 转换
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from loguru import logger

from ..core.config import QueryConfig


class NaturalLanguageParser:
    """
    基于模型的自然语言查询解析器
    
    核心特性：
    1. 模型驱动：使用 LLM 理解查询意图和实体
    2. 语义解析：深度理解查询语义和上下文
    3. SQL++ 转换：将自然语言转换为结构化查询
    4. 多模态理解：支持跨模态查询
    5. 智能推理：基于模型推理能力理解复杂查询
    """
    
    def __init__(self, config: QueryConfig, llm_client=None):
        """
        初始化自然语言解析器
        
        Args:
            config: 查询配置
            llm_client: LLM 客户端实例
        """
        self.config = config
        self.llm_client = llm_client
        
        # 解析提示模板
        self._init_prompts()
        
        logger.info("基于模型的自然语言解析器初始化完成")
    
    def _init_prompts(self):
        """初始化解析提示模板"""
        self.query_analysis_prompt = """
你是一个专业的自然语言查询解析器。请分析以下查询，理解用户的意图和需求。

查询：{query}

请返回一个结构化的 JSON 对象，包含以下信息：

1. intent: 查询意图
   - type: 查询类型 (SELECT, COUNT, SUM, AVG, MAX, MIN)
   - action: 具体动作 (find, search, show, get, analyze, compare)
   - scope: 查询范围 (all, specific, recent, latest)

2. entities: 实体列表
   - type: 实体类型 (data_type, object, location, person, event, concept)
   - value: 实体值
   - attributes: 属性列表 (如颜色、大小、状态等)
   - confidence: 置信度 (0-1)

3. time_range: 时间范围
   - type: 时间类型 (absolute, relative, specific)
   - start_date: 开始时间 (ISO格式)
   - end_date: 结束时间 (ISO格式)
   - description: 时间描述

4. filters: 过滤条件
   - type: 过滤类型 (content, metadata, semantic, numeric)
   - field: 字段名
   - operator: 操作符 (CONTAINS, EQUALS, GREATER_THAN, LESS_THAN, IN, BETWEEN)
   - value: 过滤值

5. sorting: 排序条件
   - field: 排序字段
   - order: 排序顺序 (ASC, DESC)
   - priority: 优先级

6. limit: 结果数量限制

7. context: 上下文信息
   - language: 查询语言
   - domain: 领域信息
   - user_preferences: 用户偏好

请确保返回的是有效的 JSON 格式，所有字段值都要准确反映查询的语义。
"""

        self.sql_generation_prompt = """
基于以下解析结果，生成对应的 SQL++ 查询语句：

解析结果：
{analysis_result}

数据库表结构：
- multimodal_data: 多模态数据主表
- 字段：id, content, metadata, created_at, updated_at, data_type, file_path
- 特殊函数：SEMANTIC_MATCH(content, query), CAPTION_CONTAINS(text), VECTOR_SIMILARITY(embedding, query_embedding)

请生成标准的 SQL++ 查询语句，确保：
1. 语法正确
2. 充分利用语义匹配功能
3. 合理使用过滤和排序
4. 性能优化考虑

只返回 SQL++ 语句，不要其他解释。
"""

        self.confidence_calculation_prompt = """
请评估以下查询解析结果的置信度：

查询：{original_query}
解析结果：{analysis_result}

请考虑以下因素：
1. 意图识别的准确性
2. 实体提取的完整性
3. 参数解析的合理性
4. 与原始查询的语义一致性

返回一个 0-1 之间的置信度分数，以及简短的评估说明。
"""
    
    async def parse_query(self, query: str) -> Dict[str, Any]:
        """
        使用模型解析自然语言查询
        
        Args:
            query: 自然语言查询
        
        Returns:
            解析结果
        """
        try:
            if not self.llm_client:
                logger.warning("LLM 客户端未配置，使用回退解析")
                return self._create_fallback_query(query)
            
            # 使用模型解析查询
            analysis_result = await self._analyze_query_with_model(query)
            
            # 生成 SQL++ 查询
            sql_plus = await self._generate_sql_with_model(analysis_result)
            
            # 计算置信度
            confidence = await self._calculate_confidence_with_model(query, analysis_result)
            
            return {
                'original_query': query,
                'analysis_result': analysis_result,
                'sql_plus': sql_plus,
                'confidence': confidence,
                'parsing_method': 'model_based',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"模型解析查询失败: {e}")
            return self._create_fallback_query(query)
    
    async def _analyze_query_with_model(self, query: str) -> Dict[str, Any]:
        """使用模型分析查询"""
        try:
            # 构建分析提示
            prompt = self.query_analysis_prompt.format(query=query)
            
            # 调用 LLM 进行分析
            response = await self.llm_client.generate(prompt)
            
            # 解析 JSON 响应
            try:
                analysis_result = json.loads(response)
                logger.info(f"查询分析成功: {analysis_result}")
                return analysis_result
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析失败: {e}, 响应内容: {response}")
                # 尝试修复 JSON 格式
                fixed_response = self._fix_json_response(response)
                return json.loads(fixed_response)
                
        except Exception as e:
            logger.error(f"模型分析失败: {e}")
            raise
    
    async def _generate_sql_with_model(self, analysis_result: Dict[str, Any]) -> str:
        """使用模型生成 SQL++ 查询"""
        try:
            # 构建 SQL 生成提示
            prompt = self.sql_generation_prompt.format(
                analysis_result=json.dumps(analysis_result, ensure_ascii=False, indent=2)
            )
            
            # 调用 LLM 生成 SQL
            response = await self.llm_client.generate(prompt)
            
            # 清理响应，提取 SQL 语句
            sql_plus = self._extract_sql_from_response(response)
            logger.info(f"SQL++ 生成成功: {sql_plus}")
            return sql_plus
            
        except Exception as e:
            logger.error(f"模型生成 SQL 失败: {e}")
            return self._generate_fallback_sql(analysis_result)
    
    async def _calculate_confidence_with_model(self, query: str, analysis_result: Dict[str, Any]) -> float:
        """使用模型计算置信度"""
        try:
            # 构建置信度计算提示
            prompt = self.confidence_calculation_prompt.format(
                original_query=query,
                analysis_result=json.dumps(analysis_result, ensure_ascii=False, indent=2)
            )
            
            # 调用 LLM 计算置信度
            response = await self.llm_client.generate(prompt)
            
            # 提取置信度分数
            confidence = self._extract_confidence_from_response(response)
            logger.info(f"置信度计算成功: {confidence}")
            return confidence
            
        except Exception as e:
            logger.error(f"模型计算置信度失败: {e}")
            return self._calculate_rule_based_confidence(analysis_result)
    
    def _fix_json_response(self, response: str) -> str:
        """修复 LLM 返回的 JSON 格式"""
        try:
            # 首先尝试直接解析
            try:
                json.loads(response)
                return response
            except json.JSONDecodeError:
                pass
            
            # 尝试提取 markdown 代码块中的 JSON
            if "```json" in response:
                start_marker = "```json"
                end_marker = "```"
                
                start_idx = response.find(start_marker)
                if start_idx != -1:
                    start_idx += len(start_marker)
                    end_idx = response.find(end_marker, start_idx)
                    
                    if end_idx != -1:
                        json_text = response[start_idx:end_idx].strip()
                        # 清理可能的转义字符
                        json_text = json_text.replace("\\n", "").replace("\\", "")
                        
                        # 尝试解析
                        json.loads(json_text)
                        return json_text
            
            # 查找 JSON 开始和结束位置
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end != 0:
                json_part = response[start:end]
                # 尝试解析
                json.loads(json_part)
                return json_part
            
            # 如果找不到 JSON，返回默认结构
            return '{"intent": {"type": "SELECT", "action": "find"}, "entities": [], "filters": []}'
            
        except Exception as e:
            logger.error(f"JSON 修复失败: {e}")
            return '{"intent": {"type": "SELECT", "action": "find"}, "entities": [], "filters": []}'
    
    def _extract_sql_from_response(self, response: str) -> str:
        """从 LLM 响应中提取 SQL 语句"""
        try:
            # 查找 SQL 语句
            lines = response.strip().split('\n')
            sql_lines = []
            in_sql = False
            
            for line in lines:
                line = line.strip()
                if line.upper().startswith(('SELECT', 'WITH')):
                    in_sql = True
                
                if in_sql:
                    sql_lines.append(line)
                    
                    if line.endswith(';') or line.upper().endswith(('LIMIT', 'OFFSET')):
                        break
            
            if sql_lines:
                return ' '.join(sql_lines)
            else:
                return "SELECT * FROM multimodal_data LIMIT 100"
                
        except Exception as e:
            logger.error(f"SQL 提取失败: {e}")
            return "SELECT * FROM multimodal_data LIMIT 100"
    
    def _extract_confidence_from_response(self, response: str) -> float:
        """从 LLM 响应中提取置信度分数"""
        try:
            # 查找数字
            import re
            numbers = re.findall(r'0\.\d+|\d+\.\d+|\d+', response)
            
            if numbers:
                # 取第一个 0-1 之间的数字
                for num in numbers:
                    confidence = float(num)
                    if 0 <= confidence <= 1:
                        return confidence
            
            # 如果没有找到合适的数字，返回默认值
            return 0.7
            
        except Exception as e:
            logger.error(f"置信度提取失败: {e}")
            return 0.7
    
    def _generate_fallback_sql(self, analysis_result: Dict[str, Any]) -> str:
        """生成回退 SQL 查询"""
        try:
            sql_parts = ["SELECT * FROM multimodal_data"]
            where_conditions = []
            
            # 基于分析结果构建基本查询
            intent = analysis_result.get('intent', {})
            entities = analysis_result.get('entities', [])
            filters = analysis_result.get('filters', [])
            
            # 添加实体过滤
            for entity in entities:
                if entity.get('type') == 'content':
                    where_conditions.append(f"SEMANTIC_MATCH(content, '{entity['value']}')")
                elif entity.get('type') == 'location':
                    where_conditions.append(f"metadata->>'location' = '{entity['value']}'")
            
            # 添加其他过滤条件
            for filter_cond in filters:
                if filter_cond.get('type') == 'content':
                    where_conditions.append(f"CAPTION_CONTAINS('{filter_cond['value']}')")
            
            if where_conditions:
                sql_parts.append("WHERE " + " AND ".join(where_conditions))
            
            # 添加限制
            limit = analysis_result.get('limit', self.config.default_limit)
            sql_parts.append(f"LIMIT {limit}")
            
            return " ".join(sql_parts)
            
        except Exception as e:
            logger.error(f"生成回退 SQL 失败: {e}")
            return "SELECT * FROM multimodal_data LIMIT 100"
    
    def _calculate_rule_based_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """基于规则的置信度计算（回退方案）"""
        try:
            confidence = 0.5  # 基础置信度
            
            # 意图明确性
            intent = analysis_result.get('intent', {})
            if intent.get('type'):
                confidence += 0.2
            
            # 实体丰富度
            entities = analysis_result.get('entities', [])
            if entities:
                confidence += min(len(entities) * 0.1, 0.3)
            
            # 过滤条件完整性
            filters = analysis_result.get('filters', [])
            if filters:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"规则置信度计算失败: {e}")
            return 0.5
    
    def _create_fallback_query(self, query: str) -> Dict[str, Any]:
        """创建回退查询"""
        return {
            'original_query': query,
            'analysis_result': {
                'intent': {'type': 'SELECT', 'action': 'find', 'scope': 'specific'},
                'entities': [],
                'time_range': {},
                'filters': [],
                'sorting': {},
                'limit': self.config.default_limit,
                'context': {'language': 'unknown', 'domain': 'general'}
            },
            'sql_plus': f"SELECT * FROM multimodal_data WHERE SEMANTIC_MATCH(content, '{query}') LIMIT {self.config.default_limit}",
            'confidence': 0.3,
            'parsing_method': 'fallback',
            'timestamp': datetime.now().isoformat()
        }
    
    async def parse_with_context(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """带上下文的查询解析"""
        try:
            if context:
                # 将上下文信息添加到查询中
                enhanced_query = self._enhance_query_with_context(query, context)
                return await self.parse_query(enhanced_query)
            else:
                return await self.parse_query(query)
                
        except Exception as e:
            logger.error(f"上下文解析失败: {e}")
            return await self.parse_query(query)
    
    def _enhance_query_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """使用上下文信息增强查询"""
        try:
            enhanced_parts = [query]
            
            # 添加时间上下文
            if context.get('current_time'):
                enhanced_parts.append(f"当前时间: {context['current_time']}")
            
            # 添加用户偏好
            if context.get('user_preferences'):
                prefs = context['user_preferences']
                if prefs.get('language'):
                    enhanced_parts.append(f"语言偏好: {prefs['language']}")
                if prefs.get('domain'):
                    enhanced_parts.append(f"领域: {prefs['domain']}")
            
            # 添加会话历史
            if context.get('conversation_history'):
                history = context['conversation_history'][-3:]  # 最近3条
                enhanced_parts.append(f"会话上下文: {'; '.join(history)}")
            
            return " | ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"查询增强失败: {e}")
            return query
    
    async def batch_parse(self, queries: List[str]) -> List[Dict[str, Any]]:
        """批量解析查询"""
        try:
            results = []
            for query in queries:
                result = await self.parse_query(query)
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"批量解析失败: {e}")
            return [self._create_fallback_query(q) for q in queries]
    
    async def close(self):
        """关闭解析器"""
        try:
            if self.llm_client:
                await self.llm_client.close()
            logger.info("自然语言解析器已关闭")
        except Exception as e:
            logger.error(f"关闭自然语言解析器失败: {e}") 