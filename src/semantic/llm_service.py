"""
LLM 服务模块 - 提供高级的 LLM 功能接口
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from loguru import logger

from .llm_factory import LLMClientFactory
from .llm_config import LLMConfigManager


class LLMService:
    """LLM 服务类 - 提供高级的 LLM 功能接口"""
    
    def __init__(self, config_path: Optional[str] = None, default_config=None):
        """
        初始化LLM服务
        
        Args:
            config_path: 配置文件路径（llm_config.yaml格式）
            default_config: 从default.yaml加载的配置对象
        """
        if default_config is not None:
            # 使用default.yaml配置
            self.config_manager = LLMConfigManager.from_default_config(default_config)
        else:
            # 使用llm_config.yaml配置
            self.config_manager = LLMConfigManager(config_path)
        
        self.factory = LLMClientFactory(self.config_manager)
        self.is_initialized = False
        
        logger.info("LLM 服务初始化完成")
    
    async def initialize(self):
        """初始化服务"""
        if not self.is_initialized:
            await self.factory.initialize()
            self.is_initialized = True
            logger.info("LLM 服务已初始化")
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        生成文本响应
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
        
        Returns:
            生成的文本响应
        """
        await self.initialize()
        return await self.factory.generate_with_fallback(prompt, **kwargs)
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        分析自然语言查询
        
        Args:
            query: 自然语言查询
        
        Returns:
            查询分析结果
        """
        # 获取配置的提示模板
        prompts_config = self.config_manager.get_prompts_config()
        if prompts_config and prompts_config.query_analysis:
            prompt = prompts_config.query_analysis.format(query=query)
        else:
            # 使用默认提示模板
            prompt = f"""
请分析以下自然语言查询，理解用户的意图和需求：

查询：{query}

请返回一个结构化的 JSON 对象，包含以下信息：

1. intent: 查询意图
   - type: 查询类型 (SELECT, COUNT, SUM, AVG, MAX, MIN)
   - action: 具体动作 (find, search, show, get, analyze, compare)
   - scope: 查询范围 (all, specific, recent, latest)

2. entities: 实体列表
   - type: 实体类型 (data_type, object, location, person, event, concept)
   - value: 实体值
   - attributes: 属性列表
   - confidence: 置信度 (0-1)

3. time_range: 时间范围
   - type: 时间类型 (absolute, relative, specific)
   - start_date: 开始时间
   - end_date: 结束时间
   - description: 时间描述

4. filters: 过滤条件
   - type: 过滤类型 (content, metadata, semantic, numeric)
   - field: 字段名
   - operator: 操作符
   - value: 过滤值

5. sorting: 排序条件
   - field: 排序字段
   - order: 排序顺序 (ASC, DESC)

6. limit: 结果数量限制

请确保返回的是有效的 JSON 格式。
"""
        
        try:
            response = await self.generate_text(prompt, max_tokens=1000)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            return self._create_fallback_analysis(query)
    
    async def generate_sql(self, analysis_result: Dict[str, Any]) -> str:
        """
        基于分析结果生成SQL查询
        
        Args:
            analysis_result: 查询分析结果
        
        Returns:
            SQL查询语句
        """
        # 获取配置的提示模板
        prompts_config = self.config_manager.get_prompts_config()
        if prompts_config and prompts_config.sql_generation:
            prompt = prompts_config.sql_generation.format(
                analysis_result=json.dumps(analysis_result, ensure_ascii=False, indent=2)
            )
        else:
            # 使用默认提示模板
            prompt = f"""
基于以下解析结果，生成对应的 SQL++ 查询语句：

解析结果：
{json.dumps(analysis_result, ensure_ascii=False, indent=2)}

数据库表结构：
- multimodal_data: 多模态数据主表
- 字段：id, content, metadata, created_at, updated_at, data_type, file_path
- 特殊函数：SEMANTIC_MATCH(content, query), CAPTION_CONTAINS(text), VECTOR_SIMILARITY(embedding, query_embedding)

请生成标准的 SQL++ 查询语句，确保语法正确并充分利用语义匹配功能。
"""
        
        try:
            response = await self.generate_text(prompt, max_tokens=1000)
            return self._extract_sql_from_response(response)
        except Exception as e:
            logger.error(f"SQL生成失败: {e}")
            return self._generate_fallback_sql(analysis_result)
    
    async def calculate_confidence(self, query: str, analysis_result: Dict[str, Any]) -> float:
        """
        计算查询解析的置信度
        
        Args:
            query: 原始查询
            analysis_result: 解析结果
        
        Returns:
            置信度分数 (0-1)
        """
        # 获取配置的提示模板
        prompts_config = self.config_manager.get_prompts_config()
        if prompts_config and prompts_config.confidence_calculation:
            prompt = prompts_config.confidence_calculation.format(
                original_query=query,
                analysis_result=json.dumps(analysis_result, ensure_ascii=False, indent=2)
            )
        else:
            # 使用默认提示模板
            prompt = f"""
请评估以下查询解析结果的置信度：

查询：{query}
解析结果：{json.dumps(analysis_result, ensure_ascii=False, indent=2)}

请考虑以下因素：
1. 意图识别的准确性
2. 实体提取的完整性
3. 参数解析的合理性
4. 与原始查询的语义一致性

返回一个 0-1 之间的置信度分数，只需要返回数字。
"""
        
        try:
            response = await self.generate_text(prompt, max_tokens=100)
            return self._extract_confidence_from_response(response)
        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return self._calculate_rule_based_confidence(analysis_result)
    
    async def batch_process(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        批量处理查询
        
        Args:
            queries: 查询列表
        
        Returns:
            处理结果列表
        """
        results = []
        
        for query in queries:
            try:
                analysis = await self.analyze_query(query)
                sql = await self.generate_sql(analysis)
                confidence = await self.calculate_confidence(query, analysis)
                
                result = {
                    'query': query,
                    'analysis': analysis,
                    'sql': sql,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"处理查询失败: {query}, 错误: {e}")
                results.append({
                    'query': query,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    async def semantic_search(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        语义搜索
        
        Args:
            query: 搜索查询
            context: 上下文信息
        
        Returns:
            搜索结果
        """
        # 构建增强的搜索提示
        enhanced_prompt = self._enhance_search_prompt(query, context)
        
        try:
            response = await self.generate_text(enhanced_prompt, max_tokens=1500)
            return self._parse_search_response(response)
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return {'error': str(e), 'query': query}
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析JSON响应"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            try:
                # 查找JSON开始和结束位置
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    return json.loads(json_str)
            except:
                pass
            
            # 如果都失败，返回默认结果
            logger.warning("无法解析JSON响应，使用默认结果")
            return self._create_default_analysis()
    
    def _extract_sql_from_response(self, response: str) -> str:
        """从响应中提取SQL语句"""
        # 查找SQL代码块
        sql_start = response.find('```sql')
        if sql_start >= 0:
            sql_start = response.find('\n', sql_start) + 1
            sql_end = response.find('```', sql_start)
            if sql_end > sql_start:
                return response[sql_start:sql_end].strip()
        
        # 如果没有代码块，尝试查找SELECT语句
        import re
        select_match = re.search(r'SELECT.*?(?:;|$)', response, re.IGNORECASE | re.DOTALL)
        if select_match:
            return select_match.group().strip()
        
        return response.strip()
    
    def _extract_confidence_from_response(self, response: str) -> float:
        """从响应中提取置信度分数"""
        try:
            # 查找数字
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                confidence = float(numbers[0])
                return max(0.0, min(1.0, confidence))  # 限制在0-1范围内
        except:
            pass
        
        return 0.5  # 默认置信度
    
    def _create_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """创建降级分析结果"""
        # 获取配置的默认值
        query_parsing_config = self.config_manager.get_query_parsing_config()
        default_limit = query_parsing_config.default_limit if query_parsing_config else 100
        
        return {
            'intent': {
                'type': 'SELECT',
                'action': 'find',
                'scope': 'all'
            },
            'entities': [],
            'time_range': None,
            'filters': [],
            'sorting': [],
            'limit': default_limit,
            'confidence': 0.3
        }
    
    def _generate_fallback_sql(self, analysis_result: Dict[str, Any]) -> str:
        """生成降级SQL查询"""
        return """
SELECT id, content, metadata, created_at, data_type, file_path
FROM multimodal_data
WHERE 1=1
LIMIT 100
        """.strip()
    
    def _calculate_rule_based_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """基于规则的置信度计算"""
        confidence = 0.5  # 基础置信度
        
        # 检查意图完整性
        if 'intent' in analysis_result and 'type' in analysis_result['intent']:
            confidence += 0.1
        
        # 检查实体数量
        if 'entities' in analysis_result and len(analysis_result['entities']) > 0:
            confidence += 0.1
        
        # 检查过滤条件
        if 'filters' in analysis_result and len(analysis_result['filters']) > 0:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _create_default_analysis(self) -> Dict[str, Any]:
        """创建默认分析结果"""
        # 获取配置的默认值
        query_parsing_config = self.config_manager.get_query_parsing_config()
        default_limit = query_parsing_config.default_limit if query_parsing_config else 100
        
        return {
            'intent': {'type': 'SELECT', 'action': 'find', 'scope': 'all'},
            'entities': [],
            'time_range': None,
            'filters': [],
            'sorting': [],
            'limit': default_limit
        }
    
    def _enhance_search_prompt(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """增强搜索提示"""
        base_prompt = f"请为以下查询提供详细的语义搜索建议：\n\n查询：{query}"
        
        if context:
            context_str = json.dumps(context, ensure_ascii=False, indent=2)
            base_prompt += f"\n\n上下文信息：\n{context_str}"
        
        base_prompt += "\n\n请提供：\n1. 搜索策略建议\n2. 关键词提取\n3. 语义匹配方法\n4. 过滤条件建议"
        
        return base_prompt
    
    def _parse_search_response(self, response: str) -> Dict[str, Any]:
        """解析搜索响应"""
        try:
            # 尝试解析为结构化响应
            return json.loads(response)
        except:
            # 返回原始响应
            return {'response': response, 'type': 'text'}
    
    async def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        await self.initialize()
        
        status = {
            'initialized': self.is_initialized,
            'clients': self.factory.get_client_info(),
            'config': {
                'provider': self.config_manager.config.provider,
                'available_providers': self.config_manager.get_available_providers(),
                'retry_attempts': self.config_manager.config.retry_attempts,
                'enable_fallback': self.config_manager.config.enable_fallback
            }
        }
        
        return status
    
    async def close(self):
        """关闭服务"""
        await self.factory.close_all()
        self.is_initialized = False
        logger.info("LLM 服务已关闭")