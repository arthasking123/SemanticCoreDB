"""
LLM 服务测试模块
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from loguru import logger

from src.semantic.llm_service import LLMService
from src.semantic.llm_config import LLMConfigManager
from src.semantic.llm_factory import LLMClientFactory


class TestLLMService:
    """LLM 服务测试类"""
    
    @pytest.fixture
    async def mock_service(self):
        """创建模拟服务实例"""
        with patch('src.semantic.llm_service.LLMConfigManager') as mock_config_manager:
            with patch('src.semantic.llm_service.LLMClientFactory') as mock_factory:
                # 模拟配置管理器
                mock_config = Mock()
                mock_config.provider = "openai"
                mock_config.get_available_providers.return_value = ["openai"]
                mock_config.get_provider_config.return_value = {"api_key": "test"}
                
                mock_config_manager.return_value = mock_config
                
                # 模拟工厂
                mock_factory_instance = Mock()
                mock_factory_instance.initialize = AsyncMock()
                mock_factory_instance.generate_with_fallback = AsyncMock(return_value="测试响应")
                mock_factory_instance.get_client_info.return_value = {"active_client": "openai"}
                mock_factory_instance.close_all = AsyncMock()
                
                mock_factory.return_value = mock_factory_instance
                
                service = LLMService('config/llm_config.yaml')
                yield service
                
                await service.close()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_service):
        """测试服务初始化"""
        await mock_service.initialize()
        assert mock_service.is_initialized == True
    
    @pytest.mark.asyncio
    async def test_generate_text(self, mock_service):
        """测试文本生成"""
        await mock_service.initialize()
        
        response = await mock_service.generate_text("测试提示")
        assert response == "测试响应"
    
    @pytest.mark.asyncio
    async def test_analyze_query(self, mock_service):
        """测试查询分析"""
        await mock_service.initialize()
        
        # 模拟LLM响应
        mock_service.factory.generate_with_fallback = AsyncMock(return_value='''
        {
            "intent": {
                "type": "SELECT",
                "action": "find",
                "scope": "all"
            },
            "entities": [],
            "filters": [],
            "sorting": [],
            "limit": 100
        }
        ''')
        
        result = await mock_service.analyze_query("查找所有图片")
        
        assert "intent" in result
        assert result["intent"]["type"] == "SELECT"
        assert result["intent"]["action"] == "find"
    
    @pytest.mark.asyncio
    async def test_generate_sql(self, mock_service):
        """测试SQL生成"""
        await mock_service.initialize()
        
        # 模拟LLM响应
        mock_service.factory.generate_with_fallback = AsyncMock(return_value='''
        基于分析结果，生成以下SQL查询：
        
        ```sql
        SELECT id, content, metadata, created_at, data_type, file_path
        FROM multimodal_data
        WHERE data_type = 'image'
        LIMIT 100
        ```
        ''')
        
        analysis_result = {
            "intent": {"type": "SELECT", "action": "find"},
            "entities": [{"type": "data_type", "value": "image"}]
        }
        
        sql = await mock_service.generate_sql(analysis_result)
        
        assert "SELECT" in sql
        assert "FROM multimodal_data" in sql
        assert "WHERE data_type = 'image'" in sql
    
    @pytest.mark.asyncio
    async def test_calculate_confidence(self, mock_service):
        """测试置信度计算"""
        await mock_service.initialize()
        
        # 模拟LLM响应
        mock_service.factory.generate_with_fallback = AsyncMock(return_value="0.85")
        
        analysis_result = {
            "intent": {"type": "SELECT"},
            "entities": [{"type": "data_type", "value": "image"}]
        }
        
        confidence = await mock_service.calculate_confidence("查找图片", analysis_result)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence == 0.85
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, mock_service):
        """测试批量处理"""
        await mock_service.initialize()
        
        # 模拟LLM响应
        mock_service.factory.generate_with_fallback = AsyncMock(return_value='''
        {
            "intent": {"type": "SELECT", "action": "find"},
            "entities": [],
            "filters": [],
            "sorting": [],
            "limit": 100
        }
        ''')
        
        queries = ["查询1", "查询2"]
        results = await mock_service.batch_process(queries)
        
        assert len(results) == 2
        assert all("query" in result for result in results)
        assert all("timestamp" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, mock_service):
        """测试语义搜索"""
        await mock_service.initialize()
        
        # 模拟LLM响应
        mock_service.factory.generate_with_fallback = AsyncMock(return_value='''
        {
            "search_strategy": "语义匹配",
            "keywords": ["自然语言", "处理"],
            "filters": ["document", "image"]
        }
        ''')
        
        result = await mock_service.semantic_search("查找自然语言处理相关文档")
        
        assert "search_strategy" in result
        assert "keywords" in result
    
    @pytest.mark.asyncio
    async def test_service_status(self, mock_service):
        """测试服务状态获取"""
        await mock_service.initialize()
        
        status = await mock_service.get_service_status()
        
        assert "initialized" in status
        assert "clients" in status
        assert "config" in status
        assert status["initialized"] == True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_service):
        """测试错误处理"""
        await mock_service.initialize()
        
        # 模拟LLM错误
        mock_service.factory.generate_with_fallback = AsyncMock(side_effect=Exception("LLM服务错误"))
        
        # 测试错误情况下的降级处理
        try:
            await mock_service.generate_text("测试")
            pytest.fail("应该抛出异常")
        except Exception as e:
            assert "LLM服务错误" in str(e)
    
    @pytest.mark.asyncio
    async def test_json_parsing_fallback(self, mock_service):
        """测试JSON解析降级"""
        await mock_service.initialize()
        
        # 模拟无效JSON响应
        mock_service.factory.generate_with_fallback = AsyncMock(return_value="这不是有效的JSON")
        
        result = await mock_service.analyze_query("测试查询")
        
        # 应该返回默认分析结果
        assert "intent" in result
        assert result["intent"]["type"] == "SELECT"
    
    @pytest.mark.asyncio
    async def test_sql_extraction_fallback(self, mock_service):
        """测试SQL提取降级"""
        await mock_service.initialize()
        
        # 模拟没有SQL代码块的响应
        mock_service.factory.generate_with_fallback = AsyncMock(return_value="这里没有SQL语句")
        
        analysis_result = {"intent": {"type": "SELECT"}}
        sql = await mock_service.generate_sql(analysis_result)
        
        # 应该返回降级SQL
        assert "SELECT" in sql
        assert "FROM multimodal_data" in sql


class TestLLMConfigManager:
    """LLM 配置管理器测试类"""
    
    def test_config_creation_from_env(self):
        """测试从环境变量创建配置"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'anthropic',
            'ANTHROPIC_API_KEY': 'test_key',
            'ANTHROPIC_MODEL': 'claude-test'
        }):
            config_manager = LLMConfigManager()
            
            assert config_manager.config.provider == "anthropic"
            assert config_manager.config.anthropic is not None
            assert config_manager.config.anthropic.api_key == "test_key"
            assert config_manager.config.anthropic.model == "claude-test"
    
    def test_get_available_providers(self):
        """测试获取可用提供商"""
        config_manager = LLMConfigManager()
        
        # 模拟配置
        config_manager.config.openai = Mock()
        config_manager.config.anthropic = Mock()
        
        providers = config_manager.get_available_providers()
        
        assert "openai" in providers
        assert "anthropic" in providers
        assert len(providers) == 2
    
    def test_get_fallback_provider(self):
        """测试获取降级提供商"""
        config_manager = LLMConfigManager()
        config_manager.config.enable_fallback = True
        
        # 模拟配置
        config_manager.config.openai = Mock()
        config_manager.config.anthropic = Mock()
        
        fallback = config_manager.get_fallback_provider("openai")
        assert fallback == "anthropic"
        
        fallback = config_manager.get_fallback_provider("anthropic")
        assert fallback == "openai"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])