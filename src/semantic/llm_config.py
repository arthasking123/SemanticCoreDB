"""
LLM 配置管理模块 - 统一管理各种 LLM 客户端的配置
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger


class OpenAIConfig(BaseModel):
    """OpenAI 配置"""
    model: str = Field(default="gpt-3.5-turbo", description="模型名称")
    api_key: str = Field(description="API密钥")
    max_tokens: int = Field(default=1000, description="最大token数")
    temperature: float = Field(default=0.1, description="温度参数")
    timeout: int = Field(default=30, description="超时时间(秒)")


class AnthropicConfig(BaseModel):
    """Anthropic 配置"""
    model: str = Field(default="claude-3-sonnet-20240229", description="模型名称")
    api_key: str = Field(description="API密钥")
    max_tokens: int = Field(default=1000, description="最大token数")
    timeout: int = Field(default=30, description="超时时间(秒)")


class LocalLLMConfig(BaseModel):
    """本地 LLM 配置"""
    base_url: str = Field(description="服务地址")
    model: str = Field(default="llama2", description="模型名称")
    api_key: str = Field(default="dummy", description="API密钥(本地服务通常不需要)")
    timeout: int = Field(default=60, description="超时时间(秒)")
    max_tokens: int = Field(default=1000, description="最大token数")


class LLMConfig(BaseModel):
    """LLM 配置主类"""
    provider: str = Field(default="openai", description="默认提供商")
    openai: Optional[OpenAIConfig] = None
    anthropic: Optional[AnthropicConfig] = None
    local: Optional[LocalLLMConfig] = None
    
    # 通用配置
    retry_attempts: int = Field(default=3, description="重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟(秒)")
    enable_fallback: bool = Field(default=True, description="是否启用降级策略")


class QueryParsingConfig(BaseModel):
    """查询解析配置"""
    default_limit: int = Field(default=100, description="默认结果数量限制")
    semantic_threshold: float = Field(default=0.7, description="语义匹配阈值")
    max_semantic_results: int = Field(default=100, description="最大语义结果数量")


class PromptsConfig(BaseModel):
    """提示模板配置"""
    query_analysis: str = Field(description="查询分析提示模板")
    sql_generation: str = Field(description="SQL生成提示模板")
    confidence_calculation: str = Field(description="置信度计算提示模板")


class FullLLMConfig(BaseModel):
    """完整的 LLM 配置（匹配 YAML 结构）"""
    llm: LLMConfig
    query_parsing: Optional[QueryParsingConfig] = None
    prompts: Optional[PromptsConfig] = None


class LLMConfigManager:
    """LLM 配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.full_config = self._load_config(config_path)
        self.config = self.full_config.llm  # 提取 LLM 配置部分
        self._validate_config()
        logger.info("LLM 配置管理器初始化完成")
    
    @classmethod
    def from_default_config(cls, default_config) -> "LLMConfigManager":
        """
        从default.yaml配置创建LLM配置管理器
        
        Args:
            default_config: 从default.yaml加载的配置对象
        
        Returns:
            LLMConfigManager实例
        """
        # 将default.yaml的配置转换为LLM配置格式
        llm_config_data = {
            "provider": default_config.semantic.llm_provider,
            "enable_fallback": default_config.semantic.fallback_enabled
        }
        
        # OpenAI 配置
        if hasattr(default_config.semantic, 'openai') and default_config.semantic.openai.api_key:
            llm_config_data["openai"] = {
                "model": default_config.semantic.openai.model,
                "api_key": default_config.semantic.openai.api_key,
                "max_tokens": default_config.semantic.openai.max_tokens,
                "temperature": default_config.semantic.openai.temperature,
                "timeout": default_config.semantic.openai.timeout,
                "retry_attempts": default_config.semantic.openai.retry_attempts,
                "retry_delay": default_config.semantic.openai.retry_delay
            }
        
        # Anthropic 配置
        if hasattr(default_config.semantic, 'anthropic') and default_config.semantic.anthropic.api_key:
            llm_config_data["anthropic"] = {
                "model": default_config.semantic.anthropic.model,
                "api_key": default_config.semantic.anthropic.api_key,
                "max_tokens": default_config.semantic.anthropic.max_tokens,
                "timeout": default_config.semantic.anthropic.timeout,
                "retry_attempts": default_config.semantic.anthropic.retry_attempts,
                "retry_delay": default_config.semantic.anthropic.retry_delay
            }
        
        # 本地 LLM 配置
        if hasattr(default_config.semantic, 'local') and default_config.semantic.local.base_url:
            llm_config_data["local"] = {
                "base_url": default_config.semantic.local.base_url,
                "model": default_config.semantic.local.model,
                "timeout": default_config.semantic.local.timeout,
                "max_tokens": default_config.semantic.local.max_tokens,
                "retry_attempts": default_config.semantic.local.retry_attempts,
                "retry_delay": default_config.semantic.local.retry_delay
            }
        
        # 创建完整的配置结构
        full_config_data = {
            "llm": llm_config_data,
            "query_parsing": {
                "default_limit": default_config.query.default_limit,
                "semantic_threshold": default_config.semantic.semantic_threshold,
                "max_semantic_results": default_config.semantic.max_semantic_results
            }
        }
        
        # 创建实例
        instance = cls.__new__(cls)
        instance.full_config = FullLLMConfig(**full_config_data)
        instance.config = instance.full_config.llm
        instance._validate_config()
        logger.info("LLM 配置管理器从default.yaml配置初始化完成")
        
        return instance
    
    def _load_config(self, config_path: Optional[str]) -> FullLLMConfig:
        """加载配置文件"""
        try:
            if config_path and os.path.exists(config_path):
                # 从文件加载配置
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # 处理环境变量替换
                config_data = self._replace_env_vars(config_data)
                
                return FullLLMConfig(**config_data)
            else:
                # 使用环境变量创建默认配置
                return self._create_config_from_env()
        except Exception as e:
            logger.warning(f"加载配置文件失败，使用环境变量配置: {e}")
            return self._create_config_from_env()
    
    def _replace_env_vars(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """替换配置中的环境变量"""
        import re
        
        def replace_in_value(value):
            if isinstance(value, str):
                # 匹配 ${VAR_NAME} 格式的环境变量
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, value)
                for match in matches:
                    env_value = os.getenv(match)
                    if env_value:
                        value = value.replace(f'${{{match}}}', env_value)
                    else:
                        logger.warning(f"环境变量 {match} 未设置")
            return value
        
        def process_dict(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        data[key] = process_dict(value)
                    else:
                        data[key] = replace_in_value(value)
            return data
        
        return process_dict(config_data)
    
    def _create_config_from_env(self) -> FullLLMConfig:
        """从环境变量创建配置"""
        llm_config_data = {
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "retry_attempts": int(os.getenv("LLM_RETRY_ATTEMPTS", "3")),
            "retry_delay": float(os.getenv("LLM_RETRY_DELAY", "1.0")),
            "enable_fallback": os.getenv("LLM_ENABLE_FALLBACK", "true").lower() == "true"
        }
        
        # OpenAI 配置
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            llm_config_data["openai"] = {
                "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                "api_key": openai_api_key,
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
                "timeout": int(os.getenv("OPENAI_TIMEOUT", "30"))
            }
        
        # Anthropic 配置
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            llm_config_data["anthropic"] = {
                "model": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                "api_key": anthropic_api_key,
                "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "1000")),
                "timeout": int(os.getenv("ANTHROPIC_TIMEOUT", "30"))
            }
        
        # 本地 LLM 配置
        local_base_url = os.getenv("LOCAL_LLM_BASE_URL")
        if local_base_url:
            llm_config_data["local"] = {
                "base_url": local_base_url,
                "model": os.getenv("LOCAL_LLM_MODEL", "llama2"),
                "timeout": int(os.getenv("LOCAL_LLM_TIMEOUT", "60")),
                "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "1000"))
            }
        
        # 创建完整的配置结构
        full_config_data = {
            "llm": llm_config_data,
            "query_parsing": {
                "default_limit": int(os.getenv("DEFAULT_QUERY_LIMIT", "100")),
                "semantic_threshold": float(os.getenv("SEMANTIC_THRESHOLD", "0.7")),
                "max_semantic_results": int(os.getenv("MAX_SEMANTIC_RESULTS", "100"))
            }
        }
        
        return FullLLMConfig(**full_config_data)
    
    def _validate_config(self):
        """验证配置的有效性"""
        # 检查是否有可用的提供商
        available_providers = self.get_available_providers()
        if not available_providers:
            logger.warning("未配置任何有效的LLM服务，系统将使用模拟客户端")
        
        # 验证默认提供商是否可用
        if self.config.provider not in ["openai", "anthropic", "local"]:
            logger.warning(f"不支持的LLM提供商: {self.config.provider}，将使用默认配置")
            self.config.provider = "openai"
        
        # 如果默认提供商不可用，尝试使用第一个可用的提供商
        if self.config.provider not in available_providers and available_providers:
            logger.info(f"默认提供商 {self.config.provider} 不可用，切换到 {available_providers[0]}")
            self.config.provider = available_providers[0]
    
    def get_provider_config(self, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取指定提供商的配置"""
        provider = provider or self.config.provider
        
        if provider == "openai" and self.config.openai and self.config.openai.api_key and self.config.openai.api_key != "${OPENAI_API_KEY}":
            return self.config.openai.dict()
        elif provider == "anthropic" and self.config.anthropic and self.config.anthropic.api_key and self.config.anthropic.api_key != "${ANTHROPIC_API_KEY}":
            return self.config.anthropic.dict()
        elif provider == "local" and self.config.local and self.config.local.base_url:
            return self.config.local.dict()
        else:
            return None
    
    def get_available_providers(self) -> list:
        """获取可用的LLM提供商列表"""
        providers = []
        
        # 检查 OpenAI 配置
        if self.config.openai and self.config.openai.api_key and self.config.openai.api_key != "${OPENAI_API_KEY}":
            providers.append("openai")
        
        # 检查 Anthropic 配置
        if self.config.anthropic and self.config.anthropic.api_key and self.config.anthropic.api_key != "${ANTHROPIC_API_KEY}":
            providers.append("anthropic")
        
        # 检查本地 LLM 配置
        if self.config.local and self.config.local.base_url:
            providers.append("local")
        
        return providers
    
    def get_fallback_provider(self, current_provider: str) -> Optional[str]:
        """获取降级提供商"""
        if not self.config.enable_fallback:
            return None
        
        available_providers = self.get_available_providers()
        if current_provider in available_providers:
            available_providers.remove(current_provider)
        
        return available_providers[0] if available_providers else None
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self._validate_config()
        logger.info("LLM 配置已更新")
    
    def get_query_parsing_config(self) -> Optional[QueryParsingConfig]:
        """获取查询解析配置"""
        return self.full_config.query_parsing
    
    def get_prompts_config(self) -> Optional[PromptsConfig]:
        """获取提示模板配置"""
        return self.full_config.prompts
    
    def get_full_config(self) -> FullLLMConfig:
        """获取完整配置"""
        return self.full_config
    
    def export_config(self, file_path: str):
        """导出配置到文件"""
        try:
            import yaml
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.full_config.dict(), f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置已导出到: {file_path}")
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            raise