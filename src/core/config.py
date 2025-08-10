"""
配置管理模块
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class StorageConfig:
    """存储配置"""
    data_dir: str = "data"
    event_log_dir: str = "data/events"
    object_store_dir: str = "data/objects"
    vector_index_dir: str = "data/vectors"
    metadata_graph_dir: str = "data/metadata"
    
    # 存储参数
    chunk_size: int = 1024 * 1024  # 1MB
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    compression_enabled: bool = True
    
    # 缓存配置
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1小时


@dataclass
class LLMProviderConfig:
    """LLM提供商配置基类"""
    model: str
    max_tokens: int = 1000
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class OpenAIConfig(LLMProviderConfig):
    """OpenAI配置"""
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.1


@dataclass
class AnthropicConfig(LLMProviderConfig):
    """Anthropic配置"""
    model: str = "claude-3-sonnet-20240229"
    api_key: Optional[str] = None


@dataclass
class LocalLLMConfig(LLMProviderConfig):
    """本地LLM配置"""
    base_url: str = "http://localhost:11434"
    model: str = "Qwen2.5-Coder:7b"
    timeout: int = 60


@dataclass
class SemanticConfig:
    """语义处理配置"""
    # 默认LLM提供商
    llm_provider: str = "local"  # mock, openai, anthropic, local
    
    # 各提供商配置
    openai: OpenAIConfig = None
    anthropic: AnthropicConfig = None
    local: LocalLLMConfig = None
    
    # 降级策略配置
    fallback_enabled: bool = True
    
    # 向量模型配置
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # 向量索引配置
    vector_index_type: str = "faiss"  # faiss, annoy, hnsw
    vector_index_params: Dict[str, Any] = None
    
    # 语义匹配阈值
    semantic_threshold: float = 0.7
    max_semantic_results: int = 100
    
    def __post_init__(self):
        """初始化后处理"""
        if self.openai is None:
            self.openai = OpenAIConfig()
        if self.anthropic is None:
            self.anthropic = AnthropicConfig()
        if self.local is None:
            self.local = LocalLLMConfig()


@dataclass
class QueryConfig:
    """查询配置"""
    # 查询解析
    enable_natural_language: bool = True
    enable_sql_plus: bool = True
    
    # 查询优化
    query_timeout: int = 30  # 秒
    max_query_complexity: int = 1000
    
    # 结果限制
    default_limit: int = 100
    max_limit: int = 1000


@dataclass
class APIConfig:
    """API 配置"""
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    
    # 认证
    enable_auth: bool = False
    jwt_secret: Optional[str] = None
    
    # 限流
    rate_limit: int = 1000  # 每分钟请求数
    rate_limit_window: int = 60  # 秒


class Config:
    """主配置类"""
    
    def __init__(self, 
                 storage: StorageConfig = None,
                 semantic: SemanticConfig = None,
                 query: QueryConfig = None,
                 api: APIConfig = None):
        """初始化配置"""
        self.storage = storage or StorageConfig()
        self.semantic = semantic or SemanticConfig()
        self.query = query or QueryConfig()
        self.api = api or APIConfig()
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """从配置文件加载配置"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> "Config":
        """从字典创建配置"""
        storage_config = StorageConfig(**config_data.get("storage", {}))
        
        # 处理语义配置，包括嵌套的LLM提供商配置
        semantic_data = config_data.get("semantic", {})
        semantic_config = SemanticConfig()
        
        # 设置基本属性
        if "llm_provider" in semantic_data:
            semantic_config.llm_provider = semantic_data["llm_provider"]
        if "fallback_enabled" in semantic_data:
            semantic_config.fallback_enabled = semantic_data["fallback_enabled"]
        if "embedding_model" in semantic_data:
            semantic_config.embedding_model = semantic_data["embedding_model"]
        if "embedding_dimension" in semantic_data:
            semantic_config.embedding_dimension = semantic_data["embedding_dimension"]
        if "vector_index_type" in semantic_data:
            semantic_config.vector_index_type = semantic_data["vector_index_type"]
        if "vector_index_params" in semantic_data:
            semantic_config.vector_index_params = semantic_data["vector_index_params"]
        if "semantic_threshold" in semantic_data:
            semantic_config.semantic_threshold = semantic_data["semantic_threshold"]
        if "max_semantic_results" in semantic_data:
            semantic_config.max_semantic_results = semantic_data["max_semantic_results"]
        
        # 处理LLM提供商配置
        if "openai" in semantic_data:
            semantic_config.openai = OpenAIConfig(**semantic_data["openai"])
        if "anthropic" in semantic_data:
            semantic_config.anthropic = AnthropicConfig(**semantic_data["anthropic"])
        if "local" in semantic_data:
            semantic_config.local = LocalLLMConfig(**semantic_data["local"])
        
        query_config = QueryConfig(**config_data.get("query", {}))
        api_config = APIConfig(**config_data.get("api", {}))
        
        return cls(
            storage=storage_config,
            semantic=semantic_config,
            query=query_config,
            api=api_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "storage": self.storage.__dict__,
            "semantic": {
                "llm_provider": self.semantic.llm_provider,
                "fallback_enabled": self.semantic.fallback_enabled,
                "embedding_model": self.semantic.embedding_model,
                "embedding_dimension": self.semantic.embedding_dimension,
                "vector_index_type": self.semantic.vector_index_type,
                "vector_index_params": self.semantic.vector_index_params,
                "semantic_threshold": self.semantic.semantic_threshold,
                "max_semantic_results": self.semantic.max_semantic_results,
                "openai": self.semantic.openai.__dict__,
                "anthropic": self.semantic.anthropic.__dict__,
                "local": self.semantic.local.__dict__
            },
            "query": self.query.__dict__,
            "api": self.api.__dict__
        }
    
    def save_to_file(self, config_path: str):
        """保存配置到文件"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


# 默认配置
DEFAULT_CONFIG = Config()

# 环境变量配置
def load_config_from_env() -> Config:
    """从环境变量加载配置"""
    config = Config()
    
    # 存储配置
    if os.getenv("SCDB_DATA_DIR"):
        config.storage.data_dir = os.getenv("SCDB_DATA_DIR")
    
    # 语义配置 - 默认LLM提供商
    if os.getenv("SCDB_LLM_PROVIDER"):
        config.semantic.llm_provider = os.getenv("SCDB_LLM_PROVIDER")
    if os.getenv("SCDB_FALLBACK_ENABLED"):
        config.semantic.fallback_enabled = os.getenv("SCDB_FALLBACK_ENABLED").lower() == "true"
    
    # OpenAI 配置
    if os.getenv("OPENAI_API_KEY"):
        config.semantic.openai.api_key = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_MODEL"):
        config.semantic.openai.model = os.getenv("OPENAI_MODEL")
    if os.getenv("OPENAI_MAX_TOKENS"):
        config.semantic.openai.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS"))
    if os.getenv("OPENAI_TEMPERATURE"):
        config.semantic.openai.temperature = float(os.getenv("OPENAI_TEMPERATURE"))
    if os.getenv("OPENAI_TIMEOUT"):
        config.semantic.openai.timeout = int(os.getenv("OPENAI_TIMEOUT"))
    if os.getenv("OPENAI_RETRY_ATTEMPTS"):
        config.semantic.openai.retry_attempts = int(os.getenv("OPENAI_RETRY_ATTEMPTS"))
    if os.getenv("OPENAI_RETRY_DELAY"):
        config.semantic.openai.retry_delay = float(os.getenv("OPENAI_RETRY_DELAY"))
    
    # Anthropic 配置
    if os.getenv("ANTHROPIC_API_KEY"):
        config.semantic.anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
    if os.getenv("ANTHROPIC_MODEL"):
        config.semantic.anthropic.model = os.getenv("ANTHROPIC_MODEL")
    if os.getenv("ANTHROPIC_MAX_TOKENS"):
        config.semantic.anthropic.max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS"))
    if os.getenv("ANTHROPIC_TIMEOUT"):
        config.semantic.anthropic.timeout = int(os.getenv("ANTHROPIC_TIMEOUT"))
    if os.getenv("ANTHROPIC_RETRY_ATTEMPTS"):
        config.semantic.anthropic.retry_attempts = int(os.getenv("ANTHROPIC_RETRY_ATTEMPTS"))
    if os.getenv("ANTHROPIC_RETRY_DELAY"):
        config.semantic.anthropic.retry_delay = float(os.getenv("ANTHROPIC_RETRY_DELAY"))
    
    # 本地 LLM 配置
    if os.getenv("LOCAL_LLM_BASE_URL"):
        config.semantic.local.base_url = os.getenv("LOCAL_LLM_BASE_URL")
    if os.getenv("LOCAL_LLM_MODEL"):
        config.semantic.local.model = os.getenv("LOCAL_LLM_MODEL")
    if os.getenv("LOCAL_LLM_TIMEOUT"):
        config.semantic.local.timeout = int(os.getenv("LOCAL_LLM_TIMEOUT"))
    if os.getenv("LOCAL_LLM_MAX_TOKENS"):
        config.semantic.local.max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS"))
    if os.getenv("LOCAL_LLM_RETRY_ATTEMPTS"):
        config.semantic.local.retry_attempts = int(os.getenv("LOCAL_LLM_RETRY_ATTEMPTS"))
    if os.getenv("LOCAL_LLM_RETRY_DELAY"):
        config.semantic.local.retry_delay = float(os.getenv("LOCAL_LLM_RETRY_DELAY"))
    
    # API 配置
    if os.getenv("SCDB_HOST"):
        config.api.host = os.getenv("SCDB_HOST")
    if os.getenv("SCDB_PORT"):
        config.api.port = int(os.getenv("SCDB_PORT"))
    
    return config