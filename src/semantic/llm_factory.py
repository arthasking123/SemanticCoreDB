"""
LLM 客户端工厂模块 - 统一管理 LLM 客户端的创建和切换
"""

import asyncio
from typing import Dict, Any, Optional, List
from loguru import logger

from .llm_client import LLMClient, create_llm_client
from .llm_config import LLMConfigManager


class LLMClientFactory:
    """LLM 客户端工厂类"""
    
    def __init__(self, config_manager: LLMConfigManager):
        """
        初始化工厂类
        
        Args:
            config_manager: LLM配置管理器
        """
        self.config_manager = config_manager
        self.clients: Dict[str, LLMClient] = {}
        self.active_client: Optional[LLMClient] = None
        self.client_health: Dict[str, bool] = {}
        
        logger.info("LLM 客户端工厂初始化完成")
    
    async def initialize(self):
        """初始化所有可用的LLM客户端"""
        try:
            available_providers = self.config_manager.get_available_providers()
            
            for provider in available_providers:
                await self._create_client(provider)
            
            # 设置默认客户端
            if available_providers:
                default_provider = self.config_manager.config.provider
                if default_provider in available_providers:
                    self.active_client = self.clients[default_provider]
                else:
                    self.active_client = self.clients[available_providers[0]]
                
                logger.info(f"默认LLM客户端设置为: {self.active_client.__class__.__name__}")
            else:
                logger.warning("没有可用的LLM客户端，将使用模拟客户端")
                self.active_client = create_llm_client("mock")
                
        except Exception as e:
            logger.error(f"初始化LLM客户端失败: {e}")
            # 创建模拟客户端作为后备
            self.active_client = create_llm_client("mock")
    
    async def _create_client(self, provider: str) -> Optional[LLMClient]:
        """创建指定提供商的LLM客户端"""
        try:
            config = self.config_manager.get_provider_config(provider)
            if not config:
                logger.warning(f"提供商 {provider} 的配置不存在")
                return None
            
            client = create_llm_client(provider, **config)
            
            # 测试客户端连接
            if await self._test_client(client):
                self.clients[provider] = client
                self.client_health[provider] = True
                logger.info(f"LLM客户端 {provider} 创建成功")
                return client
            else:
                logger.warning(f"LLM客户端 {provider} 连接测试失败")
                return None
                
        except Exception as e:
            logger.error(f"创建LLM客户端 {provider} 失败: {e}")
            return None
    
    async def _test_client(self, client: LLMClient) -> bool:
        """测试客户端连接"""
        try:
            # 使用简单的测试提示
            test_prompt = "Hello, this is a connection test."
            response = await asyncio.wait_for(
                client.generate(test_prompt, max_tokens=10),
                timeout=10.0
            )
            return bool(response and len(response.strip()) > 0)
        except Exception as e:
            logger.debug(f"客户端连接测试失败: {e}")
            return False
    
    async def get_client(self, provider: Optional[str] = None) -> LLMClient:
        """获取LLM客户端"""
        if provider:
            if provider in self.clients and self.client_health[provider]:
                return self.clients[provider]
            else:
                # 尝试重新创建客户端
                client = await self._create_client(provider)
                if client:
                    return client
        
        # 返回当前活跃客户端
        if self.active_client:
            return self.active_client
        
        # 如果没有可用客户端，创建模拟客户端
        logger.warning("没有可用的LLM客户端，使用模拟客户端")
        return create_llm_client("mock")
    
    async def switch_client(self, provider: str) -> bool:
        """切换LLM客户端"""
        try:
            if provider not in self.clients:
                client = await self._create_client(provider)
                if not client:
                    return False
            
            if self.client_health[provider]:
                self.active_client = self.clients[provider]
                logger.info(f"已切换到LLM客户端: {provider}")
                return True
            else:
                logger.warning(f"LLM客户端 {provider} 不健康，无法切换")
                return False
                
        except Exception as e:
            logger.error(f"切换LLM客户端失败: {e}")
            return False
    
    async def health_check(self) -> Dict[str, bool]:
        """检查所有客户端的健康状态"""
        health_status = {}
        
        for provider, client in self.clients.items():
            try:
                is_healthy = await self._test_client(client)
                self.client_health[provider] = is_healthy
                health_status[provider] = is_healthy
            except Exception as e:
                logger.error(f"检查客户端 {provider} 健康状态失败: {e}")
                self.client_health[provider] = False
                health_status[provider] = False
        
        return health_status
    
    async def get_best_client(self) -> Optional[LLMClient]:
        """获取最佳可用的LLM客户端"""
        # 首先检查当前活跃客户端
        if self.active_client and self.client_health.get(
            self._get_client_provider(self.active_client), True
        ):
            return self.active_client
        
        # 寻找健康的客户端
        for provider, is_healthy in self.client_health.items():
            if is_healthy:
                self.active_client = self.clients[provider]
                logger.info(f"切换到健康的LLM客户端: {provider}")
                return self.active_client
        
        # 尝试重新初始化所有客户端
        await self.initialize()
        return self.active_client
    
    def _get_client_provider(self, client: LLMClient) -> str:
        """获取客户端对应的提供商名称"""
        for provider, client_instance in self.clients.items():
            if client_instance == client:
                return provider
        return "unknown"
    
    async def generate_with_fallback(self, prompt: str, **kwargs) -> str:
        """使用降级策略生成响应"""
        try:
            # 首先尝试当前活跃客户端
            if self.active_client:
                try:
                    return await self.active_client.generate(prompt, **kwargs)
                except Exception as e:
                    logger.warning(f"当前LLM客户端失败，尝试降级: {e}")
            
            # 尝试其他可用的客户端
            for provider, client in self.clients.items():
                if provider != self._get_client_provider(self.active_client) and self.client_health[provider]:
                    try:
                        response = await client.generate(prompt, **kwargs)
                        # 切换到成功的客户端
                        self.active_client = client
                        logger.info(f"降级到LLM客户端: {provider}")
                        return response
                    except Exception as e:
                        logger.warning(f"LLM客户端 {provider} 降级失败: {e}")
                        continue
            
            # 所有客户端都失败，使用模拟客户端
            logger.error("所有LLM客户端都失败，使用模拟客户端")
            mock_client = create_llm_client("mock")
            return await mock_client.generate(prompt, **kwargs)
            
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            raise
    
    def get_client_info(self) -> Dict[str, Any]:
        """获取客户端信息"""
        info = {
            "active_client": self._get_client_provider(self.active_client) if self.active_client else None,
            "available_clients": list(self.clients.keys()),
            "client_health": self.client_health.copy(),
            "total_clients": len(self.clients)
        }
        return info
    
    async def close_all(self):
        """关闭所有客户端"""
        for provider, client in self.clients.items():
            try:
                await client.close()
                logger.info(f"LLM客户端 {provider} 已关闭")
            except Exception as e:
                logger.error(f"关闭LLM客户端 {provider} 失败: {e}")
        
        self.clients.clear()
        self.active_client = None
        self.client_health.clear()
        logger.info("所有LLM客户端已关闭") 