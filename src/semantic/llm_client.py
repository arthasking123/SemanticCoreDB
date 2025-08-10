"""
LLM 客户端接口 - 支持多种 LLM 服务集成
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from loguru import logger

class LLMClient(ABC):
    """LLM 客户端抽象基类"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成文本响应"""
        pass
    
    @abstractmethod
    async def close(self):
        """关闭客户端连接"""
        pass


class MockLLMClient(LLMClient):
    """模拟 LLM 客户端，用于测试和开发"""
    
    def __init__(self):
        self.responses = {
            "default": "这是一个模拟的LLM响应。请配置真实的LLM服务以获得更好的体验。",
            "query_analysis": '{"intent": {"type": "SELECT", "action": "find", "scope": "all"}, "entities": [], "filters": [], "sorting": [], "limit": 100}',
            "sql_generation": "SELECT * FROM multimodal_data LIMIT 100",
            "confidence": "0.7"
        }
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成模拟响应"""
        # 根据提示类型返回相应的模拟响应
        if "查询" in prompt or "分析" in prompt:
            return self.responses["query_analysis"]
        elif "SQL" in prompt or "生成" in prompt:
            return self.responses["sql_generation"]
        elif "置信度" in prompt or "评估" in prompt:
            return self.responses["confidence"]
        else:
            return self.responses["default"]
    
    async def close(self):
        """关闭客户端"""
        logger.info("模拟 LLM 客户端已关闭")


class OpenAILLMClient(LLMClient):
    """OpenAI API 客户端"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.client = None
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
            logger.info(f"OpenAI 客户端初始化成功，模型: {model}")
        except ImportError:
            logger.error("OpenAI 库未安装，请运行: pip install openai")
        except Exception as e:
            logger.error(f"OpenAI 客户端初始化失败: {e}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """使用 OpenAI API 生成响应"""
        try:
            if not self.client:
                raise Exception("OpenAI 客户端未初始化")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            raise
    
    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.close()
            logger.info("OpenAI 客户端已关闭")


class AnthropicLLMClient(LLMClient):
    """Anthropic Claude API 客户端"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self.client = None
        
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            logger.info(f"Anthropic 客户端初始化成功，模型: {model}")
        except ImportError:
            logger.error("Anthropic 库未安装，请运行: pip install anthropic")
        except Exception as e:
            logger.error(f"Anthropic 客户端初始化失败: {e}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """使用 Anthropic API 生成响应"""
        try:
            if not self.client:
                raise Exception("Anthropic 客户端未初始化")
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API 调用失败: {e}")
            raise
    
    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.close()
            logger.info("Anthropic 客户端已关闭")


class LocalLLMClient(LLMClient):
    """本地 LLM 客户端（支持 Ollama、LM Studio 等）"""
    
    def __init__(self, base_url: str, model: str = "llama2"):
        self.base_url = base_url
        self.model = model
        self.client = None
        
        try:
            import httpx
            self.client = httpx.AsyncClient(timeout=60.0)
            logger.info(f"本地 LLM 客户端初始化成功, 模型: {model}, 地址: {base_url}")
        except ImportError:
            logger.error("httpx 库未安装，请运行: pip install httpx")
        except Exception as e:
            logger.error(f"本地 LLM 客户端初始化失败: {e}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """使用本地 LLM 生成响应"""
        try:
            if not self.client:
                raise Exception("本地 LLM 客户端未初始化")
            
            # 设置合理的超时时间
            timeout = kwargs.get('timeout', 60)
            max_tokens = kwargs.get('max_tokens', 1000)
            temperature = kwargs.get('temperature', 0.1)
            
            # 检查是否是 Ollama 服务
            if "11434" in self.base_url:
                # 使用 Ollama 原生 API
                url = f"{self.base_url}/api/generate"
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                }
            else:
                # 使用 OpenAI 兼容 API
                url = f"{self.base_url}/v1/chat/completions"
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
            
            # 使用 httpx 发送请求
            response = await self.client.post(
                url, 
                json=payload, 
                timeout=timeout
            )
            
            if response.status_code == 200:
                try:
                    # 首先尝试直接解析 JSON
                    data = response.json()
                    
                    if "11434" in self.base_url:
                        # Ollama 原生 API 响应格式
                        return data.get("response", "")
                    else:
                        # OpenAI 兼容 API 响应格式
                        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                except json.JSONDecodeError:
                    # 如果直接解析失败，尝试提取 markdown 代码块中的 JSON
                    try:
                        if "```json" in response.text:
                            # 提取 ```json 和 ``` 之间的内容
                            start_marker = "```json"
                            end_marker = "```"
                            
                            start_idx = response.text.find(start_marker)
                            if start_idx != -1:
                                start_idx += len(start_marker)
                                end_idx = response.text.find(end_marker, start_idx)
                                
                                if end_idx != -1:
                                    json_text = response.text[start_idx:end_idx].strip()
                                    # 清理可能的转义字符
                                    json_text = json_text.replace("\\n", "").replace("\\", "")
                                    
                                    logger.debug(f"提取的 JSON 文本: {json_text}")
                                    
                                    # 尝试解析 JSON
                                    parsed_data = json.loads(json_text)
                                    
                                    # 根据不同的响应格式返回内容
                                    if "11434" in self.base_url:
                                        return parsed_data.get("response", "")
                                    else:
                                        return parsed_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        else:
                            # 如果没有 JSON 标记，直接返回文本内容
                            logger.warning("响应不是有效的 JSON 格式，返回原始文本")
                            return response.text.strip()
                            
                    except (json.JSONDecodeError, IndexError, KeyError) as e:
                        logger.error(f"JSON 解析失败: {e}")
                        logger.error(f"原始响应: {response.text}")
                        # 返回原始响应文本作为后备方案
                        return response.text.strip()
            else:
                error_text = response.text
                logger.error(f"HTTP {response.status_code}: {error_text}")
                
                if response.status_code == 502:
                    logger.error("本地 LLM 服务连接失败 (502)")
                    logger.error("请检查本地 LLM 服务是否正在运行:")
                    logger.error(f"1. 确认服务地址: {self.base_url}")
                    logger.error(f"2. 确认模型: {self.model}")
                    logger.error("3. 对于 Ollama: 运行 'ollama serve' 并确保模型已下载")
                    logger.error("4. 对于 LM Studio: 启动本地服务器")
                    raise Exception(f"本地 LLM 服务连接失败 (502): {error_text}")
                elif response.status_code == 404:
                    logger.error(f"模型未找到: {self.model}")
                    logger.error("请检查模型名称是否正确，或使用 'ollama list' 查看可用模型")
                    raise Exception(f"模型未找到: {self.model}")
                else:
                    raise Exception(f"HTTP {response.status_code}: {error_text}")
            
        except Exception as e:
            error_msg = str(e)
            if "502" in error_msg or "Bad Gateway" in error_msg:
                logger.error(f"本地 LLM 服务连接失败 (502): {error_msg}")
                logger.error("请检查本地 LLM 服务是否正在运行:")
                logger.error(f"1. 确认服务地址: {self.base_url}")
                logger.error(f"2. 确认模型: {self.model}")
                logger.error("3. 对于 Ollama: 运行 'ollama serve' 并确保模型已下载")
                logger.error("4. 对于 LM Studio: 启动本地服务器")
                raise Exception(f"本地 LLM 服务连接失败: {error_msg}")
            elif "404" in error_msg or "Model not found" in error_msg:
                logger.error(f"模型未找到: {self.model}")
                logger.error("请检查模型名称是否正确，或使用 'ollama list' 查看可用模型")
                raise Exception(f"模型未找到: {self.model}")
            else:
                logger.error(f"本地 LLM 调用失败: {error_msg}")
                raise
    
    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.aclose()
            logger.info("本地 LLM 客户端已关闭")


def create_llm_client(client_type: str, **kwargs) -> LLMClient:
    """创建 LLM 客户端工厂函数"""
    try:
        if client_type == "mock":
            return MockLLMClient()
        elif client_type == "openai":
            api_key = kwargs.get('api_key')
            if not api_key:
                raise ValueError("OpenAI API key 是必需的")
            return OpenAILLMClient(api_key, kwargs.get('model', 'gpt-3.5-turbo'))
        elif client_type == "anthropic":
            api_key = kwargs.get('api_key')
            if not api_key:
                raise ValueError("Anthropic API key 是必需的")
            return AnthropicLLMClient(api_key, kwargs.get('model', 'claude-3-sonnet-20240229'))
        elif client_type == "local":
            base_url = kwargs.get('base_url')
            if not base_url:
                raise ValueError("本地 LLM 的 base_url 是必需的")
            return LocalLLMClient(base_url, kwargs.get('model', 'llama2'))
        else:
            raise ValueError(f"不支持的 LLM 客户端类型: {client_type}。支持的类型: mock, openai, anthropic, local")
            
    except Exception as e:
        logger.error(f"创建 LLM 客户端失败: {e}")
        raise 