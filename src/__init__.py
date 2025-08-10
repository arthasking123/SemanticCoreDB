"""
SemanticCoreDB - 基于 LLM 的语义驱动数据库

一个全新的数据库系统，摒弃传统存储引擎和文件格式，
采用事件化对象存储 + 向量索引 + 元数据图结构的组合架构。
"""

__version__ = "0.1.0"
__author__ = "SemanticCoreDB Team"
__email__ = "team@semanticcoredb.com"

from .core.database import SemanticCoreDB
from .core.config import Config

__all__ = ["SemanticCoreDB", "Config"] 