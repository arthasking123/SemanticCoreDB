"""
嵌入服务模块 - 生成多模态数据的向量嵌入
"""

import asyncio
import numpy as np
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import cv2
from loguru import logger

from ..core.config import SemanticConfig


class EmbeddingService:
    """
    嵌入服务 - 为多模态数据生成向量嵌入
    """
    
    def __init__(self, config: SemanticConfig):
        """
        初始化嵌入服务
        
        Args:
            config: 语义配置
        """
        self.config = config
        self.embedding_dimension = config.embedding_dimension
        
        # 初始化模型
        self._init_models()
        
        logger.info(f"嵌入服务初始化完成，维度: {self.embedding_dimension}")
    
    def _init_models(self):
        """初始化嵌入模型"""
        try:
            # 文本嵌入模型
            model_name = self.config.embedding_model
            print(f"初始化文本嵌入模型: {model_name}")
            
            # 设置模型缓存路径，避免重复下载
            cache_folder = os.path.expanduser("~/.cache/sentence_transformers")
            os.makedirs(cache_folder, exist_ok=True)
            
            # 检查本地是否已有模型
            if self._check_local_model_exists(model_name):
                print(f"✅ 使用本地模型: {model_name}")
                logger.info(f"使用本地模型: {model_name}")
            else:
                print(f"📥 下载模型: {model_name}")
                logger.info(f"开始下载模型: {model_name}")
            
            # 加载模型，指定缓存目录
            self.text_model = SentenceTransformer(
                model_name,
                cache_folder=cache_folder
            )
            
            # 图像嵌入模型（使用 CLIP 或类似模型）
            # 这里可以根据需要加载专门的图像嵌入模型
            self.image_model = None  # 暂时使用文本模型
            
            # 音频嵌入模型
            self.audio_model = None  # 暂时使用文本模型
            
            # 视频嵌入模型
            self.video_model = None  # 暂时使用文本模型
            
            logger.info("嵌入模型初始化完成")
            
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {e}")
            raise
    
    def _check_local_model_exists(self, model_name: str) -> bool:
        """检查本地是否已有模型"""
        try:
            import os
            # 检查 HuggingFace 缓存目录
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_path = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
            
            if os.path.exists(model_path):
                return True
            
            # 检查 SentenceTransformers 缓存目录
            st_cache_dir = os.path.expanduser("~/.cache/sentence_transformers")
            st_model_path = os.path.join(st_cache_dir, model_name.replace("/", "_"))
            
            if os.path.exists(st_model_path):
                return True
            
            return False
            
        except Exception:
            return False
    
    async def generate_embedding(self, data: Dict[str, Any]) -> np.ndarray:
        """
        生成数据嵌入
        
        Args:
            data: 数据对象
        
        Returns:
            向量嵌入
        """
        try:
            data_type = data.get('type', 'text')
            data_content = data.get('data', '')
            
            if data_type == 'text':
                return await self._generate_text_embedding(data_content)
            elif data_type == 'image':
                return await self._generate_image_embedding(data_content)
            elif data_type == 'video':
                return await self._generate_video_embedding(data_content)
            elif data_type == 'audio':
                return await self._generate_audio_embedding(data_content)
            elif data_type == 'iot':
                return await self._generate_iot_embedding(data_content)
            else:
                # 默认使用文本嵌入
                return await self._generate_text_embedding(str(data_content))
                
        except Exception as e:
            logger.error(f"生成嵌入失败: {e}")
            # 返回零向量
            return np.zeros(self.embedding_dimension)
    
    async def _generate_text_embedding(self, text: str) -> np.ndarray:
        """生成文本嵌入"""
        try:
            # 使用 sentence-transformers 生成嵌入
            embedding = self.text_model.encode(text)
            
            # 确保维度正确
            if len(embedding) != self.embedding_dimension:
                # 如果维度不匹配，进行截断或填充
                if len(embedding) > self.embedding_dimension:
                    embedding = embedding[:self.embedding_dimension]
                else:
                    embedding = np.pad(embedding, (0, self.embedding_dimension - len(embedding)))
            
            return embedding
            
        except Exception as e:
            logger.error(f"生成文本嵌入失败: {e}")
            return np.zeros(self.embedding_dimension)
    
    async def _generate_image_embedding(self, image_path: str) -> np.ndarray:
        """生成图像嵌入"""
        try:
            # 这里应该使用专门的图像嵌入模型
            # 暂时使用文本描述生成嵌入
            
            # 读取图像
            if Path(image_path).exists():
                image = Image.open(image_path)
                
                # 生成图像描述（这里可以集成图像描述模型）
                description = await self._generate_image_description(image)
                
                # 使用文本嵌入模型
                return await self._generate_text_embedding(description)
            else:
                # 如果图像文件不存在，使用路径作为描述
                return await self._generate_text_embedding(f"image: {image_path}")
                
        except Exception as e:
            logger.error(f"生成图像嵌入失败: {e}")
            return np.zeros(self.embedding_dimension)
    
    async def _generate_video_embedding(self, video_path: str) -> np.ndarray:
        """生成视频嵌入"""
        try:
            # 这里应该使用专门的视频嵌入模型
            # 暂时使用视频描述生成嵌入
            
            if Path(video_path).exists():
                # 提取视频帧
                frames = await self._extract_video_frames(video_path)
                
                # 生成视频描述
                description = await self._generate_video_description(frames)
                
                # 使用文本嵌入模型
                return await self._generate_text_embedding(description)
            else:
                return await self._generate_text_embedding(f"video: {video_path}")
                
        except Exception as e:
            logger.error(f"生成视频嵌入失败: {e}")
            return np.zeros(self.embedding_dimension)
    
    async def _generate_audio_embedding(self, audio_path: str) -> np.ndarray:
        """生成音频嵌入"""
        try:
            # 这里应该使用专门的音频嵌入模型
            # 暂时使用音频描述生成嵌入
            
            if Path(audio_path).exists():
                # 提取音频特征
                features = await self._extract_audio_features(audio_path)
                
                # 生成音频描述
                description = await self._generate_audio_description(features)
                
                # 使用文本嵌入模型
                return await self._generate_text_embedding(description)
            else:
                return await self._generate_text_embedding(f"audio: {audio_path}")
                
        except Exception as e:
            logger.error(f"生成音频嵌入失败: {e}")
            return np.zeros(self.embedding_dimension)
    
    async def _generate_iot_embedding(self, iot_data: Dict[str, Any]) -> np.ndarray:
        """生成 IoT 数据嵌入"""
        try:
            # 将 IoT 数据转换为文本描述
            description = await self._generate_iot_description(iot_data)
            
            # 使用文本嵌入模型
            return await self._generate_text_embedding(description)
            
        except Exception as e:
            logger.error(f"生成 IoT 嵌入失败: {e}")
            return np.zeros(self.embedding_dimension)
    
    async def _generate_image_description(self, image: Image.Image) -> str:
        """生成图像描述"""
        # 这里可以集成图像描述模型，如 BLIP 或 CLIP
        # 暂时返回基本描述
        return f"image with size {image.size}"
    
    async def _extract_video_frames(self, video_path: str) -> List[np.ndarray]:
        """提取视频帧"""
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            while cap.isOpened() and frame_count < 10:  # 最多提取 10 帧
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"提取视频帧失败: {e}")
            return []
    
    async def _generate_video_description(self, frames: List[np.ndarray]) -> str:
        """生成视频描述"""
        return f"video with {len(frames)} frames"
    
    async def _extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """提取音频特征"""
        # 这里可以集成 librosa 等音频处理库
        return {"duration": 0, "sample_rate": 0}
    
    async def _generate_audio_description(self, features: Dict[str, Any]) -> str:
        """生成音频描述"""
        return f"audio with duration {features.get('duration', 0)}s"
    
    async def _generate_iot_description(self, iot_data: Dict[str, Any]) -> str:
        """生成 IoT 数据描述"""
        # 将 IoT 数据转换为文本描述
        description_parts = []
        
        for key, value in iot_data.items():
            if isinstance(value, (int, float)):
                description_parts.append(f"{key}: {value}")
            elif isinstance(value, str):
                description_parts.append(f"{key}: {value}")
            else:
                description_parts.append(f"{key}: {str(value)}")
        
        return "IoT data: " + ", ".join(description_parts)
    
    async def batch_generate_embeddings(self, data_list: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        批量生成嵌入
        
        Args:
            data_list: 数据列表
        
        Returns:
            嵌入列表
        """
        try:
            embeddings = []
            
            for data in data_list:
                embedding = await self.generate_embedding(data)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"批量生成嵌入失败: {e}")
            return [np.zeros(self.embedding_dimension)] * len(data_list)
    
    async def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个嵌入的相似度
        
        Args:
            embedding1: 第一个嵌入
            embedding2: 第二个嵌入
        
        Returns:
            相似度分数
        """
        try:
            # 使用余弦相似度
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return 0.0
    
    async def close(self):
        """关闭嵌入服务"""
        try:
            # 清理模型资源
            if hasattr(self, 'text_model'):
                del self.text_model
            if hasattr(self, 'image_model'):
                del self.image_model
            if hasattr(self, 'audio_model'):
                del self.audio_model
            if hasattr(self, 'video_model'):
                del self.video_model
            
            logger.info("嵌入服务已关闭")
        except Exception as e:
            logger.error(f"关闭嵌入服务失败: {e}") 