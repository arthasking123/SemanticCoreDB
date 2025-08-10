"""
对象存储模块 - 管理多模态数据的存储
"""

import asyncio
import json
import uuid
import hashlib
from typing import Dict, List, Any, Optional, BinaryIO
from pathlib import Path
import aiofiles
import aiofiles.os
from loguru import logger

from ..core.config import StorageConfig


class ObjectStore:
    """
    对象存储引擎 - 管理多模态数据的存储
    """
    
    def __init__(self, config: StorageConfig):
        """
        初始化对象存储
        
        Args:
            config: 存储配置
        """
        self.config = config
        self.object_dir = Path(config.object_store_dir)
        self.object_dir.mkdir(parents=True, exist_ok=True)
        
        # 对象索引
        self.object_index = {}
        self.object_count = 0
        
        # 加载现有索引
        self._load_index()
        
        logger.info(f"对象存储初始化完成，目录: {self.object_dir}")
    
    def _load_index(self):
        """加载对象索引"""
        try:
            index_file = self.object_dir / "object_index.json"
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    self.object_index = json.load(f)
                    self.object_count = len(self.object_index)
                
                logger.info(f"加载对象索引，对象数: {self.object_count}")
                
        except Exception as e:
            logger.error(f"加载对象索引失败: {e}")
    
    async def _save_index(self):
        """保存对象索引"""
        try:
            index_file = self.object_dir / "object_index.json"
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(self.object_index, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存对象索引失败: {e}")
    
    async def store_object(self, object_id: str, data: Dict[str, Any]) -> bool:
        """
        存储对象
        
        Args:
            object_id: 对象 ID
            data: 对象数据
        
        Returns:
            是否存储成功
        """
        try:
            # 创建对象目录
            object_path = self.object_dir / object_id
            object_path.mkdir(exist_ok=True)
            
            # 存储元数据
            metadata = {
                'object_id': object_id,
                'type': data.get('type', 'unknown'),
                'metadata': data.get('metadata', {}),
                'tags': data.get('tags', []),
                'created_at': data.get('timestamp', ''),
                'size': 0,
                'checksum': ''
            }
            
            metadata_file = object_path / "metadata.json"
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
            
            # 存储实际数据
            data_type = data.get('type', 'text')
            data_content = data.get('data', '')
            
            if data_type == 'text':
                await self._store_text_data(object_path, data_content)
            elif data_type == 'image':
                await self._store_image_data(object_path, data_content)
            elif data_type == 'video':
                await self._store_video_data(object_path, data_content)
            elif data_type == 'audio':
                await self._store_audio_data(object_path, data_content)
            elif data_type == 'iot':
                await self._store_iot_data(object_path, data_content)
            else:
                await self._store_binary_data(object_path, data_content)
            
            # 更新元数据
            await self._update_metadata(object_path, metadata)
            
            # 更新索引
            self.object_index[object_id] = {
                'path': str(object_path),
                'type': data_type,
                'created_at': metadata['created_at'],
                'size': metadata['size']
            }
            
            self.object_count += 1
            await self._save_index()
            
            logger.debug(f"对象存储成功: {object_id}")
            return True
            
        except Exception as e:
            logger.error(f"存储对象失败: {e}")
            return False
    
    async def _store_text_data(self, object_path: Path, content: str):
        """存储文本数据"""
        data_file = object_path / "data.txt"
        async with aiofiles.open(data_file, 'w', encoding='utf-8') as f:
            await f.write(content)
    
    async def _store_image_data(self, object_path: Path, image_path: str):
        """存储图像数据"""
        # 这里可以添加图像处理逻辑
        data_file = object_path / "data.jpg"
        if Path(image_path).exists():
            async with aiofiles.open(image_path, 'rb') as src:
                async with aiofiles.open(data_file, 'wb') as dst:
                    await dst.write(await src.read())
    
    async def _store_video_data(self, object_path: Path, video_path: str):
        """存储视频数据"""
        data_file = object_path / "data.mp4"
        if Path(video_path).exists():
            async with aiofiles.open(video_path, 'rb') as src:
                async with aiofiles.open(data_file, 'wb') as dst:
                    await dst.write(await src.read())
    
    async def _store_audio_data(self, object_path: Path, audio_path: str):
        """存储音频数据"""
        data_file = object_path / "data.wav"
        if Path(audio_path).exists():
            async with aiofiles.open(audio_path, 'rb') as src:
                async with aiofiles.open(data_file, 'wb') as dst:
                    await dst.write(await src.read())
    
    async def _store_iot_data(self, object_path: Path, data: Dict[str, Any]):
        """存储 IoT 数据"""
        data_file = object_path / "data.json"
        async with aiofiles.open(data_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2))
    
    async def _store_binary_data(self, object_path: Path, data: Any):
        """存储二进制数据"""
        data_file = object_path / "data.bin"
        if isinstance(data, str):
            async with aiofiles.open(data_file, 'w', encoding='utf-8') as f:
                await f.write(data)
        else:
            async with aiofiles.open(data_file, 'wb') as f:
                await f.write(data)
    
    async def _update_metadata(self, object_path: Path, metadata: Dict[str, Any]):
        """更新元数据"""
        try:
            # 计算文件大小
            total_size = 0
            for file_path in object_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            metadata['size'] = total_size
            
            # 计算校验和
            checksum = hashlib.md5()
            for file_path in sorted(object_path.rglob("*")):
                if file_path.is_file():
                    async with aiofiles.open(file_path, 'rb') as f:
                        content = await f.read()
                        checksum.update(content)
            
            metadata['checksum'] = checksum.hexdigest()
            
            # 保存更新后的元数据
            metadata_file = object_path / "metadata.json"
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
                
        except Exception as e:
            logger.error(f"更新元数据失败: {e}")
    
    async def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        获取对象
        
        Args:
            object_id: 对象 ID
        
        Returns:
            对象数据
        """
        try:
            if object_id not in self.object_index:
                return None
            
            object_path = Path(self.object_index[object_id]['path'])
            if not object_path.exists():
                return None
            
            # 读取元数据
            metadata_file = object_path / "metadata.json"
            if not metadata_file.exists():
                return None
            
            async with aiofiles.open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.loads(await f.read())
            
            # 读取数据
            data_type = metadata.get('type', 'text')
            data = None
            
            if data_type == 'text':
                data_file = object_path / "data.txt"
                if data_file.exists():
                    async with aiofiles.open(data_file, 'r', encoding='utf-8') as f:
                        data = await f.read()
            
            elif data_type == 'iot':
                data_file = object_path / "data.json"
                if data_file.exists():
                    async with aiofiles.open(data_file, 'r', encoding='utf-8') as f:
                        data = json.loads(await f.read())
            
            else:
                # 二进制数据
                data_file = object_path / f"data.{self._get_extension(data_type)}"
                if data_file.exists():
                    async with aiofiles.open(data_file, 'rb') as f:
                        data = await f.read()
            
            return {
                'object_id': object_id,
                'type': data_type,
                'data': data,
                'metadata': metadata.get('metadata', {}),
                'tags': metadata.get('tags', []),
                'created_at': metadata.get('created_at', ''),
                'size': metadata.get('size', 0)
            }
            
        except Exception as e:
            logger.error(f"获取对象失败: {e}")
            return None
    
    def _get_extension(self, data_type: str) -> str:
        """获取文件扩展名"""
        extensions = {
            'image': 'jpg',
            'video': 'mp4',
            'audio': 'wav',
            'binary': 'bin'
        }
        return extensions.get(data_type, 'bin')
    
    async def update_object(self, object_id: str, data: Dict[str, Any]) -> bool:
        """
        更新对象
        
        Args:
            object_id: 对象 ID
            data: 更新数据
        
        Returns:
            是否更新成功
        """
        try:
            if object_id not in self.object_index:
                return await self.store_object(object_id, data)
            
            # 获取现有对象
            existing_object = await self.get_object(object_id)
            if not existing_object:
                return False
            
            # 合并数据
            updated_data = {
                'type': data.get('type', existing_object['type']),
                'data': data.get('data', existing_object['data']),
                'metadata': {**existing_object.get('metadata', {}), **data.get('metadata', {})},
                'tags': list(set(existing_object.get('tags', []) + data.get('tags', []))),
                'timestamp': data.get('timestamp', existing_object.get('created_at', ''))
            }
            
            # 重新存储
            return await self.store_object(object_id, updated_data)
            
        except Exception as e:
            logger.error(f"更新对象失败: {e}")
            return False
    
    async def delete_object(self, object_id: str) -> bool:
        """
        删除对象
        
        Args:
            object_id: 对象 ID
        
        Returns:
            是否删除成功
        """
        try:
            if object_id not in self.object_index:
                return True
            
            object_path = Path(self.object_index[object_id]['path'])
            
            # 删除目录
            if object_path.exists():
                await self._delete_directory(object_path)
            
            # 从索引中移除
            del self.object_index[object_id]
            self.object_count -= 1
            
            await self._save_index()
            
            logger.debug(f"对象删除成功: {object_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除对象失败: {e}")
            return False
    
    async def _delete_directory(self, directory: Path):
        """删除目录"""
        try:
            if directory.exists():
                for item in directory.iterdir():
                    if item.is_dir():
                        await self._delete_directory(item)
                    else:
                        item.unlink()
                directory.rmdir()
        except Exception as e:
            logger.error(f"删除目录失败: {e}")
    
    async def get_object_count(self) -> int:
        """获取对象总数"""
        return self.object_count
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            total_size = 0
            type_counts = {}
            
            for object_info in self.object_index.values():
                total_size += object_info.get('size', 0)
                obj_type = object_info.get('type', 'unknown')
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            
            stats = {
                'total_objects': self.object_count,
                'total_size': total_size,
                'type_distribution': type_counts,
                'storage_directory': str(self.object_dir)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取存储统计信息失败: {e}")
            return {}
    
    async def close(self):
        """关闭对象存储"""
        try:
            await self._save_index()
            logger.info("对象存储已关闭")
        except Exception as e:
            logger.error(f"关闭对象存储失败: {e}")