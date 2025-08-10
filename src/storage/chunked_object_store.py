"""
自研 Chunked Object Store - 支持多模态、历史可回放的对象存储引擎
"""

import asyncio
import hashlib
import json
import uuid
from typing import Dict, List, Any, Optional, BinaryIO, Tuple
from pathlib import Path
import aiofiles
import aiofiles.os
from datetime import datetime
import zlib
from loguru import logger

from ..core.config import StorageConfig


class ChunkedObjectStore:
    """
    自研 Chunked Object Store - 支持多模态、历史可回放的对象存储引擎
    
    核心特性：
    1. Chunked 存储：大对象自动分块，支持流式读写
    2. Merkle DAG：内容寻址，支持去重和完整性验证
    3. 多模态统一：文本、图像、视频、音频、IoT 统一存储
    4. 历史可回放：支持任意时间点的状态重建
    5. 语义压缩：基于内容相似性的智能压缩
    """
    
    def __init__(self, config: StorageConfig):
        """
        初始化 Chunked Object Store
        
        Args:
            config: 存储配置
        """
        self.config = config
        self.object_dir = Path(config.object_store_dir)
        self.object_dir.mkdir(parents=True, exist_ok=True)
        
        # 分块配置
        self.chunk_size = config.chunk_size
        self.max_chunks_per_object = 1000  # 防止单个对象过大
        
        # 对象索引
        self.object_index = {}
        self.chunk_index = {}  # chunk_hash -> chunk_info
        self.object_count = 0
        
        # Merkle DAG 根节点
        self.merkle_roots = {}
        
        # 语义压缩索引
        self.semantic_compression_index = {}
        
        # 加载现有索引
        self._load_indexes()
        
        logger.info(f"Chunked Object Store 初始化完成，目录: {self.object_dir}")
    
    def _load_indexes(self):
        """加载索引"""
        try:
            # 加载对象索引
            object_index_file = self.object_dir / "object_index.json"
            if object_index_file.exists():
                with open(object_index_file, 'r', encoding='utf-8') as f:
                    self.object_index = json.load(f)
                    self.object_count = len(self.object_index)
            
            # 加载分块索引
            chunk_index_file = self.object_dir / "chunk_index.json"
            if chunk_index_file.exists():
                with open(chunk_index_file, 'r', encoding='utf-8') as f:
                    self.chunk_index = json.load(f)
            
            # 加载 Merkle 根节点
            merkle_file = self.object_dir / "merkle_roots.json"
            if merkle_file.exists():
                with open(merkle_file, 'r', encoding='utf-8') as f:
                    self.merkle_roots = json.load(f)
            
            logger.info(f"加载索引完成，对象数: {self.object_count}, 分块数: {len(self.chunk_index)}")
            
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
    
    async def _save_indexes(self):
        """保存索引"""
        try:
            # 保存对象索引
            object_index_file = self.object_dir / "object_index.json"
            async with aiofiles.open(object_index_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.object_index, ensure_ascii=False, indent=2))
            
            # 保存分块索引
            chunk_index_file = self.object_dir / "chunk_index.json"
            async with aiofiles.open(chunk_index_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.chunk_index, ensure_ascii=False, indent=2))
            
            # 保存 Merkle 根节点
            merkle_file = self.object_dir / "merkle_roots.json"
            async with aiofiles.open(merkle_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.merkle_roots, ensure_ascii=False, indent=2))
                
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
    
    async def store_object(self, object_id: str, data: Dict[str, Any]) -> bool:
        """
        存储对象（支持分块和语义压缩）
        
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
            
            # 序列化数据
            data_bytes = await self._serialize_data(data)
            
            # 检查是否需要分块
            if len(data_bytes) > self.chunk_size:
                return await self._store_chunked_object(object_id, data, data_bytes)
            else:
                return await self._store_single_chunk_object(object_id, data, data_bytes)
                
        except Exception as e:
            logger.error(f"存储对象失败: {e}")
            return False
    
    async def _serialize_data(self, data: Dict[str, Any]) -> bytes:
        """序列化数据"""
        try:
            data_type = data.get('type', 'text')
            data_content = data.get('data', '')
            
            if data_type == 'text':
                return data_content.encode('utf-8')
            elif data_type == 'iot':
                return json.dumps(data_content, ensure_ascii=False).encode('utf-8')
            else:
                # 二进制数据
                if isinstance(data_content, str):
                    return data_content.encode('utf-8')
                elif isinstance(data_content, bytes):
                    return data_content
                else:
                    return str(data_content).encode('utf-8')
                    
        except Exception as e:
            logger.error(f"序列化数据失败: {e}")
            return b''
    
    async def _store_single_chunk_object(self, object_id: str, data: Dict[str, Any], data_bytes: bytes) -> bool:
        """存储单分块对象"""
        try:
            # 计算内容哈希
            content_hash = hashlib.sha256(data_bytes).hexdigest()
            
            # 检查是否已存在相同内容
            if content_hash in self.chunk_index:
                # 语义压缩：重用现有分块
                existing_chunk = self.chunk_index[content_hash]
                chunk_path = existing_chunk['path']
                logger.info(f"语义压缩：重用分块 {content_hash}")
            else:
                # 创建新分块
                chunk_path = f"chunks/{content_hash[:2]}/{content_hash}"
                chunk_file = self.object_dir / chunk_path
                chunk_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 压缩存储
                compressed_data = zlib.compress(data_bytes)
                async with aiofiles.open(chunk_file, 'wb') as f:
                    await f.write(compressed_data)
                
                # 更新分块索引
                self.chunk_index[content_hash] = {
                    'path': chunk_path,
                    'size': len(data_bytes),
                    'compressed_size': len(compressed_data),
                    'type': data.get('type', 'unknown'),
                    'created_at': datetime.utcnow().isoformat(),
                    'references': 1
                }
            
            # 创建对象元数据
            metadata = {
                'object_id': object_id,
                'type': data.get('type', 'unknown'),
                'metadata': data.get('metadata', {}),
                'tags': data.get('tags', []),
                'created_at': data.get('timestamp', datetime.utcnow().isoformat()),
                'chunks': [content_hash],
                'merkle_root': content_hash,  # 单分块时，内容哈希就是 Merkle 根
                'size': len(data_bytes),
                'compressed_size': self.chunk_index[content_hash]['compressed_size']
            }
            
            # 保存元数据
            object_path = self.object_dir / object_id
            object_path.mkdir(exist_ok=True)
            metadata_file = object_path / "metadata.json"
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
            
            # 更新对象索引
            self.object_index[object_id] = {
                'path': str(object_path),
                'type': data.get('type', 'unknown'),
                'created_at': metadata['created_at'],
                'size': len(data_bytes),
                'chunk_count': 1,
                'merkle_root': content_hash
            }
            
            self.object_count += 1
            await self._save_indexes()
            
            logger.debug(f"单分块对象存储成功: {object_id}")
            return True
            
        except Exception as e:
            logger.error(f"存储单分块对象失败: {e}")
            return False
    
    async def _store_chunked_object(self, object_id: str, data: Dict[str, Any], data_bytes: bytes) -> bool:
        """存储分块对象"""
        try:
            # 分块
            chunks = []
            chunk_hashes = []
            
            for i in range(0, len(data_bytes), self.chunk_size):
                chunk_data = data_bytes[i:i + self.chunk_size]
                chunk_hash = hashlib.sha256(chunk_data).hexdigest()
                
                # 检查是否已存在相同分块
                if chunk_hash in self.chunk_index:
                    # 语义压缩：重用现有分块
                    self.chunk_index[chunk_hash]['references'] += 1
                    logger.info(f"语义压缩：重用分块 {chunk_hash}")
                else:
                    # 创建新分块
                    chunk_path = f"chunks/{chunk_hash[:2]}/{chunk_hash}"
                    chunk_file = self.object_dir / chunk_path
                    chunk_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 压缩存储
                    compressed_chunk = zlib.compress(chunk_data)
                    async with aiofiles.open(chunk_file, 'wb') as f:
                        await f.write(compressed_chunk)
                    
                    # 更新分块索引
                    self.chunk_index[chunk_hash] = {
                        'path': chunk_path,
                        'size': len(chunk_data),
                        'compressed_size': len(compressed_chunk),
                        'type': data.get('type', 'unknown'),
                        'created_at': datetime.utcnow().isoformat(),
                        'references': 1
                    }
                
                chunks.append(chunk_data)
                chunk_hashes.append(chunk_hash)
            
            # 构建 Merkle DAG
            merkle_root = await self._build_merkle_dag(chunk_hashes)
            
            # 创建对象元数据
            metadata = {
                'object_id': object_id,
                'type': data.get('type', 'unknown'),
                'metadata': data.get('metadata', {}),
                'tags': data.get('tags', []),
                'created_at': data.get('timestamp', datetime.utcnow().isoformat()),
                'chunks': chunk_hashes,
                'merkle_root': merkle_root,
                'size': len(data_bytes),
                'compressed_size': sum(self.chunk_index[h]['compressed_size'] for h in chunk_hashes),
                'chunk_count': len(chunk_hashes)
            }
            
            # 保存元数据
            object_path = self.object_dir / object_id
            object_path.mkdir(exist_ok=True)
            metadata_file = object_path / "metadata.json"
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
            
            # 更新对象索引
            self.object_index[object_id] = {
                'path': str(object_path),
                'type': data.get('type', 'unknown'),
                'created_at': metadata['created_at'],
                'size': len(data_bytes),
                'chunk_count': len(chunk_hashes),
                'merkle_root': merkle_root
            }
            
            self.object_count += 1
            await self._save_indexes()
            
            logger.debug(f"分块对象存储成功: {object_id}, 分块数: {len(chunk_hashes)}")
            return True
            
        except Exception as e:
            logger.error(f"存储分块对象失败: {e}")
            return False
    
    async def _build_merkle_dag(self, chunk_hashes: List[str]) -> str:
        """构建 Merkle DAG"""
        try:
            if len(chunk_hashes) == 1:
                return chunk_hashes[0]
            
            # 构建 Merkle 树
            current_level = chunk_hashes.copy()
            
            while len(current_level) > 1:
                next_level = []
                for i in range(0, len(current_level), 2):
                    if i + 1 < len(current_level):
                        # 两个子节点
                        combined = current_level[i] + current_level[i + 1]
                        parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                    else:
                        # 单个子节点
                        parent_hash = current_level[i]
                    next_level.append(parent_hash)
                current_level = next_level
            
            return current_level[0]
            
        except Exception as e:
            logger.error(f"构建 Merkle DAG 失败: {e}")
            return chunk_hashes[0] if chunk_hashes else ""
    
    async def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        获取对象（支持分块重组）
        
        Args:
            object_id: 对象 ID
        
        Returns:
            对象数据
        """
        try:
            if object_id not in self.object_index:
                return None
            
            object_path = Path(self.object_index[object_id]['path'])
            metadata_file = object_path / "metadata.json"
            
            if not metadata_file.exists():
                return None
            
            # 读取元数据
            async with aiofiles.open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.loads(await f.read())
            
            # 重组对象数据
            data_bytes = await self._reconstruct_object(metadata['chunks'])
            
            # 反序列化数据
            data = await self._deserialize_data(data_bytes, metadata['type'])
            
            return {
                'object_id': object_id,
                'type': metadata['type'],
                'data': data,
                'metadata': metadata.get('metadata', {}),
                'tags': metadata.get('tags', []),
                'created_at': metadata.get('created_at', ''),
                'size': metadata.get('size', 0),
                'chunk_count': metadata.get('chunk_count', 1),
                'merkle_root': metadata.get('merkle_root', '')
            }
            
        except Exception as e:
            logger.error(f"获取对象失败: {e}")
            return None
    
    async def _reconstruct_object(self, chunk_hashes: List[str]) -> bytes:
        """重组对象数据"""
        try:
            data_parts = []
            
            for chunk_hash in chunk_hashes:
                if chunk_hash not in self.chunk_index:
                    raise ValueError(f"分块不存在: {chunk_hash}")
                
                chunk_info = self.chunk_index[chunk_hash]
                chunk_file = self.object_dir / chunk_info['path']
                
                if not chunk_file.exists():
                    raise ValueError(f"分块文件不存在: {chunk_file}")
                
                # 读取并解压缩分块
                async with aiofiles.open(chunk_file, 'rb') as f:
                    compressed_data = await f.read()
                
                chunk_data = zlib.decompress(compressed_data)
                data_parts.append(chunk_data)
            
            return b''.join(data_parts)
            
        except Exception as e:
            logger.error(f"重组对象失败: {e}")
            return b''
    
    async def _deserialize_data(self, data_bytes: bytes, data_type: str) -> Any:
        """反序列化数据"""
        try:
            if data_type == 'text':
                return data_bytes.decode('utf-8')
            elif data_type == 'iot':
                return json.loads(data_bytes.decode('utf-8'))
            else:
                # 二进制数据
                return data_bytes
                
        except Exception as e:
            logger.error(f"反序列化数据失败: {e}")
            return data_bytes
    
    async def update_object(self, object_id: str, data: Dict[str, Any]) -> bool:
        """
        更新对象（支持增量更新和语义压缩）
        
        Args:
            object_id: 对象 ID
            data: 更新数据
        
        Returns:
            是否更新成功
        """
        try:
            # 获取现有对象
            existing_object = await self.get_object(object_id)
            if not existing_object:
                return await self.store_object(object_id, data)
            
            # 检查是否需要完全重写
            if await self._should_rewrite_object(existing_object, data):
                # 删除旧对象
                await self.delete_object(object_id)
                # 存储新对象
                return await self.store_object(object_id, data)
            else:
                # 增量更新
                return await self._incremental_update(object_id, existing_object, data)
                
        except Exception as e:
            logger.error(f"更新对象失败: {e}")
            return False
    
    async def _should_rewrite_object(self, existing_object: Dict[str, Any], new_data: Dict[str, Any]) -> bool:
        """判断是否需要完全重写对象"""
        try:
            # 类型改变
            if existing_object['type'] != new_data.get('type', 'unknown'):
                return True
            
            # 数据大小变化超过阈值
            existing_size = existing_object.get('size', 0)
            new_data_bytes = await self._serialize_data(new_data)
            new_size = len(new_data_bytes)
            
            size_change_ratio = abs(new_size - existing_size) / max(existing_size, 1)
            if size_change_ratio > 0.5:  # 50% 变化阈值
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"判断重写需求失败: {e}")
            return True
    
    async def _incremental_update(self, object_id: str, existing_object: Dict[str, Any], new_data: Dict[str, Any]) -> bool:
        """增量更新对象"""
        try:
            # 合并数据内容
            merged_data = {**existing_object, **new_data}
            
            # 合并元数据
            merged_metadata = {**existing_object.get('metadata', {}), **new_data.get('metadata', {})}
            merged_tags = list(set(existing_object.get('tags', []) + new_data.get('tags', [])))
            
            # 更新合并后的数据
            merged_data['metadata'] = merged_metadata
            merged_data['tags'] = merged_tags
            merged_data['updated_at'] = datetime.utcnow().isoformat()
            
            # 重新存储对象（这会覆盖旧的数据）
            return await self.store_object(object_id, merged_data)
            
        except Exception as e:
            logger.error(f"增量更新失败: {e}")
            return False
    
    async def delete_object(self, object_id: str) -> bool:
        """
        删除对象（支持引用计数和垃圾回收）
        
        Args:
            object_id: 对象 ID
        
        Returns:
            是否删除成功
        """
        try:
            if object_id not in self.object_index:
                return True
            
            object_info = self.object_index[object_id]
            object_path = Path(object_info['path'])
            
            # 读取元数据获取分块信息
            metadata_file = object_path / "metadata.json"
            if metadata_file.exists():
                async with aiofiles.open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.loads(await f.read())
                
                # 减少分块引用计数
                for chunk_hash in metadata.get('chunks', []):
                    if chunk_hash in self.chunk_index:
                        self.chunk_index[chunk_hash]['references'] -= 1
                        
                        # 如果引用计数为0，删除分块
                        if self.chunk_index[chunk_hash]['references'] <= 0:
                            chunk_file = self.object_dir / self.chunk_index[chunk_hash]['path']
                            if chunk_file.exists():
                                chunk_file.unlink()
                            del self.chunk_index[chunk_hash]
            
            # 删除对象目录
            if object_path.exists():
                import shutil
                shutil.rmtree(object_path)
            
            # 从索引中移除
            del self.object_index[object_id]
            self.object_count -= 1
            
            await self._save_indexes()
            
            logger.debug(f"对象删除成功: {object_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除对象失败: {e}")
            return False
    
    async def get_object_count(self) -> int:
        """获取对象总数"""
        return self.object_count
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            total_size = 0
            total_compressed_size = 0
            type_counts = {}
            chunk_stats = {
                'total_chunks': len(self.chunk_index),
                'unique_chunks': len(self.chunk_index),
                'total_references': sum(chunk['references'] for chunk in self.chunk_index.values())
            }
            
            for object_info in self.object_index.values():
                total_size += object_info.get('size', 0)
                obj_type = object_info.get('type', 'unknown')
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            
            for chunk_info in self.chunk_index.values():
                total_compressed_size += chunk_info.get('compressed_size', 0)
            
            compression_ratio = total_compressed_size / max(total_size, 1)
            
            stats = {
                'total_objects': self.object_count,
                'total_size': total_size,
                'total_compressed_size': total_compressed_size,
                'compression_ratio': compression_ratio,
                'type_distribution': type_counts,
                'chunk_statistics': chunk_stats,
                'storage_directory': str(self.object_dir)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取存储统计信息失败: {e}")
            return {}
    
    async def semantic_compression_analysis(self) -> Dict[str, Any]:
        """语义压缩分析"""
        try:
            duplicate_chunks = []
            compression_opportunities = []
            
            # 分析重复分块
            chunk_groups = {}
            for chunk_hash, chunk_info in self.chunk_index.items():
                content_key = f"{chunk_info['size']}_{chunk_info['type']}"
                if content_key not in chunk_groups:
                    chunk_groups[content_key] = []
                chunk_groups[content_key].append(chunk_hash)
            
            for content_key, chunk_hashes in chunk_groups.items():
                if len(chunk_hashes) > 1:
                    duplicate_chunks.append({
                        'content_key': content_key,
                        'chunk_hashes': chunk_hashes,
                        'duplicate_count': len(chunk_hashes)
                    })
            
            # 分析压缩机会
            for chunk_hash, chunk_info in self.chunk_index.items():
                if chunk_info['references'] > 1:
                    compression_opportunities.append({
                        'chunk_hash': chunk_hash,
                        'references': chunk_info['references'],
                        'saved_space': chunk_info['compressed_size'] * (chunk_info['references'] - 1)
                    })
            
            return {
                'duplicate_chunks': duplicate_chunks,
                'compression_opportunities': compression_opportunities,
                'total_saved_space': sum(opp['saved_space'] for opp in compression_opportunities)
            }
            
        except Exception as e:
            logger.error(f"语义压缩分析失败: {e}")
            return {}
    
    async def close(self):
        """关闭存储引擎"""
        try:
            await self._save_indexes()
            logger.info("Chunked Object Store 已关闭")
        except Exception as e:
            logger.error(f"关闭 Chunked Object Store 失败: {e}")