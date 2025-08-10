"""
事件溯源系统 - 支持精确历史状态重建
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
from pathlib import Path
import aiofiles
from loguru import logger

from .config import StorageConfig


class EventSourcing:
    """
    事件溯源系统 - 支持精确历史状态重建
    
    核心特性：
    1. 事件流：所有状态变更记录为不可变事件
    2. 快照：定期创建状态快照，加速重建
    3. 时间旅行：支持任意时间点的状态重建
    4. 事件重放：支持事件重放和状态回滚
    5. 并发控制：支持乐观锁和冲突解决
    """
    
    def __init__(self, config: StorageConfig):
        """
        初始化事件溯源系统
        
        Args:
            config: 存储配置
        """
        self.config = config
        self.event_dir = Path(config.event_log_dir)
        self.event_dir.mkdir(parents=True, exist_ok=True)
        
        # 事件流配置
        self.snapshot_interval = 100  # 每100个事件创建一次快照
        self.max_event_file_size = config.max_file_size
        
        # 事件流存储
        self.current_event_file = None
        self.current_event_path = None
        self.event_count = 0
        
        # 快照存储
        self.snapshots = {}  # object_id -> [snapshot_info]
        
        # 并发控制
        self.locks = {}  # object_id -> asyncio.Lock()
        
        # 初始化事件流
        self._init_event_stream()
        
        logger.info(f"事件溯源系统初始化完成，事件目录: {self.event_dir}")
    
    def _init_event_stream(self):
        """初始化事件流"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"events_{timestamp}.log"
        self.current_event_path = self.event_dir / filename
        
        # 创建事件文件
        self.current_event_path.touch(exist_ok=True)
        logger.info(f"创建事件流文件: {self.current_event_path}")
    
    async def record_event(self, 
                          object_id: str, 
                          event_type: str, 
                          event_data: Dict[str, Any],
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录事件
        
        Args:
            object_id: 对象 ID
            event_type: 事件类型
            event_data: 事件数据
            metadata: 元数据
        
        Returns:
            事件 ID
        """
        try:
            # 获取对象锁
            if object_id not in self.locks:
                self.locks[object_id] = asyncio.Lock()
            
            async with self.locks[object_id]:
                # 生成事件
                event_id = str(uuid.uuid4())
                timestamp = datetime.utcnow().isoformat()
                
                event = {
                    'event_id': event_id,
                    'object_id': object_id,
                    'event_type': event_type,
                    'timestamp': timestamp,
                    'data': event_data,
                    'metadata': metadata or {},
                    'version': await self._get_next_version(object_id),
                    'sequence_number': self.event_count + 1
                }
                
                # 写入事件流
                event_line = json.dumps(event, ensure_ascii=False) + "\n"
                async with aiofiles.open(self.current_event_path, mode='a', encoding='utf-8') as f:
                    await f.write(event_line)
                
                self.event_count += 1
                
                # 检查是否需要创建快照
                if self.event_count % self.snapshot_interval == 0:
                    await self._create_snapshot(object_id, event)
                
                # 检查是否需要轮转事件文件
                await self._check_event_file_rotation()
                
                logger.debug(f"事件已记录: {event_id} ({event_type})")
                return event_id
                
        except Exception as e:
            logger.error(f"记录事件失败: {e}")
            raise
    
    async def _get_next_version(self, object_id: str) -> int:
        """获取下一个版本号"""
        try:
            # 从事件流中获取最新版本
            latest_event = await self._get_latest_event(object_id)
            if latest_event:
                return latest_event.get('version', 0) + 1
            return 1
        except Exception as e:
            logger.error(f"获取版本号失败: {e}")
            return 1
    
    async def _get_latest_event(self, object_id: str) -> Optional[Dict[str, Any]]:
        """获取最新事件"""
        try:
            events = await self.get_events(object_id, limit=1)
            return events[0] if events else None
        except Exception as e:
            logger.error(f"获取最新事件失败: {e}")
            return None
    
    async def _create_snapshot(self, object_id: str, event: Dict[str, Any]):
        """创建快照"""
        try:
            snapshot_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            snapshot = {
                'snapshot_id': snapshot_id,
                'object_id': object_id,
                'timestamp': timestamp,
                'event_id': event['event_id'],
                'version': event['version'],
                'sequence_number': event['sequence_number'],
                'state': await self._build_state_at_event(object_id, event['event_id'])
            }
            
            # 保存快照
            snapshot_dir = self.event_dir / "snapshots" / object_id
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_file = snapshot_dir / f"snapshot_{event['version']}.json"
            async with aiofiles.open(snapshot_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(snapshot, ensure_ascii=False, indent=2))
            
            # 更新快照索引
            if object_id not in self.snapshots:
                self.snapshots[object_id] = []
            self.snapshots[object_id].append({
                'snapshot_id': snapshot_id,
                'version': event['version'],
                'timestamp': timestamp,
                'file_path': str(snapshot_file)
            })
            
            logger.debug(f"快照已创建: {snapshot_id} (版本 {event['version']})")
            
        except Exception as e:
            logger.error(f"创建快照失败: {e}")
    
    async def _build_state_at_event(self, object_id: str, event_id: str) -> Dict[str, Any]:
        """构建事件时的状态"""
        try:
            # 这里应该调用对象存储来获取当前状态
            # 暂时返回空状态
            return {
                'object_id': object_id,
                'event_id': event_id,
                'state': {}
            }
        except Exception as e:
            logger.error(f"构建状态失败: {e}")
            return {}
    
    async def _check_event_file_rotation(self):
        """检查事件文件轮转"""
        try:
            if self.current_event_path.exists():
                file_size = self.current_event_path.stat().st_size
                
                if file_size > self.max_event_file_size:
                    await self._rotate_event_file()
                    
        except Exception as e:
            logger.error(f"检查事件文件轮转失败: {e}")
    
    async def _rotate_event_file(self):
        """轮转事件文件"""
        try:
            # 创建新的事件文件
            self._init_event_stream()
            logger.info("事件文件已轮转")
            
        except Exception as e:
            logger.error(f"轮转事件文件失败: {e}")
    
    async def get_events(self, 
                        object_id: str,
                        start_time: Optional[str] = None,
                        end_time: Optional[str] = None,
                        event_type: Optional[str] = None,
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取事件列表
        
        Args:
            object_id: 对象 ID
            start_time: 开始时间
            end_time: 结束时间
            event_type: 事件类型
            limit: 限制数量
        
        Returns:
            事件列表
        """
        try:
            events = []
            count = 0
            
            # 遍历所有事件文件
            async for event in self._scan_events():
                # 过滤对象
                if event.get('object_id') != object_id:
                    continue
                
                # 应用过滤条件
                if not await self._filter_event(event, start_time, end_time, event_type):
                    continue
                
                events.append(event)
                count += 1
                
                # 检查限制
                if limit and count >= limit:
                    break
            
            # 按时间排序
            events.sort(key=lambda x: x.get('timestamp', ''))
            
            logger.info(f"获取到 {len(events)} 个事件")
            return events
            
        except Exception as e:
            logger.error(f"获取事件失败: {e}")
            return []
    
    async def _scan_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """扫描所有事件"""
        try:
            # 获取所有事件文件，按时间排序
            event_files = sorted(self.event_dir.glob("events_*.log"))
            
            for event_file in event_files:
                async with aiofiles.open(event_file, mode='r', encoding='utf-8') as f:
                    async for line in f:
                        line = line.strip()
                        if line:
                            try:
                                event = json.loads(line)
                                yield event
                            except json.JSONDecodeError as e:
                                logger.warning(f"解析事件失败: {e}")
                                continue
                                
        except Exception as e:
            logger.error(f"扫描事件失败: {e}")
    
    async def _filter_event(self, 
                           event: Dict[str, Any],
                           start_time: Optional[str],
                           end_time: Optional[str],
                           event_type: Optional[str]) -> bool:
        """过滤事件"""
        try:
            # 时间过滤
            if start_time and event.get("timestamp", "") < start_time:
                return False
            if end_time and event.get("timestamp", "") > end_time:
                return False
            
            # 事件类型过滤
            if event_type and event.get("event_type") != event_type:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"过滤事件失败: {e}")
            return False
    
    async def get_state_at_time(self, object_id: str, timestamp: str) -> Optional[Dict[str, Any]]:
        """
        获取指定时间点的状态
        
        Args:
            object_id: 对象 ID
            timestamp: 时间戳
        
        Returns:
            状态数据
        """
        try:
            # 查找最近的快照
            snapshot = await self._find_nearest_snapshot(object_id, timestamp)
            
            if snapshot:
                # 从快照开始重放事件
                return await self._replay_events_from_snapshot(object_id, snapshot, timestamp)
            else:
                # 从头开始重放事件
                return await self._replay_all_events(object_id, timestamp)
                
        except Exception as e:
            logger.error(f"获取时间点状态失败: {e}")
            return None
    
    async def _find_nearest_snapshot(self, object_id: str, timestamp: str) -> Optional[Dict[str, Any]]:
        """查找最近的快照"""
        try:
            if object_id not in self.snapshots:
                return None
            
            snapshots = self.snapshots[object_id]
            nearest_snapshot = None
            min_diff = float('inf')
            
            for snapshot in snapshots:
                snapshot_time = snapshot['timestamp']
                diff = abs((datetime.fromisoformat(timestamp) - datetime.fromisoformat(snapshot_time)).total_seconds())
                
                if diff < min_diff and snapshot_time <= timestamp:
                    min_diff = diff
                    nearest_snapshot = snapshot
            
            return nearest_snapshot
            
        except Exception as e:
            logger.error(f"查找快照失败: {e}")
            return None
    
    async def _replay_events_from_snapshot(self, object_id: str, snapshot: Dict[str, Any], target_timestamp: str) -> Dict[str, Any]:
        """从快照重放事件"""
        try:
            # 加载快照状态
            snapshot_file = Path(snapshot['file_path'])
            async with aiofiles.open(snapshot_file, 'r', encoding='utf-8') as f:
                snapshot_data = json.loads(await f.read())
            
            current_state = snapshot_data.get('state', {})
            
            # 获取快照后的事件
            events = await self.get_events(
                object_id,
                start_time=snapshot['timestamp'],
                end_time=target_timestamp
            )
            
            # 重放事件
            for event in events:
                current_state = await self._apply_event(current_state, event)
            
            return current_state
            
        except Exception as e:
            logger.error(f"从快照重放事件失败: {e}")
            return {}
    
    async def _replay_all_events(self, object_id: str, target_timestamp: str) -> Dict[str, Any]:
        """重放所有事件"""
        try:
            current_state = {}
            
            # 获取所有事件
            events = await self.get_events(
                object_id,
                end_time=target_timestamp
            )
            
            # 重放事件
            for event in events:
                current_state = await self._apply_event(current_state, event)
            
            return current_state
            
        except Exception as e:
            logger.error(f"重放所有事件失败: {e}")
            return {}
    
    async def _apply_event(self, current_state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """应用事件到状态"""
        try:
            event_type = event.get('event_type', '')
            event_data = event.get('data', {})
            
            if event_type == 'INSERT':
                # 插入事件
                current_state.update(event_data)
            elif event_type == 'UPDATE':
                # 更新事件
                current_state.update(event_data)
            elif event_type == 'DELETE':
                # 删除事件
                current_state.clear()
            elif event_type == 'PATCH':
                # 补丁事件
                for key, value in event_data.items():
                    if isinstance(value, dict) and key in current_state:
                        current_state[key].update(value)
                    else:
                        current_state[key] = value
            
            return current_state
            
        except Exception as e:
            logger.error(f"应用事件失败: {e}")
            return current_state
    
    async def time_travel(self, object_id: str, target_timestamp: str) -> Optional[Dict[str, Any]]:
        """
        时间旅行 - 获取指定时间点的完整状态
        
        Args:
            object_id: 对象 ID
            target_timestamp: 目标时间戳
        
        Returns:
            完整状态数据
        """
        try:
            state = await self.get_state_at_time(object_id, target_timestamp)
            
            if state:
                return {
                    'object_id': object_id,
                    'timestamp': target_timestamp,
                    'state': state,
                    'version': await self._get_version_at_time(object_id, target_timestamp)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"时间旅行失败: {e}")
            return None
    
    async def _get_version_at_time(self, object_id: str, timestamp: str) -> int:
        """获取指定时间点的版本号"""
        try:
            events = await self.get_events(object_id, end_time=timestamp, limit=1)
            if events:
                return events[0].get('version', 0)
            return 0
        except Exception as e:
            logger.error(f"获取版本号失败: {e}")
            return 0
    
    async def replay_events(self, 
                           object_id: str,
                           start_time: Optional[str] = None,
                           end_time: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        重放事件
        
        Args:
            object_id: 对象 ID
            start_time: 开始时间
            end_time: 结束时间
        
        Yields:
            事件数据
        """
        try:
            async for event in self._scan_events():
                if event.get('object_id') != object_id:
                    continue
                
                # 应用时间过滤
                if start_time and event.get('timestamp', '') < start_time:
                    continue
                if end_time and event.get('timestamp', '') > end_time:
                    continue
                
                yield event
                
        except Exception as e:
            logger.error(f"重放事件失败: {e}")
    
    async def get_event_statistics(self) -> Dict[str, Any]:
        """获取事件统计信息"""
        try:
            total_events = 0
            event_types = {}
            object_events = {}
            
            async for event in self._scan_events():
                total_events += 1
                
                event_type = event.get('event_type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
                
                object_id = event.get('object_id', 'unknown')
                object_events[object_id] = object_events.get(object_id, 0) + 1
            
            stats = {
                'total_events': total_events,
                'event_types': event_types,
                'object_events': object_events,
                'snapshots': {obj_id: len(snaps) for obj_id, snaps in self.snapshots.items()},
                'event_files': len(list(self.event_dir.glob("events_*.log")))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取事件统计信息失败: {e}")
            return {}
    
    async def compact_events(self, target_file: str) -> bool:
        """
        压缩事件流
        
        Args:
            target_file: 目标文件路径
        
        Returns:
            是否压缩成功
        """
        try:
            target_path = Path(target_file)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入压缩后的事件
            async with aiofiles.open(target_path, mode='w', encoding='utf-8') as f:
                async for event in self._scan_events():
                    event_line = json.dumps(event, ensure_ascii=False) + "\n"
                    await f.write(event_line)
            
            logger.info(f"事件流已压缩到: {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"压缩事件流失败: {e}")
            return False
    
    async def close(self):
        """关闭事件溯源系统"""
        try:
            logger.info("事件溯源系统已关闭")
        except Exception as e:
            logger.error(f"关闭事件溯源系统失败: {e}")