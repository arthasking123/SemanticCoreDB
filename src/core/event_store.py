"""
事件存储模块 - 实现事件溯源功能
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
from pathlib import Path
import aiofiles
from loguru import logger

from .config import StorageConfig


class EventStore:
    """
    事件存储引擎 - 实现 Append-only 事件日志
    """
    
    def __init__(self, config: StorageConfig):
        """
        初始化事件存储
        
        Args:
            config: 存储配置
        """
        self.config = config
        self.event_log_dir = Path(config.event_log_dir)
        self.event_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前事件日志文件
        self.current_log_file = None
        self.current_log_path = None
        self.event_count = 0
        
        # 初始化当前日志文件
        self._init_current_log_file()
        
        logger.info(f"事件存储初始化完成，日志目录: {self.event_log_dir}")
    
    def _init_current_log_file(self):
        """初始化当前日志文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"events_{timestamp}.log"
        self.current_log_path = self.event_log_dir / filename
        
        # 创建日志文件
        self.current_log_path.touch(exist_ok=True)
        logger.info(f"创建事件日志文件: {self.current_log_path}")
    
    async def append_event(self, event: Dict[str, Any]) -> bool:
        """
        追加事件到日志
        
        Args:
            event: 事件数据
        
        Returns:
            是否追加成功
        """
        try:
            # 确保事件有必要的字段
            if "event_id" not in event:
                event["event_id"] = str(uuid.uuid4())
            if "timestamp" not in event:
                event["timestamp"] = datetime.utcnow().isoformat()
            
            # 序列化事件
            event_line = json.dumps(event, ensure_ascii=False) + "\n"
            
            # 写入日志文件
            async with aiofiles.open(self.current_log_path, mode='a', encoding='utf-8') as f:
                await f.write(event_line)
            
            self.event_count += 1
            
            # 检查是否需要轮转日志文件
            await self._check_log_rotation()
            
            logger.debug(f"事件已追加: {event['event_id']}")
            return True
            
        except Exception as e:
            logger.error(f"追加事件失败: {e}")
            return False
    
    async def _check_log_rotation(self):
        """检查是否需要轮转日志文件"""
        try:
            # 检查当前文件大小
            if self.current_log_path.exists():
                file_size = self.current_log_path.stat().st_size
                
                # 如果文件超过最大大小，创建新文件
                if file_size > self.config.max_file_size:
                    await self._rotate_log_file()
                    
        except Exception as e:
            logger.error(f"检查日志轮转失败: {e}")
    
    async def _rotate_log_file(self):
        """轮转日志文件"""
        try:
            # 关闭当前文件
            if self.current_log_file:
                self.current_log_file.close()
            
            # 创建新的日志文件
            self._init_current_log_file()
            
            logger.info("事件日志文件已轮转")
            
        except Exception as e:
            logger.error(f"轮转日志文件失败: {e}")
    
    async def get_events(self, 
                        start_time: Optional[str] = None,
                        end_time: Optional[str] = None,
                        event_type: Optional[str] = None,
                        object_id: Optional[str] = None,
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取事件列表
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            event_type: 事件类型
            object_id: 对象 ID
            limit: 限制数量
        
        Returns:
            事件列表
        """
        try:
            events = []
            count = 0
            
            # 遍历所有日志文件
            async for event in self._scan_events():
                # 应用过滤条件
                if not await self._filter_event(event, start_time, end_time, event_type, object_id):
                    continue
                
                events.append(event)
                count += 1
                
                # 检查限制
                if limit and count >= limit:
                    break
            
            logger.info(f"获取到 {len(events)} 个事件")
            return events
            
        except Exception as e:
            logger.error(f"获取事件失败: {e}")
            return []
    
    async def _scan_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """扫描所有事件"""
        try:
            # 获取所有日志文件，按时间排序
            log_files = sorted(self.event_log_dir.glob("events_*.log"))
            
            for log_file in log_files:
                async with aiofiles.open(log_file, mode='r', encoding='utf-8') as f:
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
                           event_type: Optional[str],
                           object_id: Optional[str]) -> bool:
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
            
            # 对象 ID 过滤
            if object_id and event.get("object_id") != object_id:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"过滤事件失败: {e}")
            return False
    
    async def get_event_count(self) -> int:
        """获取事件总数"""
        try:
            count = 0
            async for _ in self._scan_events():
                count += 1
            return count
        except Exception as e:
            logger.error(f"获取事件总数失败: {e}")
            return 0
    
    async def replay_events(self, 
                           start_time: Optional[str] = None,
                           end_time: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        重放事件
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Yields:
            事件数据
        """
        try:
            async for event in self._scan_events():
                # 应用时间过滤
                if start_time and event.get("timestamp", "") < start_time:
                    continue
                if end_time and event.get("timestamp", "") > end_time:
                    continue
                
                yield event
                
        except Exception as e:
            logger.error(f"重放事件失败: {e}")
    
    async def compact_events(self, target_file: str) -> bool:
        """
        压缩事件日志
        
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
            
            logger.info(f"事件日志已压缩到: {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"压缩事件日志失败: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """获取事件存储统计信息"""
        try:
            stats = {
                "event_count": await self.get_event_count(),
                "log_files": len(list(self.event_log_dir.glob("events_*.log"))),
                "current_log_file": str(self.current_log_path),
                "log_directory": str(self.event_log_dir),
                "max_file_size": self.config.max_file_size
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取事件存储统计信息失败: {e}")
            return {}
    
    async def close(self):
        """关闭事件存储"""
        try:
            # 这里可以做一些清理工作
            logger.info("事件存储已关闭")
        except Exception as e:
            logger.error(f"关闭事件存储失败: {e}") 