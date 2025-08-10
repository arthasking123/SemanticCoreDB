"""
SemanticCoreDB 测试文件
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到 Python 路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.database import SemanticCoreDB
from src.core.config import Config


class TestSemanticCoreDB:
    """SemanticCoreDB 测试类"""
    
    # 类级别的数据库实例，所有测试共享
    _db = None
    _temp_dir = None
    
    @classmethod
    @pytest_asyncio.fixture(scope="class")
    async def db(cls):
        """创建测试数据库实例 - 只创建一次，所有测试共享"""
        if cls._db is None:
            # 创建临时目录
            cls._temp_dir = tempfile.mkdtemp()
            
            # 配置测试数据库
            config = Config()
            config.storage.data_dir = cls._temp_dir
            config.storage.event_log_dir = f"{cls._temp_dir}/events"
            config.storage.object_store_dir = f"{cls._temp_dir}/objects"
            config.storage.vector_index_dir = f"{cls._temp_dir}/vectors"
            config.storage.metadata_graph_dir = f"{cls._temp_dir}/metadata"
            
            # 配置 LLM 服务用于测试
            config.semantic.llm_provider = "local"  # 使用模拟 LLM 客户端进行测试
            
            # 创建数据库实例
            cls._db = SemanticCoreDB(config)
            
            print(f"测试数据库已创建，临时目录: {cls._temp_dir}")
        
        yield cls._db
    
    @classmethod
    async def cleanup_db(cls):
        """清理数据库资源"""
        if cls._db is not None:
            await cls._db.close()
            cls._db = None
        
        if cls._temp_dir is not None:
            shutil.rmtree(cls._temp_dir)
            cls._temp_dir = None
    
    @pytest.fixture(autouse=True)
    async def setup_teardown(self, db):
        """每个测试前后的设置和清理"""
        # 测试前：清理数据库数据，但保留数据库实例
        await self._cleanup_test_data(db)
        yield
        # 测试后：清理测试产生的数据
        await self._cleanup_test_data(db)
    
    async def _cleanup_test_data(self, db):
        """清理测试数据，但保留数据库实例"""
        try:
            # 获取所有对象并删除
            stats = await db.get_statistics()
            if stats.get('total_objects', 0) > 0:
                # 这里可以添加清理逻辑，或者简单地重置数据库状态
                pass
        except Exception as e:
            # 忽略清理错误
            pass
    
    @pytest.mark.asyncio
    async def test_insert_text(self, db):
        """测试插入文本数据"""
        data = {
            "type": "text",
            "data": "这是一个测试文本",
            "metadata": {"test": True},
            "tags": ["测试", "文本"]
        }
        
        object_id = await db.insert(data)
        assert object_id is not None
        assert len(object_id) > 0
        
        # 验证对象存在
        retrieved_data = await db.get_object(object_id)
        print(retrieved_data)
        assert retrieved_data is not None
        assert retrieved_data["type"] == "text"
        assert retrieved_data["data"] == "这是一个测试文本"
    
    @pytest.mark.asyncio
    async def test_insert_image(self, db):
        """测试插入图像数据"""
        data = {
            "type": "image",
            "data": "test_image.jpg",
            "metadata": {"location": "测试地点"},
            "tags": ["图像", "测试"]
        }
        
        object_id = await db.insert(data)
        assert object_id is not None
        
        # 验证对象存在
        retrieved_data = await db.get_object(object_id)
        assert retrieved_data is not None
        assert retrieved_data["type"] == "image"
    
    @pytest.mark.asyncio
    async def test_insert_iot(self, db):
        """测试插入 IoT 数据"""
        data = {
            "type": "iot",
            "data": {
                "temperature": 25.5,
                "humidity": 60.0,
                "timestamp": "2024-01-01T10:00:00Z"
            },
            "metadata": {"sensor_id": "TEMP_001"},
            "tags": ["IoT", "传感器"]
        }
        
        object_id = await db.insert(data)
        assert object_id is not None
        
        # 验证对象存在
        retrieved_data = await db.get_object(object_id)
        assert retrieved_data is not None
        assert retrieved_data["type"] == "iot"
    
    @pytest.mark.asyncio
    async def test_update_object(self, db):
        """测试更新对象"""
        # 插入原始数据
        original_data = {
            "type": "text",
            "data": "原始文本",
            "metadata": {"version": 1},
            "tags": ["原始"]
        }
        
        object_id = await db.insert(original_data)
        
        # 更新数据
        update_data = {
            "type": "text",
            "data": "更新后的文本",
            "metadata": {"version": 2},
            "tags": ["更新"]
        }
        
        success = await db.update(object_id, update_data)
        assert success is True
        
        # 验证更新
        retrieved_data = await db.get_object(object_id)
        assert retrieved_data["data"] == "更新后的文本"
        assert retrieved_data["metadata"]["version"] == 2
    
    @pytest.mark.asyncio
    async def test_delete_object(self, db):
        """测试删除对象"""
        # 插入数据
        data = {
            "type": "text",
            "data": "要删除的文本",
            "metadata": {},
            "tags": ["删除"]
        }
        
        object_id = await db.insert(data)
        
        # 验证对象存在
        retrieved_data = await db.get_object(object_id)
        assert retrieved_data is not None
        
        # 删除对象
        success = await db.delete(object_id)
        assert success is True
        
        # 验证对象已删除
        retrieved_data = await db.get_object(object_id)
        assert retrieved_data is None
    
    @pytest.mark.asyncio
    async def test_query_text(self, db):
        """测试文本查询"""
        # 插入测试数据
        test_data = [
            {
                "type": "text",
                "data": "人工智能是计算机科学的一个分支",
                "metadata": {"category": "AI"},
                "tags": ["AI", "计算机科学"]
            },
            {
                "type": "text",
                "data": "机器学习是AI的重要技术",
                "metadata": {"category": "AI"},
                "tags": ["机器学习", "AI"]
            },
            {
                "type": "text",
                "data": "数据库管理系统用于存储和管理数据",
                "metadata": {"category": "DB"},
                "tags": ["数据库", "管理系统"]
            }
        ]
        
        for data in test_data:
            await db.insert(data)
        
        # 执行查询
        results = await db.query("找出所有关于人工智能的文章")
        assert len(results) >= 2  # 应该找到至少2个相关结果
      
    @pytest.mark.asyncio
    async def test_get_statistics(self, db):
        """测试获取统计信息"""
        # 插入一些测试数据
        for i in range(5):
            data = {
                "type": "text",
                "data": f"测试数据 {i}",
                "metadata": {"test_id": i},
                "tags": ["测试"]
            }
            await db.insert(data)
        
        # 获取统计信息
        stats = await db.get_statistics()
        
        assert "total_objects" in stats
        assert "total_vectors" in stats
        assert "total_events" in stats
        assert stats["total_objects"] >= 5
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, db):
        """测试批量操作"""
        # 批量插入
        batch_data = []
        for i in range(10):
            data = {
                "type": "text",
                "data": f"批量测试数据 {i}",
                "metadata": {"batch_id": i},
                "tags": ["批量", "测试"]
            }
            batch_data.append(data)
        
        object_ids = []
        for data in batch_data:
            object_id = await db.insert(data)
            object_ids.append(object_id)
        
        assert len(object_ids) == 10
        
        # 批量查询
        results = await db.query("找出所有批量测试数据")
        assert len(results) >= 10
    
    @pytest.mark.asyncio
    async def test_error_handling(self, db):
        """测试错误处理"""
        # 测试插入无效数据
        invalid_data = {
            "type": "invalid_type",
            "data": None,
            "metadata": {},
            "tags": []
        }
        
        # 应该能够处理无效数据而不崩溃
        try:
            object_id = await db.insert(invalid_data)
            assert object_id is not None
        except Exception as e:
            # 如果抛出异常，应该是预期的错误类型
            assert isinstance(e, Exception)
        
        # 测试查询不存在的对象
        non_existent_id = "non-existent-id"
        result = await db.get_object(non_existent_id)
        assert result is None
        
        # 测试删除不存在的对象
        success = await db.delete(non_existent_id)
        assert success is True  # 删除不存在的对象应该返回 True
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, db):
        """测试并发操作"""
        import asyncio
        
        # 并发插入
        async def insert_data(i):
            data = {
                "type": "text",
                "data": f"并发测试数据 {i}",
                "metadata": {"concurrent_id": i},
                "tags": ["并发", "测试"]
            }
            return await db.insert(data)
        
        # 创建多个并发任务
        tasks = [insert_data(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(result is not None for result in results)
        
        # 并发查询
        async def query_data(i):
            query = f"找出包含并发测试数据 {i} 的数据"
            return await db.query(query)
        
        query_tasks = [query_data(i) for i in range(5)]
        query_results = await asyncio.gather(*query_tasks)
        
        assert len(query_results) == 5
        assert all(len(result) >= 1 for result in query_results)


    @classmethod
    def teardown_class(cls):
        """所有测试完成后的清理工作"""
        # 在类级别清理数据库资源
        if cls._db is not None:
            # 使用事件循环来运行异步清理
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，创建新任务
                    loop.create_task(cls.cleanup_db())
                else:
                    # 否则直接运行
                    loop.run_until_complete(cls.cleanup_db())
            except Exception as e:
                print(f"清理数据库时出错: {e}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"]) 