"""
SemanticCoreDB 基本使用示例
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.database import SemanticCoreDB
from src.core.config import Config


async def basic_example():
    """基本使用示例"""
    print("=== SemanticCoreDB 基本使用示例 ===\n")
    
    # 初始化数据库
    config = Config()
    db = SemanticCoreDB(config)
    
    try:
        # 1. 插入文本数据
        print("1. 插入文本数据...")
        text_data = {
            "type": "text",
            "data": "这是一篇关于人工智能的文章，介绍了机器学习的基本概念和应用场景。",
            "metadata": {
                "title": "人工智能简介",
                "author": "张三",
                "category": "技术",
                "tags": ["AI", "机器学习", "技术"]
            },
            "tags": ["人工智能", "机器学习", "技术文章"]
        }
        
        text_id = await db.insert(text_data)
        print(f"文本数据插入成功，对象 ID: {text_id}")
        
        # 2. 插入图像数据
        print("\n2. 插入图像数据...")
        image_data = {
            "type": "image",
            "data": "examples/data/sample_image.jpg",  # 假设有这个文件
            "metadata": {
                "location": "新加坡",
                "timestamp": "2024-01-01T10:00:00Z",
                "camera": "iPhone 15",
                "tags": ["风景", "城市", "建筑"]
            },
            "tags": ["新加坡", "城市风景", "建筑摄影"]
        }
        
        image_id = await db.insert(image_data)
        print(f"图像数据插入成功，对象 ID: {image_id}")
        
        # 3. 插入 IoT 数据
        print("\n3. 插入 IoT 数据...")
        iot_data = {
            "type": "iot",
            "data": {
                "temperature": 25.5,
                "humidity": 65.2,
                "pressure": 1013.25,
                "timestamp": "2024-01-01T10:00:00Z"
            },
            "metadata": {
                "sensor_id": "TEMP_001",
                "location": "办公室",
                "unit": "celsius"
            },
            "tags": ["温度传感器", "环境监测", "IoT"]
        }
        
        iot_id = await db.insert(iot_data)
        print(f"IoT 数据插入成功，对象 ID: {iot_id}")
        
        # 4. 执行自然语言查询
        print("\n4. 执行自然语言查询...")
        query = "找出所有关于人工智能的文章"
        results = await db.query(query)
        print(f"查询结果数量: {len(results)}")
        for i, result in enumerate(results[:3]):  # 只显示前3个结果
            print(f"  结果 {i+1}: {result.get('object_id', 'N/A')}")
        
        # 5. 执行语义查询
        print("\n5. 执行语义查询...")
        semantic_query = "找出在新加坡拍摄的照片"
        semantic_results = await db.query(semantic_query)
        print(f"语义查询结果数量: {len(semantic_results)}")
        for i, result in enumerate(semantic_results[:3]):
            print(f"  结果 {i+1}: {result.get('object_id', 'N/A')}")
        
        # 6. 获取单个对象
        print("\n6. 获取单个对象...")
        object_data = await db.get_object(text_id)
        if object_data:
            print(f"对象类型: {object_data.get('type')}")
            print(f"对象数据: {object_data.get('data', '')[:50]}...")
        
        # 7. 更新对象
        print("\n7. 更新对象...")
        update_data = {
            "type": "text",
            "data": "这是更新后的文章内容，包含了更多关于深度学习的详细信息。",
            "metadata": {
                "title": "人工智能与深度学习",
                "author": "张三",
                "category": "技术",
                "updated_at": datetime.utcnow().isoformat()
            },
            "tags": ["AI", "深度学习", "技术", "更新"]
        }
        
        success = await db.update(text_id, update_data)
        print(f"对象更新成功: {success}")
        
        # 8. 获取数据库统计信息
        print("\n8. 获取数据库统计信息...")
        stats = await db.get_statistics()
        print(f"数据库统计信息:")
        print(f"  - 总对象数: {stats.get('total_objects', 0)}")
        print(f"  - 总向量数: {stats.get('total_vectors', 0)}")
        print(f"  - 总事件数: {stats.get('total_events', 0)}")
        
        # 9. 删除对象
        print("\n9. 删除对象...")
        success = await db.delete(iot_id)
        print(f"对象删除成功: {success}")
        
    except Exception as e:
        print(f"示例执行失败: {e}")
    
    finally:
        # 关闭数据库
        await db.close()
        print("\n数据库连接已关闭")


async def advanced_example():
    """高级使用示例"""
    print("\n=== SemanticCoreDB 高级使用示例 ===\n")
    
    config = Config()
    db = SemanticCoreDB(config)
    
    try:
        # 批量插入数据
        print("1. 批量插入数据...")
        sample_data = [
            {
                "type": "text",
                "data": "机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习。",
                "metadata": {"category": "AI", "language": "zh"},
                "tags": ["机器学习", "AI", "技术"]
            },
            {
                "type": "text", 
                "data": "深度学习通过神经网络模拟人脑的学习过程，在图像识别和自然语言处理方面取得了突破性进展。",
                "metadata": {"category": "AI", "language": "zh"},
                "tags": ["深度学习", "神经网络", "AI"]
            },
            {
                "type": "text",
                "data": "计算机视觉技术使机器能够理解和分析图像内容，广泛应用于自动驾驶、医疗诊断等领域。",
                "metadata": {"category": "AI", "language": "zh"},
                "tags": ["计算机视觉", "图像处理", "AI"]
            }
        ]
        
        object_ids = []
        for data in sample_data:
            object_id = await db.insert(data)
            object_ids.append(object_id)
            print(f"插入对象: {object_id}")
        
        # 复杂查询
        print("\n2. 执行复杂查询...")
        complex_query = "找出所有关于深度学习和神经网络的文章，按相关性排序"
        results = await db.query(complex_query)
        print(f"复杂查询结果数量: {len(results)}")
        
        # 跨模态查询
        print("\n3. 执行跨模态查询...")
        multimodal_query = "找出所有包含技术内容的文本和图像"
        multimodal_results = await db.query(multimodal_query)
        print(f"跨模态查询结果数量: {len(multimodal_results)}")
        
    except Exception as e:
        print(f"高级示例执行失败: {e}")
    
    finally:
        await db.close()


async def performance_test():
    """性能测试示例"""
    print("\n=== SemanticCoreDB 性能测试 ===\n")
    
    config = Config()
    db = SemanticCoreDB(config)
    
    try:
        import time
        
        # 写入性能测试
        print("1. 写入性能测试...")
        start_time = time.time()
        
        for i in range(100):
            data = {
                "type": "text",
                "data": f"这是第 {i+1} 条测试数据，用于性能测试。",
                "metadata": {"test_id": i, "category": "performance_test"},
                "tags": [f"测试{i}", "性能测试"]
            }
            await db.insert(data)
        
        end_time = time.time()
        write_time = end_time - start_time
        print(f"写入 100 条数据耗时: {write_time:.2f} 秒")
        print(f"平均写入速度: {100/write_time:.2f} 条/秒")
        
        # 查询性能测试
        print("\n2. 查询性能测试...")
        start_time = time.time()
        
        for i in range(10):
            query = f"找出包含测试{i}的数据"
            results = await db.query(query)
        
        end_time = time.time()
        query_time = end_time - start_time
        print(f"执行 10 次查询耗时: {query_time:.2f} 秒")
        print(f"平均查询时间: {query_time/10:.2f} 秒")
        
    except Exception as e:
        print(f"性能测试失败: {e}")
    
    finally:
        await db.close()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(basic_example())
    asyncio.run(advanced_example())
    asyncio.run(performance_test())
    
    print("\n=== 所有示例执行完成 ===") 