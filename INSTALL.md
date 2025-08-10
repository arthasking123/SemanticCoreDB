# SemanticCoreDB 安装和使用指南

## 快速开始

### 1. 环境要求

- Python 3.9+
- Node.js 16+ (可选，用于前端)
- Docker (可选，用于容器化部署)
- 至少 4GB RAM
- 至少 10GB 可用磁盘空间

### 2. 安装步骤

#### 方法一：本地安装

```bash
# 1. 克隆项目
git clone https://github.com/semanticcoredb/semantic-core-db.git
cd semantic-core-db

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 创建必要目录
mkdir -p data/events data/objects data/vectors data/metadata logs

# 5. 设置环境变量
export OPENAI_API_KEY="your-openai-api-key"  # Linux/Mac
# 或
set OPENAI_API_KEY=your-openai-api-key  # Windows
```

#### 方法二：Docker 安装

```bash
# 1. 克隆项目
git clone https://github.com/semanticcoredb/semantic-core-db.git
cd semantic-core-db

# 2. 设置环境变量
export OPENAI_API_KEY="your-openai-api-key"

# 3. 启动服务
docker-compose up -d
```

### 3. 验证安装

```bash
# 运行演示程序
python run_demo.py

# 或运行测试
python -m pytest tests/ -v
```

## 基本使用

### 1. 初始化数据库

```python
from src.core.database import SemanticCoreDB
from src.core.config import Config

# 创建配置
config = Config()

# 初始化数据库
db = SemanticCoreDB(config)
```

### 2. 插入数据

```python
# 插入文本数据
text_data = {
    "type": "text",
    "data": "这是一篇关于人工智能的文章",
    "metadata": {
        "title": "AI 简介",
        "author": "张三"
    },
    "tags": ["人工智能", "技术"]
}

object_id = await db.insert(text_data)
print(f"插入成功，对象 ID: {object_id}")

# 插入图像数据
image_data = {
    "type": "image",
    "data": "path/to/image.jpg",
    "metadata": {
        "location": "新加坡",
        "camera": "iPhone 15"
    },
    "tags": ["新加坡", "风景"]
}

image_id = await db.insert(image_data)

# 插入 IoT 数据
iot_data = {
    "type": "iot",
    "data": {
        "temperature": 25.5,
        "humidity": 65.2,
        "timestamp": "2024-01-01T10:00:00Z"
    },
    "metadata": {
        "sensor_id": "TEMP_001",
        "location": "办公室"
    },
    "tags": ["温度传感器", "IoT"]
}

iot_id = await db.insert(iot_data)
```

### 3. 执行查询

```python
# 自然语言查询
results = await db.query("找出所有关于人工智能的文章")
print(f"找到 {len(results)} 个结果")

# 语义查询
results = await db.query("找出在新加坡拍摄的照片")
print(f"找到 {len(results)} 个结果")

# 跨模态查询
results = await db.query("找出所有包含技术内容的文本和图像")
print(f"找到 {len(results)} 个结果")
```

### 4. 对象操作

```python
# 获取单个对象
object_data = await db.get_object(object_id)
if object_data:
    print(f"对象类型: {object_data['type']}")
    print(f"对象数据: {object_data['data']}")

# 更新对象
update_data = {
    "type": "text",
    "data": "更新后的文章内容",
    "metadata": {"updated": True},
    "tags": ["更新"]
}

success = await db.update(object_id, update_data)
print(f"更新成功: {success}")

# 删除对象
success = await db.delete(object_id)
print(f"删除成功: {success}")
```

### 5. 获取统计信息

```python
stats = await db.get_statistics()
print(f"总对象数: {stats['total_objects']}")
print(f"总向量数: {stats['total_vectors']}")
print(f"总事件数: {stats['total_events']}")
```

## API 使用

### 1. 启动 API 服务

```bash
# 本地启动
python -m src.api.main

# 或使用 uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. API 调用示例

```python
import requests

# 插入数据
response = requests.post("http://localhost:8000/insert", json={
    "type": "text",
    "data": "测试文本",
    "metadata": {"test": True},
    "tags": ["测试"]
})

object_id = response.json()["object_id"]

# 查询数据
response = requests.post("http://localhost:8000/query", json={
    "query": "找出所有测试数据",
    "limit": 10
})

results = response.json()["results"]
print(f"找到 {len(results)} 个结果")

# 获取对象
response = requests.get(f"http://localhost:8000/object/{object_id}")
object_data = response.json()["data"]

# 更新对象
response = requests.put(f"http://localhost:8000/object/{object_id}", json={
    "type": "text",
    "data": "更新后的文本",
    "metadata": {"updated": True},
    "tags": ["更新"]
})

# 删除对象
response = requests.delete(f"http://localhost:8000/object/{object_id}")

# 获取统计信息
response = requests.get("http://localhost:8000/statistics")
stats = response.json()["statistics"]
```

## 配置说明

### 1. 配置文件

项目支持多种配置方式：

- `config/default.yaml`: 默认配置
- `config/development.yaml`: 开发环境配置
- `config/production.yaml`: 生产环境配置

### 2. 环境变量

```bash
# 数据库配置
export SCDB_HOST=localhost
export SCDB_PORT=8000
export SCDB_DEBUG=true

# LLM 配置
export SCDB_LLM_PROVIDER=openai
export SCDB_LLM_MODEL=gpt-3.5-turbo
export SCDB_LLM_API_KEY=your-api-key

# 存储配置
export SCDB_DATA_DIR=./data
export SCDB_CACHE_SIZE=1000
```

### 3. 自定义配置

```python
from src.core.config import Config, StorageConfig, SemanticConfig

# 创建自定义配置
config = Config()

# 修改存储配置
config.storage.data_dir = "/custom/data/path"
config.storage.cache_size = 2000

# 修改语义配置
config.semantic.llm_provider = "anthropic"
config.semantic.embedding_model = "sentence-transformers/all-mpnet-base-v2"

# 使用自定义配置初始化数据库
db = SemanticCoreDB(config)
```

## 高级功能

### 1. 批量操作

```python
# 批量插入
batch_data = []
for i in range(100):
    data = {
        "type": "text",
        "data": f"批量数据 {i}",
        "metadata": {"batch_id": i},
        "tags": ["批量"]
    }
    batch_data.append(data)

object_ids = []
for data in batch_data:
    object_id = await db.insert(data)
    object_ids.append(object_id)

print(f"批量插入了 {len(object_ids)} 条数据")
```

### 2. 复杂查询

```python
# 多条件查询
results = await db.query("找出所有关于人工智能和机器学习的文章，按相关性排序")

# 时间范围查询
results = await db.query("找出上个月在新加坡拍摄的所有照片")

# 跨模态查询
results = await db.query("找出所有包含技术内容的文本、图像和视频")
```

### 3. 性能优化

```python
# 设置查询限制
config.query.default_limit = 50
config.query.max_limit = 500

# 设置缓存
config.storage.cache_size = 2000
config.storage.cache_ttl = 7200  # 2小时

# 设置向量索引参数
config.semantic.vector_index_params = {
    "nlist": 100,
    "nprobe": 10
}
```

## 故障排除

### 1. 常见问题

**问题**: 导入模块失败
```bash
# 解决方案：确保在项目根目录运行
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**问题**: LLM API 调用失败
```bash
# 解决方案：检查 API Key 设置
echo $OPENAI_API_KEY
```

**问题**: 存储空间不足
```bash
# 解决方案：清理数据或增加存储空间
rm -rf data/events/*.log
```

### 2. 日志查看

```bash
# 查看应用日志
tail -f logs/application.log

# 查看错误日志
tail -f logs/error.log

# 查看访问日志
tail -f logs/access.log
```

### 3. 性能监控

```bash
# 查看数据库统计
curl http://localhost:8000/statistics

# 健康检查
curl http://localhost:8000/health
```

## 开发指南

### 1. 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_database.py -v

# 运行性能测试
python -m pytest tests/test_performance.py -v
```

### 2. 代码格式化

```bash
# 格式化代码
black src/ tests/ examples/

# 检查代码质量
flake8 src/ tests/ examples/

# 类型检查
mypy src/
```

### 3. 贡献代码

1. Fork 项目
2. 创建功能分支
3. 编写测试
4. 提交代码
5. 创建 Pull Request

## 部署指南

### 1. 生产环境部署

```bash
# 使用 Docker 部署
docker-compose -f docker-compose.prod.yml up -d

# 或使用 Kubernetes
kubectl apply -f k8s/
```

### 2. 监控和日志

```bash
# 启动监控服务
docker-compose -f docker-compose.monitoring.yml up -d

# 访问 Grafana
open http://localhost:3000
```

### 3. 备份和恢复

```bash
# 备份数据库
curl -X POST "http://localhost:8000/backup" \
  -H "Content-Type: application/json" \
  -d '{"backup_path": "/backup/semanticcoredb_20240101.tar.gz"}'

# 恢复数据库
curl -X POST "http://localhost:8000/restore" \
  -H "Content-Type: application/json" \
  -d '{"backup_path": "/backup/semanticcoredb_20240101.tar.gz"}'
```

## 更多资源

- [技术白皮书](docs/whitepaper/semantic_core_db_whitepaper.md)
- [API 文档](http://localhost:8000/docs)
- [示例代码](examples/)
- [测试代码](tests/)
- [项目结构](PROJECT_STRUCTURE.md)

## 支持

- 问题反馈：https://github.com/semanticcoredb/semantic-core-db/issues
- 文档：https://docs.semanticcoredb.com
- 邮件：team@semanticcoredb.com