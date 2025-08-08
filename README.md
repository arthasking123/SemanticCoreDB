# SemanticCoreDB - 基于 LLM 的语义驱动数据库

## 项目概述

SemanticCoreDB 是一个全新的数据库系统，摒弃传统存储引擎和文件格式，采用事件化对象存储 + 向量索引 + 元数据图结构的组合架构，原生支持语义索引与推理。

## 核心特性

- **语义优先**：查询基于意义而非纯字段匹配
- **事件溯源**：所有数据变更记录为可重放事件流
- **多模态统一存储**：无需外部文件系统管理非结构化数据
- **可扩展推理**：内置 AI 模型推理接口，支持推理触发器
- **灵活 API**：SQL++ + 自然语言混合查询

## 项目结构

```
llm-database/
├── docs/                    # 技术文档
│   ├── whitepaper/         # 技术白皮书
│   ├── api/                # API 文档
│   └── architecture/       # 架构设计文档
├── src/                    # 源代码
│   ├── core/              # 核心引擎
│   ├── storage/           # 存储层
│   ├── query/             # 查询层
│   ├── semantic/          # 语义处理
│   └── api/               # API 接口
├── tests/                 # 测试代码
├── examples/              # 示例代码
├── scripts/               # 构建和部署脚本
└── config/               # 配置文件
```

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 16+
- Docker

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd llm-database

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Node.js 依赖
npm install

# 启动开发环境
docker-compose up -d
```

### 运行示例

```python
from semantic_core import SemanticCoreDB

# 初始化数据库
db = SemanticCoreDB()

# 插入多模态数据
db.insert({
    "type": "image",
    "data": "path/to/image.jpg",
    "metadata": {"location": "Singapore", "timestamp": "2024-01-01"}
})

# 语义查询
results = db.query("找出上个月在新加坡拍摄的所有包含黄色汽车的照片")
```

## 开发计划

- **M0 (PoC)**：核心引擎可运行，支持文本+图片语义查询（9周）
- **M1 (Alpha)**：支持视频/音频存储与检索（+12周）
- **M2 (Beta)**：分布式部署，性能优化（+16周）
- **M3 (GA)**：商业化版本（+24周）

## 贡献指南

请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目开发。

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。 