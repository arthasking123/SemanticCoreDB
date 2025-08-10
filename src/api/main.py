"""
SemanticCoreDB API 主应用
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging

from ..core.database import SemanticCoreDB
from ..core.config import Config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="SemanticCoreDB API",
    description="基于 LLM 的语义驱动数据库 API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局数据库实例
db: Optional[SemanticCoreDB] = None


# Pydantic 模型
class InsertRequest(BaseModel):
    type: str
    data: Any
    metadata: Optional[Dict[str, Any]] = {}
    tags: Optional[List[str]] = []


class UpdateRequest(BaseModel):
    type: str
    data: Any
    metadata: Optional[Dict[str, Any]] = {}
    tags: Optional[List[str]] = []


class QueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 100


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int
    query_time: float


# 依赖注入
async def get_database() -> SemanticCoreDB:
    """获取数据库实例"""
    global db
    if db is None:
        config = Config()
        db = SemanticCoreDB(config)
    return db


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("SemanticCoreDB API 启动中...")
    global db
    config = Config.from_file("config/default.yaml")
    db = SemanticCoreDB(config)
    logger.info("SemanticCoreDB API 启动完成")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("SemanticCoreDB API 关闭中...")
    global db
    if db:
        await db.close()
    logger.info("SemanticCoreDB API 已关闭")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "SemanticCoreDB API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        global db
        if db:
            stats = await db.get_statistics()
            return {
                "status": "healthy",
                "database": "connected",
                "statistics": stats
            }
        else:
            return {
                "status": "unhealthy",
                "database": "disconnected"
            }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/insert", response_model=Dict[str, str])
async def insert_data(
    request: InsertRequest,
    database: SemanticCoreDB = Depends(get_database)
):
    """插入数据"""
    try:
        data = {
            "type": request.type,
            "data": request.data,
            "metadata": request.metadata,
            "tags": request.tags
        }
        
        object_id = await database.insert(data)
        
        return {
            "object_id": object_id,
            "status": "success",
            "message": "数据插入成功"
        }
        
    except Exception as e:
        logger.error(f"插入数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"插入数据失败: {str(e)}")


@app.get("/object/{object_id}")
async def get_object(
    object_id: str,
    database: SemanticCoreDB = Depends(get_database)
):
    """获取单个对象"""
    try:
        object_data = await database.get_object(object_id)
        
        if object_data is None:
            raise HTTPException(status_code=404, detail="对象不存在")
        
        return {
            "object_id": object_id,
            "data": object_data,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取对象失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取对象失败: {str(e)}")


@app.put("/object/{object_id}")
async def update_object(
    object_id: str,
    request: UpdateRequest,
    database: SemanticCoreDB = Depends(get_database)
):
    """更新对象"""
    try:
        data = {
            "type": request.type,
            "data": request.data,
            "metadata": request.metadata,
            "tags": request.tags
        }
        
        success = await database.update(object_id, data)
        
        if not success:
            raise HTTPException(status_code=404, detail="对象不存在或更新失败")
        
        return {
            "object_id": object_id,
            "status": "success",
            "message": "对象更新成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新对象失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新对象失败: {str(e)}")


@app.delete("/object/{object_id}")
async def delete_object(
    object_id: str,
    database: SemanticCoreDB = Depends(get_database)
):
    """删除对象"""
    try:
        success = await database.delete(object_id)
        
        return {
            "object_id": object_id,
            "status": "success",
            "message": "对象删除成功"
        }
        
    except Exception as e:
        logger.error(f"删除对象失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除对象失败: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_data(
    request: QueryRequest,
    database: SemanticCoreDB = Depends(get_database)
):
    """查询数据"""
    try:
        import time
        start_time = time.time()
        
        results = await database.query(request.query)
        
        query_time = time.time() - start_time
        
        return QueryResponse(
            results=results,
            total=len(results),
            query_time=query_time
        )
        
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/statistics")
async def get_statistics(
    database: SemanticCoreDB = Depends(get_database)
):
    """获取数据库统计信息"""
    try:
        stats = await database.get_statistics()
        
        return {
            "statistics": stats,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@app.post("/backup")
async def backup_database(
    backup_path: str,
    database: SemanticCoreDB = Depends(get_database)
):
    """备份数据库"""
    try:
        success = await database.backup(backup_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="备份失败")
        
        return {
            "backup_path": backup_path,
            "status": "success",
            "message": "数据库备份成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"备份数据库失败: {e}")
        raise HTTPException(status_code=500, detail=f"备份数据库失败: {str(e)}")


@app.post("/restore")
async def restore_database(
    backup_path: str,
    database: SemanticCoreDB = Depends(get_database)
):
    """恢复数据库"""
    try:
        success = await database.restore(backup_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="恢复失败")
        
        return {
            "backup_path": backup_path,
            "status": "success",
            "message": "数据库恢复成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复数据库失败: {e}")
        raise HTTPException(status_code=500, detail=f"恢复数据库失败: {str(e)}")


# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )