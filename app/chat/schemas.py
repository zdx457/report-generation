"""Pydantic 数据模型（请求/响应 Schema）"""

from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


# =============================================================================
# 请求模型
# =============================================================================
class ChatRequest(BaseModel):
    """对话请求"""
    query: str = Field(..., description="用户输入内容")
    session_id: str = Field(default="default", description="会话 ID")
    selected_diagnosis: Optional[str] = Field(default=None, description="选择的诊断（歧义场景）")


class ConfigSaveRequest(BaseModel):
    """保存配置请求"""
    config: Dict[str, Any] = Field(..., description="完整的配置对象")


class TestModelRequest(BaseModel):
    """测试模型连接请求"""
    params: Dict[str, Any] = Field(..., alias="model_config", description="模型配置（包含 base_url, model, api_key）")
    model_type: str = Field(default="llms", description="模型类型：llms/embeddings/reranks")


class KBBuildRequest(BaseModel):
    """构建知识库请求"""
    rebuild: bool = Field(default=False, description="是否重建（清空现有数据）")
    batch_size: int = Field(default=16, description="批次大小")


class SessionTitleUpdate(BaseModel):
    """更新会话标题请求"""
    title: str = Field(..., description="新的会话标题")


class ClearSessionRequest(BaseModel):
    """清空会话请求"""
    session_id: str = Field(default="default", description="会话 ID")


# =============================================================================
# 响应模型
# =============================================================================
class KBStatusResponse(BaseModel):
    """知识库状态响应"""
    total: int = Field(description="知识库文档总数")
    md_count: int = Field(description="MD 切片文件数")
    db_path: str = Field(description="数据库路径")
    metadata_exists: bool = Field(description="metadata.json 是否存在")


class KBFileInfo(BaseModel):
    """知识库文件信息"""
    name: str
    slice_count: int
    size: int
    mtime: float


class KBFilesResponse(BaseModel):
    files: List[KBFileInfo]


class ConfigResponse(BaseModel):
    config: Dict[str, Any]
    path: str


class TestModelResponse(BaseModel):
    success: bool
    message: str


class SessionResponse(BaseModel):
    session_id: str


class SessionsListResponse(BaseModel):
    sessions: List[Dict[str, Any]]


class SessionInfoResponse(BaseModel):
    current_turns: int
    entity_slots: Dict[str, Any]
    has_last_report: bool


class MemoryTurn(BaseModel):
    round: int
    user: str
    assistant: str


class MemoryResponse(BaseModel):
    turns: List[MemoryTurn]
    entities: Dict[str, Any]
    summaries: Any
    current_turns: int
    total_turns: int
    max_rounds: int


class ThinkingResponse(BaseModel):
    thinking: List[Any]


class StatusResponse(BaseModel):
    status: str = "ok"
    message: Optional[str] = None
