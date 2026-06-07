"""
Echo API Server
=================
FastAPI 服务，将 Echo 全链路暴露为 REST API。

启动: uvicorn server:app --host 0.0.0.0 --port 8848

端点:
  POST /api/chat          — 发送消息，获取 Echo 回复
  GET  /api/status        — Echo 当前状态（记忆统计、时间感、信号）
  GET  /api/memory/events — 最近事件记忆列表
  POST /api/session/reset — 开始新会话
"""

import sys
import os
import uuid
import json
from datetime import datetime, timezone

# 确保项目根在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from voice_loop import (
    CONFIG,
    LLMClient,
    EchoVoiceLoop,
)

app = FastAPI(
    title="Echo API",
    description="AI 语音伴侣 Echo 的 HTTP API",
    version="1.0.0",
)

# CORS（允许 Web UI 跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════
# 全局状态
# ═══════════════════════════════════════════

_llm = None
_api_key_checked = False


def get_llm():
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def check_api_key():
    global _api_key_checked
    if not _api_key_checked:
        key = CONFIG.get("llm_api_key", "")
        if not key:
            print("[Echo Server] ⚠ DEEPSEEK_API_KEY 未设置，LLM 调用将失败")
        else:
            print(f"[Echo Server] ✓ API key 已配置 (模型: {CONFIG['llm_model']})")
        _api_key_checked = True


# ═══════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    mode: str | None = None
    gap_detected: bool = False
    gap_size: float = 0.0
    session_id: str


class StatusResponse(BaseModel):
    time_since_last: str
    relationship_stage: str
    event_count: int
    moment_count: int
    signal_count: int
    model_layers: dict


class MemoryEvent(BaseModel):
    id: str
    summary: str
    emotional_tone: str
    topics: list[str]
    created_at: str


# ═══════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════

@app.on_event("startup")
async def startup():
    check_api_key()
    # 预热数据库
    from db.database import get_db
    get_db()
    print("[Echo Server] 启动完成")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """发送消息给 Echo，获取回复。"""
    llm = get_llm()
    session_id = req.session_id or uuid.uuid4().hex[:8]

    # 构建 system prompt（含记忆上下文 + 策略引擎）
    try:
        from core.memory_awakener import MemoryAwakener
        from core.strategy_engine import StrategyEngine
        from core.gap_detector import GapDetector

        ctx = MemoryAwakener.build_context()
        base_prompt = EchoVoiceLoop.ECHO_BASE_PROMPT.format(
            memory_context=ctx.get("full_context", "")
        )

        # 策略引擎
        gd = GapDetector(llm_chat_fn=llm.chat_stateless)
        se = StrategyEngine(gd)
        strategy = se.decide(req.message)
        mode_block = f"\n\n{strategy['mode_prompt']}"
        system_prompt = base_prompt + mode_block

        gap = strategy["gap_result"]
        mode = strategy["mode"]
        gap_detected = gap.get("gap_detected", False)
        gap_size = gap.get("gap_size", 0.0)
    except Exception as e:
        print(f"[Echo Server] 策略引擎加载失败: {e}", file=sys.stderr)
        system_prompt = EchoVoiceLoop.DEFAULT_SYSTEM_PROMPT
        mode = None
        gap_detected = False
        gap_size = 0.0

    # LLM
    try:
        reply = llm.chat(req.message, system_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {e}")

    # 后台写记忆
    try:
        import threading
        from core.memory_writer import MemoryWriter
        writer = MemoryWriter(llm_chat_fn=llm.chat_stateless)
        dialogue = [f"用户: {req.message}", f"Echo: {reply}"]
        threading.Thread(
            target=writer.write,
            args=(dialogue, session_id),
            daemon=True,
        ).start()
    except Exception:
        pass

    return ChatResponse(
        reply=reply,
        mode=mode,
        gap_detected=gap_detected,
        gap_size=gap_size,
        session_id=session_id,
    )


@app.get("/api/status", response_model=StatusResponse)
async def status():
    """Echo 当前状态概览。"""
    from db.database import get_db
    from db.schema import get_identity, get_user_model
    from core.memory_awakener import MemoryAwakener

    db = get_db()
    identity = get_identity(db)
    user_model = get_user_model(db)

    event_count = db.execute("SELECT COUNT(*) FROM event_memory").fetchone()[0]
    moment_count = db.execute("SELECT COUNT(*) FROM relationship_moments").fetchone()[0]
    signal_count = db.execute("SELECT COUNT(*) FROM signals").fetchone()[0]

    model_layers = {}
    for k, v in user_model.items():
        content = v.get("content", "") if isinstance(v, dict) else ""
        if content and content not in (
            "尚未建立足够对话数据", "尚无足够数据追踪变化",
            "尚未发现深层话题", "刚开始了解",
        ):
            model_layers[k] = content

    return StatusResponse(
        time_since_last=MemoryAwakener.time_since_last(),
        relationship_stage=identity.get("relationship_stage", ""),
        event_count=event_count,
        moment_count=moment_count,
        signal_count=signal_count,
        model_layers=model_layers,
    )


@app.get("/api/memory/events")
async def memory_events(limit: int = 20):
    """最近的事件记忆列表。"""
    from db.database import get_db
    from db.schema import get_recent_events

    events = get_recent_events(get_db(), limit=limit)
    return [
        {
            "id": e["id"],
            "summary": e["summary"],
            "emotional_tone": e["emotional_tone"],
            "topics": json.loads(e.get("topics", "[]")),
            "created_at": e["created_at"],
        }
        for e in events
    ]


@app.post("/api/session/reset")
async def reset_session():
    """重置 LLM 对话历史（不影响长期记忆）。"""
    llm = get_llm()
    llm.reset_memory()
    return {"message": "会话已重置", "session_id": uuid.uuid4().hex[:8]}


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": CONFIG["llm_model"]}


# ═══════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8848)
