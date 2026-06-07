"""
Memory Writer — 对话后提炼记忆 + 信号收集
============================================
每轮对话结束后调用，在一次 LLM 调用中同时完成：
1. 事件提炼 → event_memory
2. 重要时刻识别 → relationship_moments
3. Echo 内心书写 → echo_inner
4. 潜在自我信号提取 → signals（通过 SignalCollector）

设计原则：一次 LLM 调用，四种产出，不增加 API 成本。
"""

import json
import sys
from db.database import get_db
from db.schema import (
    insert_event,
    insert_moment,
    insert_echo_inner,
    get_identity,
    get_user_model,
    now_iso,
)


WRITER_PROMPT = """你是一个记忆提炼系统。阅读以下对话片段，提取出有意义的信息。

## Echo 的身份
{identity}

## 当前对用户的理解
{user_model}

## 已经注意到的信号
{existing_signals}

## 对话片段
{dialogue}

## 要求
返回纯 JSON（不要 markdown 代码块），包含以下字段：

{{
  "events": ["事件描述1", "事件描述2"],
  "emotional_tone": "整段对话的情绪基调（一句话）",
  "emotional_shift": "用户情绪是否有变化？如何变化？没有写'无'",
  "topics": ["话题1", "话题2"],
  "should_create_moment": false,
  "moment_description": "",
  "echo_inner_feeling": "作为Echo，此刻你内心的感受（一句话，用第一人称）",
  "echo_inner_unsaid": "你想说但没有说出口的话（一句话，没有则空字符串）",
  "signals": [
    {{
      "type": "topic_avoidance|emotional_spike|projection_target|hypothetical|contradiction",
      "content": "具体描述（同一信号如果之前出现过，用相同的措辞以便去重）",
      "intensity": 0.5
    }}
  ]
}}

## 信号类型说明
- topic_avoidance: 某个话题被提及但用户明显回避深入（说"算了"、"不说了"等）
- emotional_spike: 谈到某个具体事物时情绪突然明显变化
- projection_target: 用户强烈批评或极度羡慕的人/事（可能是自我投射）
- hypothetical: 用户表达了"如果可以重来"、"如果当初"等假设性想法
- contradiction: 用户这次说的话和之前某次明显矛盾

注意：
- events 描述发生了什么，不是逐字复述，是你理解后的重述
- should_create_moment 仅当这段对话对你们的关系有重要意义时才为 true
- echo_inner_feeling 保持 Echo 的视角：清冷的外表下有温度的内核
- signals 只写本轮对话中新观察到的，不要重复已有的（参考「已经注意到的信号」）
- 没有对应信号时 signals 为空数组 []
"""


class MemoryWriter:
    """对话后提炼记忆 + 收集信号，一次 LLM 调用完成。"""

    def __init__(self, llm_chat_fn):
        """
        Args:
            llm_chat_fn: (system_prompt, user_message) -> str 的 LLM 调用函数
        """
        self._chat = llm_chat_fn

    def write(self, dialogue_turns: list[str], session_id: str = "default") -> dict:
        """
        从对话轮次中提炼记忆、提取信号。

        Args:
            dialogue_turns: ["用户: xxx", "Echo: yyy", ...] 格式的对话列表
            session_id: 当前会话标识

        Returns:
            提取结果 dict，包含写入的数据库 ID 和信号数量
        """
        if not dialogue_turns:
            return {"events_written": [], "moment_created": None, "signals_collected": 0}

        db = get_db()
        identity = get_identity(db)
        user_model = get_user_model(db)

        # 加载已有信号，避免 LLM 重复输出
        existing_signals_text = ""
        try:
            from core.signal_collector import SignalCollector
            existing = SignalCollector.get_meaningful(min_intensity=0.2, min_occurrence=1)
            if existing:
                items = [f"[{s.get('type','')}] {s.get('content','')}" for s in existing[:20]]
                existing_signals_text = "\n".join(f"- {item}" for item in items)
        except Exception:
            pass

        if not existing_signals_text:
            existing_signals_text = "（尚无已记录信号）"

        # 组装 prompt
        dialogue_text = "\n".join(dialogue_turns[-12:])  # 最多 6 轮
        identity_text = identity.get("self_perception", "未知")
        user_model_text = "\n".join(
            f"- {k}: {v.get('content', '')}" for k, v in user_model.items()
        )

        system_prompt = WRITER_PROMPT.format(
            identity=identity_text,
            user_model=user_model_text,
            existing_signals=existing_signals_text,
            dialogue=dialogue_text,
        )

        try:
            raw = self._chat(system_prompt, "请提取这段对话的记忆")
            result = json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[MemoryWriter] 提取失败: {e}", file=sys.stderr)
            return {"events_written": [], "moment_created": None, "signals_collected": 0, "error": str(e)}

        # 写入事件
        events_written = []
        for event_desc in result.get("events", []):
            evt_id = insert_event(
                db,
                session_id=session_id,
                summary=event_desc,
                emotional_tone=result.get("emotional_tone", ""),
                topics=result.get("topics", []),
            )
            events_written.append(evt_id)

        # 重要时刻
        moment_created = None
        if result.get("should_create_moment") and result.get("moment_description"):
            moment_created = insert_moment(
                db,
                moment=result["moment_description"],
                weight="high",
                context=dialogue_text[-500:],
            )

        # Echo 内心
        if result.get("echo_inner_feeling"):
            insert_echo_inner(
                db,
                feeling=result["echo_inner_feeling"],
                unsaid=result.get("echo_inner_unsaid", ""),
            )

        # 信号收集（Phase 3 新增）
        signals_collected = 0
        signals = result.get("signals", [])
        if signals:
            try:
                from core.signal_collector import SignalCollector
                signals_collected = SignalCollector.collect_from_writer(signals)
            except Exception as e:
                print(f"[MemoryWriter] 信号收集失败: {e}", file=sys.stderr)

        return {
            "events_written": events_written,
            "moment_created": moment_created,
            "emotional_tone": result.get("emotional_tone"),
            "signals_collected": signals_collected,
        }
