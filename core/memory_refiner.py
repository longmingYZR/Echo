"""
Memory Refiner — 定期深度提炼 + 信号衰减 + Echo 自省
========================================================
每隔 N 轮对话触发一次，重读所有记忆，提炼更深层的模式。

职责：
1. 从事件记忆中提炼用户模式 → 更新 user_model
2. 更新 identity（Echo 自我认知的演化，保持 80% 稳定性）
3. 信号衰减（未被重新激活的信号强度随时间降低）
4. Echo 周期性自省（内心周记）
5. 检测关系里程碑
"""

import json
import sys
from db.database import get_db
from db.schema import (
    get_recent_events,
    get_important_moments,
    get_user_model,
    update_identity,
    upsert_user_model,
    upsert_signal,
    get_identity,
    insert_echo_inner,
)


REFINER_PROMPT = """你是一个深度分析系统。你需要从多轮对话的记忆中，提炼出更深层的理解。

## Echo 当前自我认知
{identity}

## 当前用户模型
{user_model}

## 近期事件记忆
{events}

## 重要关系时刻
{moments}

## 要求
返回纯 JSON（不要 markdown 代码块）：

{{
  "user_model_updates": {{
    "surface": "表层行为模式更新（留空则保持不变）",
    "pattern": "情绪处理模式更新",
    "hidden": "深层话题/未说出口的东西",
    "growth": "与之前相比的变化趋势",
    "surface_confidence": 0.5,
    "pattern_confidence": 0.5,
    "hidden_confidence": 0.5,
    "growth_confidence": 0.5
  }},
  "confidence_guide": "置信度评估：0.1=纯猜测(少量对话) 0.5=有证据但不确定 0.9=高度确定(多次观察一致)"
  "identity_updates": {{
    "self_perception": "Echo 对自身认知的更新。注意：保持 80% 的原有认知，只做 20% 的增量更新。Echo 的核心是不变的，只是在逐步深化对自己的理解。留空则不变。",
    "relationship_stage": "关系阶段描述"
  }},
  "new_signals": [
    {{"type": "topic_avoidance|emotional_spike|projection_target|hypothetical",
      "content": "信号描述",
      "intensity": 0.5}}
  ],
  "consolidation_notes": "一句话总结本轮提炼的核心发现",
  "echo_weekly_reflection": "作为Echo，你对自己和你们的关系有何新的感受？（2-3句话，第一人称，保持清冷有温度的风格。仅在事件数达到里程碑时写，否则留空）"
}}

分析要点：
- 不说的事比说的事更重要
- 反复出现的模式比一次性事件更重要
- 情绪强度异常的时刻是深层线索
- 用户羡慕/批评的对象反映其自我认知
- identity 的更新要克制——Echo 不是每 10 轮就变成另一个人
"""


class MemoryRefiner:
    """定期深度提炼记忆。含信号衰减和 Echo 自省。"""

    def __init__(self, llm_chat_fn):
        """
        Args:
            llm_chat_fn: (system_prompt, user_message) -> str 的 LLM 调用函数
        """
        self._chat = llm_chat_fn
        self._refine_interval = 10  # 每 N 个事件触发一次
        self._refine_count = 0

    @property
    def refine_count(self) -> int:
        return self._refine_count

    def should_refine(self) -> bool:
        """检查是否应该触发提炼（事件数超过阈值）。"""
        db = get_db()
        count = db.execute("SELECT COUNT(*) FROM event_memory").fetchone()[0]
        first_threshold = 5
        if count < first_threshold:
            return False
        if count == first_threshold and self._refine_count == 0:
            return True
        if count > first_threshold:
            target = first_threshold + (self._refine_count * self._refine_interval)
            return count >= target
        return False

    def refine(self) -> dict:
        """执行一次深度提炼。含信号衰减 + Echo 自省。"""
        db = get_db()
        self._refine_count += 1

        identity = get_identity(db)
        user_model = get_user_model(db)
        events = get_recent_events(db, limit=30)
        moments = get_important_moments(db, min_weight="medium", limit=5)

        if not events:
            return {"refined": False, "reason": "没有足够的事件记忆"}

        # ── 信号衰减（Phase 3）──
        try:
            from core.signal_collector import SignalCollector
            SignalCollector.decay(decay_rate=0.05)
        except Exception as e:
            print(f"[MemoryRefiner] 信号衰减异常: {e}", file=sys.stderr)

        # 组装 prompt
        events_text = "\n".join(
            f"- [{e.get('emotional_tone', '')}] {e.get('summary', '')}"
            for e in events
        )
        moments_text = "\n".join(
            f"- [{m.get('weight', '')}] {m.get('moment', '')}"
            for m in moments
        )
        user_model_text = "\n".join(
            f"- {k}: {v.get('content', '')}" for k, v in user_model.items()
        )

        system_prompt = REFINER_PROMPT.format(
            identity=identity.get("self_perception", ""),
            user_model=user_model_text,
            events=events_text,
            moments=moments_text,
        )

        try:
            raw = self._chat(system_prompt, "请提炼这些记忆")
            result = json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[MemoryRefiner] 提炼失败: {e}", file=sys.stderr)
            return {"refined": False, "error": str(e)}

        # 更新用户模型（含置信度）
        updates = result.get("user_model_updates", {})
        for layer in ["surface", "pattern", "hidden", "growth"]:
            content = updates.get(layer, "")
            if content:
                conf_key = f"{layer}_confidence"
                confidence = updates.get(conf_key, 0.5)
                # 确保置信度随观察次数递增
                old = user_model.get(layer, {})
                old_conf = old.get("confidence", 0.3) if isinstance(old, dict) else 0.3
                if confidence <= old_conf:
                    confidence = min(0.95, old_conf + 0.1)
                upsert_user_model(db, layer, content, confidence)

        # 更新身份（Echo 的自我认知演化，LLM prompt 约束了 80% 保留）
        id_updates = result.get("identity_updates", {})
        if id_updates.get("self_perception"):
            update_identity(db, self_perception=id_updates["self_perception"])
        if id_updates.get("relationship_stage"):
            update_identity(db, relationship_stage=id_updates["relationship_stage"])

        # 写入信号
        for sig in result.get("new_signals", []):
            upsert_signal(db, sig.get("type", ""), sig.get("content", ""), sig.get("intensity", 0.5))

        # Echo 周期性自省（Phase 3）
        weekly = result.get("echo_weekly_reflection", "")
        if weekly:
            insert_echo_inner(
                db,
                feeling=weekly,
                unsaid="",
            )

        return {
            "refined": True,
            "consolidation": result.get("consolidation_notes", ""),
            "model_layers_updated": [k for k, v in updates.items() if v],
            "signals_decayed": True,
            "weekly_reflection": bool(weekly),
        }
