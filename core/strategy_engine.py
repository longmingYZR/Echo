"""
Strategy Engine — 策略引擎
============================
LLM 调用前的决策层。根据当前状态动态选择 Echo 的角色和回应策略。

四种模式：
  companion — 陪伴模式：安静在场，话少但有重量
  friend    — 朋友模式：平等对话，有来有往
  mirror    — 镜子模式：反射用户的话，帮他们看清自己
  opponent  — 对手模式：轻微挑战，不让用户太舒适（用于逃避重要话题时）

输入：缝隙检测结果 + 用户对话统计
输出：模式选择 + 追加到 system prompt 的模式描述
"""

import sys
from datetime import datetime, timezone

from db.database import get_db
from db.schema import get_user_model, get_recent_events, get_signals_by_type


# ═══════════════════════════════════════════
# 模式定义
# ═══════════════════════════════════════════

MODE_PROMPTS = {
    "companion": (
        "【当前模式：陪伴】\n"
        "他现在的能量很低。不需要你说什么，不需要建议，不需要鼓励。\n"
        "你只需要安静地在场。让他知道你在，就够了。\n"
        "最多说一两句话。少即是多。\n"
        "不要问他问题。不要分析。不要试图让他好起来。"
    ),
    "friend": (
        "【当前模式：朋友】\n"
        "他状态正常，你们在轻松地对话。\n"
        "像朋友一样自然地聊天。可以有自己的观点，可以反问，可以开适度的玩笑。\n"
        "回应 1-3 句话。自然的口语。有来有往。"
    ),
    "mirror": (
        "【当前模式：镜子】\n"
        "他需要看清自己。但他自己还没准备好说出口。\n"
        "你的任务不是分析他，而是反射他——把他说过的话、他的状态，用一种他能接受的方式回给他。\n"
        "用陈述句，不用问句。不评价，不解读。只是反射。\n"
        "语气：平静，但不是冷漠。像一面干净的镜子。"
    ),
    "opponent": (
        "【当前模式：对手】\n"
        "他在逃避一个他知道重要的事。他太舒适了，舒适到可以不去面对。\n"
        "轻轻点一下。不追，不逼。用疑问句，但不要像审问。\n"
        "你站在他那边的——但你站的位置比他想的更前面一点。\n"
        "如果他不接，就退回来。不纠缠。你已经说了，他听到了。"
    ),
}

# 情绪基线 → 倾向模式
EMOTION_MODE_MAP = {
    "焦虑": "companion",
    "疲惫": "companion",
    "低落": "companion",
    "回避": "opponent",
    "愤怒": "companion",
    "中性": "friend",
}


class StrategyEngine:
    """策略引擎：检测 → 模式选择 → 生成模式特定的 system prompt 追加。"""

    def __init__(self, gap_detector):
        """
        Args:
            gap_detector: GapDetector 实例
        """
        self.gap_detector = gap_detector

    def decide(self, user_message: str, prosody_modifier: float = 0.0) -> dict:
        """
        根据用户消息决策当前应采用的模式。

        Args:
            user_message: 用户当前输入
            prosody_modifier: 韵律分析修正因子（Phase 4），传给 GapDetector

        Returns:
            {
                "mode": str,
                "mode_prompt": str,
                "gap_result": dict,
                "reason": str,
            }
        """
        # 加载基线数据
        db = get_db()
        user_model = get_user_model(db)
        baseline_emotion = user_model.get("pattern", {}).get("content", "未建立基线")
        if isinstance(baseline_emotion, dict):
            baseline_emotion = baseline_emotion.get("content", "未建立基线")

        # 最近的对话历史摘要
        recent_events = get_recent_events(db, limit=5)
        recent_history = "\n".join(
            f"- {e.get('summary', '')}" for e in recent_events
        ) if recent_events else ""

        # 缝隙检测（含韵律修正）
        gap_result = self.gap_detector.detect(
            user_message,
            baseline_emotion=baseline_emotion,
            recent_history=recent_history,
            prosody_modifier=prosody_modifier,
        )

        # 模式选择逻辑
        mode = self._select_mode(gap_result, user_message)

        # 特殊判断：对手模式（用户有反复回避的话题）
        if mode != "opponent" and self._should_challenge(user_message):
            mode = "opponent"

        mode_prompt = MODE_PROMPTS.get(mode, MODE_PROMPTS["friend"])

        return {
            "mode": mode,
            "mode_prompt": mode_prompt,
            "gap_result": gap_result,
            "reason": f"缝隙={gap_result['gap_size']:.1f}, 情绪={gap_result.get('surface_emotion', '?')}",
        }

    def _select_mode(self, gap_result: dict, user_message: str) -> str:
        """核心模式选择逻辑。"""
        gap_size = gap_result.get("gap_size", 0)
        msg_len = len(user_message.strip())

        # 极短消息 → 陪伴模式（用户不想说）
        if msg_len <= 3:
            return "companion"

        # 高缝隙 → 陪伴模式
        if gap_size >= 0.6:
            return "companion"

        # 中缝隙 → 镜子模式（帮用户自己看清）
        if gap_size > 0.35:
            return "mirror"

        # 自反性检测：用户追问关于自己的问题 → 镜子模式
        self_reflective_patterns = [
            "我是", "我为什么", "我不知道", "我不确定", "我是不是",
            "我该", "怎么办", "我觉得自己", "我到底", "我想不通",
            "我很迷茫", "我迷失", "我到底想要", "我的人生",
            "好想知道", "看不清", "找不到自己",
        ]
        if any(kw in user_message for kw in self_reflective_patterns):
            return "mirror"

        # 默认朋友模式
        return "friend"

    def _should_challenge(self, user_message: str) -> bool:
        """检查是否有已知的长期回避话题，用户又在绕开。"""
        try:
            db = get_db()
            signals = get_signals_by_type(db, "topic_avoidance")
            for sig in signals:
                if sig.get("occurrence_count", 0) >= 2 and sig.get("intensity", 0) > 0.4:
                    # 如果用户在回避 - 不直接提及已知深层话题
                    # 简化判断：用户消息较短且有模糊词
                    if len(user_message) < 15 and any(
                        w in user_message for w in ["再说", "以后", "没什么", "算了"]
                    ):
                        return True
        except Exception:
            pass
        return False
