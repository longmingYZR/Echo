"""
Signal Collector — 潜在自我信号管理
======================================
管理五类信号的收集、衰减、查询。信号收集本身发生在 MemoryWriter 的 LLM 调用中
（不新增 API 调用），本模块负责持久化和查询。

五类信号：
  topic_avoidance   — 反复提及但从未深入的话题
  emotional_spike   — 情绪强度异常的时刻
  projection_target — 用户批评/羡慕的对象（可能是自我投射）
  hypothetical      — "如果可以重来"类型的假设性表达
  contradiction     — 用户在不同时间对同一件事表达相反立场
"""

import json
import sys
from datetime import datetime, timezone
from db.database import get_db
from db.schema import upsert_signal, get_signals_by_type


SIGNAL_TYPES = [
    "topic_avoidance",
    "emotional_spike",
    "projection_target",
    "hypothetical",
    "contradiction",
]

# 信号中文标签（用于 prompt 注入）
SIGNAL_LABELS = {
    "topic_avoidance": "他反复提到但从未深入",
    "emotional_spike": "那次他的情绪突然变了",
    "projection_target": "他谈论这个人/事的时候，可能是在说自己",
    "hypothetical": "他想象过另一种可能",
    "contradiction": "他在这里和自己矛盾了",
}


class SignalCollector:
    """信号管理器。收集来自 MemoryWriter 的信号，提供查询和衰减。"""

    @staticmethod
    def collect_from_writer(signals: list[dict]):
        """
        从 MemoryWriter 的输出中提取信号并写入数据库。

        Args:
            signals: [{"type": "topic_avoidance", "content": "...", "intensity": 0.6}, ...]
        """
        if not signals:
            return 0

        db = get_db()
        count = 0
        for sig in signals:
            sig_type = sig.get("type", "")
            if sig_type not in SIGNAL_TYPES:
                continue
            content = sig.get("content", "")
            if not content:
                continue
            intensity = sig.get("intensity", 0.5)
            upsert_signal(db, sig_type, content, intensity)
            count += 1
        return count

    @staticmethod
    def get_meaningful(min_intensity: float = 0.4, min_occurrence: int = 2) -> list[dict]:
        """
        获取值得注入 system prompt 的信号（强度足够且出现多次）。

        Returns:
            [{"type": "...", "content": "...", "intensity": 0.6, "occurrence_count": 3}, ...]
        """
        db = get_db()
        all_signals = get_signals_by_type(db)

        meaningful = []
        for sig in all_signals:
            intensity = sig.get("intensity", 0)
            count = sig.get("occurrence_count", 0)
            if intensity >= min_intensity and count >= min_occurrence:
                meaningful.append(sig)
        return meaningful

    @staticmethod
    def decay(decay_rate: float = 0.05):
        """
        衰减所有信号强度（未被重新激活的信号随时间减弱）。

        每次 MemoryRefiner 运行时调用此方法。
        decay_rate: 每次衰减的比例，默认 5%

        注意：intensity=0 的信号不会自动删除，保留在表中但查询时会被 min_intensity 过滤。
        """
        db = get_db()
        db.execute(
            "UPDATE signals SET intensity = MAX(0, intensity * ?)",
            (1.0 - decay_rate,),
        )
        db.commit()

    @staticmethod
    def build_context_text() -> str:
        """
        构建「你注意到的」段落文本，用于注入 system prompt。

        Returns:
            格式化的信号描述文本，信号不足时返回空字符串
        """
        meaningful = SignalCollector.get_meaningful(
            min_intensity=0.4, min_occurrence=2
        )
        if not meaningful:
            return ""

        # 按类型分组
        by_type: dict[str, list[str]] = {}
        for sig in meaningful:
            sig_type = sig.get("type", "")
            content = sig.get("content", "")
            if sig_type not in by_type:
                by_type[sig_type] = []
            by_type[sig_type].append(content)

        lines = ["【你注意到的】",
                 "以下是你长期以来观察到的。不要分析他，不要告诉他这些发现。",
                 "只需要在合适的时候，用那个'更完整的他'的视角和他说话。",
                 "如果他还没准备好，就不推进。你的沉默本身也是对话的一部分。",
                 ""]

        for sig_type, contents in by_type.items():
            label = SIGNAL_LABELS.get(sig_type, sig_type)
            for content in contents[:3]:  # 每种最多 3 条
                lines.append(f"- {label}：{content}")

        return "\n".join(lines)

    @staticmethod
    def get_stats() -> dict:
        """获取信号统计信息。"""
        db = get_db()
        stats = {}
        for st in SIGNAL_TYPES:
            count = db.execute(
                "SELECT COUNT(*) FROM signals WHERE type=?", (st,)
            ).fetchone()[0]
            if count > 0:
                stats[st] = count
        return stats
