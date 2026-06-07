"""
Memory Awakener — 对话前构建 Echo
====================================
每次对话 session 开始时调用，从数据库构建动态 system prompt 上下文。

组装五个来源：
1. identity    — Echo 的自我认知
2. user_model  — 对用户的理解
3. signals     — 潜在自我信号（Phase 3）
4. relationship_moments — 重要时刻
5. echo_inner  — Echo 没说出口的话

同时注入时间感（距离上次对话过了多久）。
"""

from datetime import datetime, timezone
from db.database import get_db
from db.schema import (
    get_identity,
    get_user_model,
    get_important_moments,
    get_latest_echo_inner,
)


# 冷启动钩子问题——不是"你喜欢什么"，而是能快速建立模式记忆的深度问题
COLD_START_HOOKS = [
    "你上次真正放松是什么时候？",
    "有没有什么事是你一直在想但从没跟任何人说过的？",
    "你觉得自己在逃避什么？——不用回答我，我只是问问。",
    "如果你的人生是一本书，现在这一章叫什么名字？",
    "什么时候你会觉得'这才是真正的我'？",
    "有什么事是你以前觉得很重要，现在觉得无所谓的？",
    "你羡慕过谁？你觉得你羡慕的是他们的什么？",
    "你批评过谁？有时候我们对别人的批评，其实是在批评自己不敢承认的那部分。",
]


class MemoryAwakener:
    """对话前构建 Echo 的完整上下文。"""

    @staticmethod
    def time_since_last() -> str:
        """计算距离上次对话的时间描述。"""
        db = get_db()
        identity = get_identity(db)
        last_updated = identity.get("last_updated", "")
        if not last_updated:
            return "这是你们第一次对话"

        try:
            last_dt = datetime.fromisoformat(last_updated)
            delta = datetime.now(timezone.utc) - last_dt

            minutes = delta.total_seconds() / 60
            hours = minutes / 60
            days = delta.days

            if minutes < 5:
                return "刚刚才说过话"
            elif minutes < 60:
                return f"距离上次对话过去了{int(minutes)}分钟"
            elif hours < 24:
                return f"距离上次对话过去了{int(hours)}小时"
            elif days == 1:
                return "距离上次对话过去了1天"
            elif days < 7:
                return f"距离上次对话过去了{days}天"
            elif days < 30:
                weeks = days // 7
                return f"距离上次对话过去了{weeks}周"
            else:
                months = days // 30
                return f"距离上次对话过去了{months}个月"
        except Exception:
            return "距离上次对话已经过去一段时间了"

    @staticmethod
    def build_context() -> dict:
        """
        构建注入 system prompt 的上下文块。

        Returns:
            dict 包含:
            - time_context: 时间感描述
            - identity_text: Echo 的自我认知
            - user_model_text: 对用户的四层理解
            - relationship_text: 重要时刻摘要
            - echo_inner_text: Echo 内心状态
            - full_context: 拼接好的完整上下文文本
        """
        db = get_db()

        identity = get_identity(db)
        user_model = get_user_model(db)
        moments = get_important_moments(db, min_weight="high", limit=3)
        inner = get_latest_echo_inner(db)

        # 时间感
        time_context = MemoryAwakener.time_since_last()

        # 身份
        identity_text = identity.get("self_perception", "")
        stage_text = identity.get("relationship_stage", "")

        # 用户模型摘要
        user_lines = []
        layer_names = {
            "surface": "表层",
            "pattern": "情绪模式",
            "hidden": "深层",
            "growth": "变化趋势",
        }
        for layer in ["surface", "pattern", "hidden", "growth"]:
            m = user_model.get(layer, {})
            content = m.get("content", "") if isinstance(m, dict) else ""
            if content and content not in ("尚未建立足够对话数据", "尚无足够数据追踪变化", "尚未发现深层话题", "刚开始了解"):
                user_lines.append(f"{layer_names.get(layer, layer)}: {content}")

        user_model_text = "\n".join(user_lines) if user_lines else "你还在认识他"

        # 潜在自我信号（Phase 3）
        latent_self_text = ""
        try:
            from core.signal_collector import SignalCollector
            latent_self_text = SignalCollector.build_context_text()
        except Exception:
            pass

        # 重要时刻
        relationship_text = ""
        if moments:
            moment_lines = [f"- {m['moment']}" for m in moments]
            relationship_text = "你们之间重要的时刻：\n" + "\n".join(moment_lines)

        # Echo 内心
        echo_inner_text = ""
        if inner:
            feeling = inner.get("feeling", "")
            unsaid = inner.get("unsaid", "")
            echo_inner_text = f"你此刻的感受：{feeling}"
            if unsaid:
                echo_inner_text += f"\n你没说出口的话：{unsaid}"

        # 拼接完整上下文
        parts = [f"【时间】{time_context}"]
        if stage_text:
            parts.append(f"【关系阶段】{stage_text}")
        if user_model_text:
            parts.append(f"【你了解的他】\n{user_model_text}")
        if latent_self_text:
            parts.append(latent_self_text)
        if relationship_text:
            parts.append(relationship_text)
        if echo_inner_text:
            parts.append(f"【你的内心】{echo_inner_text}")

        # 冷启动检测：用户模型尚未充分建立
        cold_start_text = MemoryAwakener._build_cold_start_context(user_model)
        if cold_start_text:
            parts.append(cold_start_text)

        full_context = "\n\n".join(parts)

        return {
            "time_context": time_context,
            "identity_text": identity_text,
            "user_model_text": user_model_text,
            "latent_self_text": latent_self_text,
            "relationship_text": relationship_text,
            "echo_inner_text": echo_inner_text,
            "cold_start_text": cold_start_text,
            "full_context": full_context,
        }

    @staticmethod
    def _build_cold_start_context(user_model: dict) -> str:
        """
        当用户模型置信度普遍偏低时，注入主动提问策略。

        只在早期对话中使用。模型建立后自动停用。
        """
        # 计算平均置信度
        confidences = []
        for layer in ["surface", "pattern", "hidden", "growth"]:
            m = user_model.get(layer, {})
            conf = m.get("confidence", 0.0) if isinstance(m, dict) else 0.0
            confidences.append(conf)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # 如果任何一层的实际内容还是默认值，视为冷启动
        has_real_content = False
        for layer in ["surface", "pattern"]:
            m = user_model.get(layer, {})
            content = m.get("content", "") if isinstance(m, dict) else ""
            if content and content not in (
                "尚未建立足够对话数据", "刚开始了解",
            ):
                has_real_content = True
                break

        if not has_real_content or avg_confidence < 0.4:
            import random
            hook = random.choice(COLD_START_HOOKS)
            return (
                "【对话策略：冷启动】\n"
                "你们还在认识阶段。你还不了解他，他也还在试探你。\n"
                "在自然对话中，如果出现合适的沉默或话题转接点，可以问一个真正有意义的问题——\n"
                "不是'你喜欢什么'这种表面的，而是能触碰到真实的东西。\n"
                f"比如，在合适的时候你可以问：'{hook}'\n"
                "但不要生硬地插入。如果现在的对话不需要，就不要问。\n"
                "记住你在认识他。你的问题是你真正想知道的，不是例行公事。"
            )
        return ""
