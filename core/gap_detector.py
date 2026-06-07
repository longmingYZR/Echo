"""
Gap Detector — 缝隙感知（双通道分析）
========================================
捕捉用户"说的话"和"真实状态"之间的距离。

通道 1（语义，规则匹配）：模糊词、回避、话量偏离 → 免费，毫秒级
通道 2（情感，LLM 调用）：语义通道标记异常时才触发 → 省 API 费用

输出：gap_size (0-1)、建议模式、检测到的问题标签
"""

import sys
import re

# ═══════════════════════════════════════════
# 语义通道：规则匹配（无 LLM 调用）
# ═══════════════════════════════════════════

# 模糊词/回避信号
HEDGING_PATTERNS = [
    r'没什么',
    r'还好吧',
    r'就是',
    r'随便',
    r'算了',
    r'没事',
    r'无所谓',
    r'不重要',
    r'没什么大不了的',
    r'还行吧',
    r'都行',
    r'无所谓了',
    r'不说了',
    r'别提了',
]

# 防御性/回避信号
DEFLECTION_PATTERNS = [
    r'你想多了',
    r'不用管',
    r'没事的',
    r'我没事',
    r'我很好',
    r'挺好的',
    r'不用担心',
    r'别担心',
    r'还行',
]

# 话量极短（可能对应情绪低/不想说）
MIN_MEANINGFUL_LENGTH = 4  # 字符

# 情绪关键词（粗略分类）
EMOTION_KEYWORDS = {
    "焦虑": ["焦虑", "紧张", "压力", "担心", "怕", "不安", "失眠", "睡不着"],
    "疲惫": ["累", "困", "倦", "没劲", "没力气", "不想动"],
    "低落": ["难过", "伤心", "失望", "没意思", "无聊", "空虚"],
    "回避": ["不知道", "再说吧", "以后再说", "不想说", "别提"],
    "愤怒": ["烦死了", "受不了", "讨厌", "恨", "气死", "火大"],
}


class SemanticAnalyzer:
    """语义通道分析器。纯规则，无 LLM 开销。"""

    def analyze(self, text: str, history_length: int = 0) -> dict:
        """
        Args:
            text: 用户当前输入
            history_length: 当前 session 历史消息数（用于话量基线对比）

        Returns:
            dict: {has_signals, markers, estimated_emotion, deflection_score}
        """
        markers = []
        deflection_score = 0.0

        # 1. 模糊词检测
        hedging_count = 0
        for pattern in HEDGING_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                hedging_count += len(matches)
                markers.append(f"模糊词: {pattern}")

        if hedging_count > 0:
            deflection_score += min(1.0, hedging_count * 0.2)

        # 2. 防御性检测
        for pattern in DEFLECTION_PATTERNS:
            if re.search(pattern, text):
                deflection_score += 0.3
                markers.append(f"防御: {pattern}")

        # 3. 话量异常
        if len(text.strip()) < MIN_MEANINGFUL_LENGTH and len(text.strip()) > 0:
            markers.append("话量极短")
            deflection_score += 0.2

        # 4. 情绪关键词匹配
        estimated_emotion = "中性"
        max_matches = 0
        for emotion, keywords in EMOTION_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches > max_matches:
                max_matches = matches
                estimated_emotion = emotion

        if max_matches > 0:
            markers.append(f"情绪信号: {estimated_emotion}")

        deflection_score = min(1.0, deflection_score)

        return {
            "has_signals": deflection_score > 0.1,
            "markers": markers,
            "estimated_emotion": estimated_emotion,
            "deflection_score": deflection_score,
            "text_length": len(text.strip()),
            "hedging_count": hedging_count,
        }


# ═══════════════════════════════════════════
# 情感通道：LLM 分析（仅在语义通道标记异常时调用）
# ═══════════════════════════════════════════

EMOTION_ANALYSIS_PROMPT = """分析以下用户消息的情绪状态。

## 用户当前消息
{user_message}

## 上下文
近期情绪基线: {baseline_emotion}
对话历史: {recent_history}

## 要求
返回纯 JSON（不要 markdown 代码块）：

{{
  "surface_emotion": "用户表达的表面情绪",
  "likely_real_emotion": "用户可能的真实情绪",
  "gap_size": 0.0,
  "gap_reason": "为什么判断有/无缝隙",
  "emotional_intensity": 0.5
}}

gap_size 评分：
- 0.0-0.2: 没有缝隙，说啥就是啥
- 0.3-0.5: 轻微缝隙，可能有点情绪但大体真实
- 0.6-0.8: 明显缝隙，说的话和真实状态有距离
- 0.9-1.0: 巨大缝隙，说的话可能是反向的

注意：不要说"没什么"就一定是有事。结合上下文判断。有时候真的没什么。"""


class EmotionalAnalyzer:
    """情感通道分析器。使用 LLM 进行深度情绪分析。"""

    def __init__(self, llm_chat_fn):
        """
        Args:
            llm_chat_fn: (system_prompt, user_message) -> str
        """
        self._chat = llm_chat_fn

    def analyze(self, user_message: str, baseline_emotion: str = "未建立基线",
                recent_history: str = "") -> dict:
        """
        用 LLM 分析情绪缝隙。

        Returns:
            dict: {surface_emotion, likely_real_emotion, gap_size, gap_reason, emotional_intensity}
        """
        import json

        system_prompt = EMOTION_ANALYSIS_PROMPT.format(
            user_message=user_message,
            baseline_emotion=baseline_emotion,
            recent_history=recent_history or "无近期历史",
        )

        try:
            raw = self._chat(system_prompt, "分析这条消息的情绪")
            return json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[EmotionalAnalyzer] LLM 分析失败: {e}", file=sys.stderr)
            return {
                "surface_emotion": "未知",
                "likely_real_emotion": "未知",
                "gap_size": 0.0,
                "gap_reason": f"分析失败: {e}",
                "emotional_intensity": 0.3,
            }


# ═══════════════════════════════════════════
# 缝隙检测主类（组合双通道）
# ═══════════════════════════════════════════

class GapDetector:
    """双通道缝隙检测。语义通道免费快速，情感通道 LLM 精准。"""

    def __init__(self, llm_chat_fn):
        self.semantic = SemanticAnalyzer()
        self.emotional = EmotionalAnalyzer(llm_chat_fn)

    def detect(self, user_message: str, baseline_emotion: str = "未建立基线",
               recent_history: str = "", history_length: int = 0,
               prosody_modifier: float = 0.0) -> dict:
        """
        检测用户话语与真实状态之间的缝隙。

        Args:
            user_message: 用户当前输入
            baseline_emotion: 来自数据库的用户情绪基线
            recent_history: 最近几轮对话摘要
            history_length: 当前 session 历史消息数
            prosody_modifier: 韵律分析修正因子（0-1），来自 ProsodyAnalyzer。
                              语音特征捕捉文字无法捕捉的信号。
                              >0.3 时显著上调 gap_size。

        Returns:
            {
                "gap_detected": bool,
                "gap_size": float (0-1),
                "surface_emotion": str,
                "likely_real_emotion": str,
                "semantic_markers": list[str],
                "suggested_mode": str,
                "analysis_used_llm": bool,
            }
        """
        # 通道 1：语义规则（总是执行）
        semantic_result = self.semantic.analyze(user_message, history_length)

        # 韵律修正：产品文档核心逻辑——语音特征能捕捉文字捕捉不到的缝隙
        # 当韵律分析检测到异常（语速骤降+停顿增多+音量降低），上调语义层面的缝隙估计
        adjusted_deflection = semantic_result["deflection_score"]
        if prosody_modifier > 0.2:
            adjusted_deflection = min(1.0, adjusted_deflection + prosody_modifier * 0.5)
            semantic_result["deflection_score"] = adjusted_deflection
            if prosody_modifier > 0.0:
                semantic_result["markers"].append(f"韵律异常: +{prosody_modifier:.2f}")

        # 如果语义通道没有信号且话量正常 → 跳过 LLM
        if not semantic_result["has_signals"] and semantic_result["text_length"] >= 4:
            return {
                "gap_detected": False,
                "gap_size": prosody_modifier * 0.3,  # 韵律异常但话量正常 → 小缝隙
                "surface_emotion": semantic_result["estimated_emotion"],
                "likely_real_emotion": semantic_result["estimated_emotion"],
                "semantic_markers": semantic_result["markers"],
                "suggested_mode": "friend" if prosody_modifier < 0.4 else "mirror",
                "analysis_used_llm": False,
            }

        # 极短消息 → 直接标记缝隙，不调 LLM（LLM 无法从单个字里分析情绪）
        if semantic_result["text_length"] <= 3:
            return {
                "gap_detected": True,
                "gap_size": 0.5,
                "surface_emotion": "不明显",
                "likely_real_emotion": "不想说话",
                "semantic_markers": semantic_result["markers"],
                "suggested_mode": "companion",
                "analysis_used_llm": False,
            }

        # 如果语义通道有明显信号但不够强 → 小缝隙，不调 LLM
        if semantic_result["deflection_score"] < 0.3 and semantic_result["text_length"] >= 8:
            return {
                "gap_detected": semantic_result["deflection_score"] > 0.15,
                "gap_size": semantic_result["deflection_score"],
                "surface_emotion": semantic_result["estimated_emotion"],
                "likely_real_emotion": semantic_result["estimated_emotion"],
                "semantic_markers": semantic_result["markers"],
                "suggested_mode": "friend",
                "analysis_used_llm": False,
            }

        # 通道 2：LLM 情感分析（语义通道信号较强时触发）
        emotional_result = self.emotional.analyze(
            user_message,
            baseline_emotion=baseline_emotion,
            recent_history=recent_history,
        )

        gap_size = emotional_result.get("gap_size", semantic_result["deflection_score"])
        # 韵律修正：语音特征捕捉到的异常上调缝隙估计
        if prosody_modifier > 0.2:
            gap_size = min(1.0, gap_size + prosody_modifier * 0.4)
        gap_detected = gap_size > 0.25

        # 根据缝隙大小建议模式
        if gap_size > 0.6:
            suggested_mode = "companion"
        elif gap_size > 0.35:
            suggested_mode = "mirror"
        elif gap_size > 0.15:
            suggested_mode = "friend"
        else:
            suggested_mode = "friend"

        return {
            "gap_detected": gap_detected,
            "gap_size": gap_size,
            "surface_emotion": emotional_result.get("surface_emotion", semantic_result["estimated_emotion"]),
            "likely_real_emotion": emotional_result.get("likely_real_emotion", semantic_result["estimated_emotion"]),
            "semantic_markers": semantic_result["markers"],
            "suggested_mode": suggested_mode,
            "analysis_used_llm": True,
            "emotional_intensity": emotional_result.get("emotional_intensity", 0.5),
        }
