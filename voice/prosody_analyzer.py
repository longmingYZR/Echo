"""
Prosody Analyzer — 语音韵律特征提取
======================================
从原始 PCM 音频中提取文本层面无法获取的信号，增强缝隙检测。

特征：
  speech_rate    — 语速（音节/秒），变慢=谨慎/回避，变快=激动/焦虑
  pause_ratio    — 停顿时间占比，异常多=在字斟句酌
  volume_mean    — 平均音量
  volume_variance — 音量变化度，降低=压抑，升高=激动
  pitch_mean     — 基频均值（Hz）
  pitch_range    — 基频范围

设计原则：
  - 优先使用 librosa（精度高），不可用时回退到纯 numpy 实现
  - 所有特征归一化到 0-1 范围方便与 GapDetector 的语义分数融合
  - 不新增 LLM 调用——只产出数值特征
"""

import sys
import math
import numpy as np


# ═══════════════════════════════════════════
# 默认基线值（无历史数据时的回退）
# ═══════════════════════════════════════════

DEFAULT_BASELINE = {
    "speech_rate": 3.0,       # 中文正常语速 ~3 字/秒
    "pause_ratio": 0.25,       # 正常停顿占比
    "volume_mean": 1500.0,     # RMS 均值（int16）
    "volume_variance": 500.0,  # RMS 标准差
    "pitch_mean": 180.0,       # 男声平均基频（偏高估计，中文女声 ~220）
    "pitch_range": 60.0,       # 正常基频范围
}


class ProsodyAnalyzer:
    """从 PCM 音频提取韵律特征。"""

    def __init__(self, baseline: dict = None, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.baseline = {**DEFAULT_BASELINE, **(baseline or {})}
        self._librosa = None

    @property
    def librosa(self):
        """延迟导入 librosa（可选依赖）。"""
        if self._librosa is None:
            try:
                import librosa as _librosa
                self._librosa = _librosa
            except ImportError:
                self._librosa = False
        return self._librosa if self._librosa is not False else None

    # ── 公开 API ──

    def analyze(self, pcm_data: bytes | np.ndarray) -> dict:
        """
        分析一段 PCM 音频，返回所有韵律特征。

        Args:
            pcm_data: int16 PCM 字节或 numpy 数组

        Returns:
            {
                "speech_rate": float,
                "pause_ratio": float,
                "volume_mean": float,
                "volume_variance": float,
                "pitch_mean": float,
                "pitch_range": float,
                "duration_s": float,
            }
        """
        # 统一转为 float32 numpy 数组
        audio = self._to_float32(pcm_data)
        if len(audio) == 0:
            return self._empty_result()

        duration_s = len(audio) / self.sample_rate

        # 静音检测（帧级别）
        is_speech_frames = self._detect_speech(audio)

        # 各特征
        speech_rate = self._estimate_speech_rate(audio, is_speech_frames, duration_s)
        pause_ratio = self._compute_pause_ratio(is_speech_frames)
        volume_mean, volume_variance = self._compute_volume(audio, is_speech_frames)
        pitch_mean, pitch_range = self._estimate_pitch(audio, is_speech_frames)

        return {
            "speech_rate": round(speech_rate, 2),
            "pause_ratio": round(pause_ratio, 3),
            "volume_mean": round(volume_mean, 1),
            "volume_variance": round(volume_variance, 1),
            "pitch_mean": round(pitch_mean, 1),
            "pitch_range": round(pitch_range, 1),
            "duration_s": round(duration_s, 2),
        }

    def compare_to_baseline(self, features: dict) -> dict:
        """
        将当前特征与基线对比，计算偏离度。

        Returns:
            {
                "deviation_score": float,  # 综合偏离度 0-1
                "deviations": {
                    "speech_rate_slower": bool,   # 语速显著变慢
                    "speech_rate_faster": bool,   # 语速显著变快
                    "pause_more": bool,           # 停顿显著增多
                    "volume_lower": bool,         # 音量显著降低
                    "volume_higher": bool,        # 音量显著升高
                    "pitch_lower": bool,          # 音高显著降低
                    "pitch_narrower": bool,       # 音域变窄（压抑）
                }
            }
        """
        if not features:
            return {"deviation_score": 0.0, "deviations": {}}

        baseline = self.baseline
        deviations = {}
        weight = 0.0

        # 语速偏离
        sr = features.get("speech_rate", baseline["speech_rate"])
        sr_ratio = sr / max(baseline["speech_rate"], 0.1)
        if sr_ratio < 0.7:
            deviations["speech_rate_slower"] = True
            weight += 0.25
        elif sr_ratio > 1.5:
            deviations["speech_rate_faster"] = True
            weight += 0.15

        # 停顿偏离
        pr = features.get("pause_ratio", baseline["pause_ratio"])
        pr_ratio = pr / max(baseline["pause_ratio"], 0.01)
        if pr_ratio > 1.8:
            deviations["pause_more"] = True
            weight += 0.25

        # 音量偏离
        vm = features.get("volume_mean", baseline["volume_mean"])
        vm_ratio = vm / max(baseline["volume_mean"], 1.0)
        if vm_ratio < 0.6:
            deviations["volume_lower"] = True
            weight += 0.2
        elif vm_ratio > 1.6:
            deviations["volume_higher"] = True
            weight += 0.1

        # 音高偏离
        pm = features.get("pitch_mean", baseline["pitch_mean"])
        pm_ratio = pm / max(baseline["pitch_mean"], 1.0)
        if pm_ratio < 0.85:
            deviations["pitch_lower"] = True
            weight += 0.15

        # 音域偏离
        prange = features.get("pitch_range", baseline["pitch_range"])
        prange_ratio = prange / max(baseline["pitch_range"], 1.0)
        if prange_ratio < 0.6:
            deviations["pitch_narrower"] = True
            weight += 0.2

        return {
            "deviation_score": round(min(1.0, weight), 2),
            "deviations": deviations,
        }

    def update_baseline(self, features: dict, alpha: float = 0.1):
        """
        指数移动平均更新基线。

        Args:
            features: analyze() 的返回值
            alpha: 更新速率（0.1 = 新值占 10%）
        """
        for key in DEFAULT_BASELINE:
            if key in features and features[key] is not None:
                old = self.baseline.get(key, DEFAULT_BASELINE[key])
                self.baseline[key] = old * (1 - alpha) + features[key] * alpha

    # ── 内部方法 ──

    def _to_float32(self, pcm_data: bytes | np.ndarray) -> np.ndarray:
        if isinstance(pcm_data, bytes):
            return np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        elif isinstance(pcm_data, np.ndarray):
            if pcm_data.dtype == np.int16:
                return pcm_data.astype(np.float32)
            return pcm_data.astype(np.float32)
        return np.array([], dtype=np.float32)

    def _detect_speech(self, audio: np.ndarray, frame_ms: int = 30) -> np.ndarray:
        """
        检测每一帧是否为语音（非静音）。

        Returns:
            bool 数组，长度 = 帧数
        """
        frame_len = int(self.sample_rate * frame_ms / 1000)
        n_frames = len(audio) // frame_len
        if n_frames == 0:
            return np.array([], dtype=bool)

        is_speech = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            start = i * frame_len
            end = start + frame_len
            frame = audio[start:end]
            energy = np.sqrt(np.mean(frame ** 2))
            # 静音阈值：低于 300 (int16 scale) 视为静音
            is_speech[i] = energy > 300.0

        return is_speech

    def _frame_to_sample_mask(self, is_speech_frames: np.ndarray, audio_len: int,
                               frame_ms: int = 30) -> np.ndarray:
        """将帧级别的语音检测扩展为采样级别的布尔掩码。"""
        frame_len = int(self.sample_rate * frame_ms / 1000)
        mask = np.zeros(audio_len, dtype=bool)
        for i, speech in enumerate(is_speech_frames):
            if speech:
                start = i * frame_len
                end = min(start + frame_len, audio_len)
                mask[start:end] = True
        return mask

    def _compute_volume(self, audio: np.ndarray, is_speech_frames: np.ndarray) -> tuple:
        """计算语音段的音量和音量变化（使用采样级掩码）。"""
        mask = self._frame_to_sample_mask(is_speech_frames, len(audio))
        if not np.any(mask):
            return 0.0, 0.0
        speech_audio = np.abs(audio[mask])
        return float(np.mean(speech_audio)), float(np.std(speech_audio))

    def _compute_pause_ratio(self, is_speech: np.ndarray) -> float:
        if len(is_speech) == 0:
            return 0.0
        return 1.0 - np.mean(is_speech.astype(float))

    def _estimate_speech_rate(self, audio: np.ndarray, is_speech: np.ndarray,
                               duration_s: float) -> float:
        """
        估算语速（音节/秒）。

        方法：基于过零率变化检测音节边界。
        中文每个汉字 ≈ 一个音节，过零率的突变通常对应音节边界。
        """
        if duration_s < 0.1 or not np.any(is_speech):
            return 0.0

        # 只对语音段计算
        speech_frames = []
        frame_len = int(self.sample_rate * 0.01)  # 10ms frames
        for i, speech in enumerate(is_speech):
            if speech:
                start = i * frame_len
                end = start + frame_len
                if end <= len(audio):
                    speech_frames.append(audio[start:end])

        if not speech_frames:
            return 0.0

        # 过零率
        zcr_values = []
        for frame in speech_frames:
            if len(frame) >= 2:
                zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
                zcr_values.append(zcr)

        if not zcr_values:
            return 0.0

        zcr = np.array(zcr_values)
        # 过零率突变次数 / 语音时长 ≈ 音节速率
        zcr_diff = np.abs(np.diff(zcr))
        # 阈值：过零率变化超过 1.5 倍标准差视为音节边界
        threshold = np.std(zcr_diff) * 1.5 if np.std(zcr_diff) > 0 else 0.01
        syllable_boundaries = np.sum(zcr_diff > threshold)

        speech_duration = np.sum(is_speech) * 0.03  # 30ms per frame
        if speech_duration < 0.1:
            return 0.0

        return syllable_boundaries / speech_duration

    def _estimate_pitch(self, audio: np.ndarray, is_speech: np.ndarray) -> tuple:
        """
        估算基频均值和范围。

        优先使用 librosa.pyin（高精度），回退到自相关法。
        """
        # 只对语音段计算
        speech_segments = []
        frame_len = int(self.sample_rate * 0.03)
        for i, speech in enumerate(is_speech):
            if speech:
                start = i * frame_len
                end = start + frame_len
                if end <= len(audio):
                    speech_segments.append(audio[start:end])

        if not speech_segments:
            return 0.0, 0.0

        speech_audio = np.concatenate(speech_segments)

        # 尝试 librosa
        lb = self.librosa
        if lb:
            try:
                f0, voiced_flag, _ = lb.pyin(
                    speech_audio.astype(np.float64),
                    fmin=80,
                    fmax=500,
                    sr=self.sample_rate,
                )
                f0_voiced = f0[voiced_flag]
                if len(f0_voiced) > 0:
                    return float(np.mean(f0_voiced)), float(np.ptp(f0_voiced))
            except Exception:
                pass

        # 回退：自相关法
        if len(speech_audio) < self.sample_rate * 0.02:
            return 0.0, 0.0

        pitch_values = []
        min_lag = int(self.sample_rate / 500)  # 500 Hz max
        max_lag = int(self.sample_rate / 80)   # 80 Hz min

        step = max(1, len(speech_audio) // 10)
        for offset in range(0, len(speech_audio) - max_lag, step):
            segment = speech_audio[offset:offset + max_lag * 2]
            if len(segment) < max_lag * 2:
                continue

            # 归一化自相关
            autocorr = np.correlate(segment, segment, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)

            # 找最大峰值
            search = autocorr[min_lag:max_lag]
            if len(search) == 0:
                continue
            peak_idx = np.argmax(search) + min_lag
            if autocorr[peak_idx] > 0.3:  # 有意义的周期性
                pitch = self.sample_rate / peak_idx
                if 80 <= pitch <= 500:
                    pitch_values.append(pitch)

        if pitch_values:
            return float(np.mean(pitch_values)), float(np.ptp(pitch_values))
        return 0.0, 0.0

    def _empty_result(self) -> dict:
        return {
            "speech_rate": 0.0,
            "pause_ratio": 0.0,
            "volume_mean": 0.0,
            "volume_variance": 0.0,
            "pitch_mean": 0.0,
            "pitch_range": 0.0,
            "duration_s": 0.0,
        }


# ═══════════════════════════════════════════
# 与 GapDetector 的桥接
# ═══════════════════════════════════════════

def prosody_to_gap_modifier(prosody_features: dict, baseline_comparison: dict) -> float:
    """
    将韵律分析结果转化为缝隙检测的修正因子（0-1）。

    产品文档核心逻辑：
    语音特征能捕捉文字捕捉不到的缝隙信号。
    这个函数产出的是一个"确信度加成"——韵律异常越多，文字层的 gap_size 上调越多。

    Returns:
        修正因子（0.0 = 韵律无异常，1.0 = 韵律强烈暗示有缝隙）
    """
    if not prosody_features or not baseline_comparison:
        return 0.0

    deviations = baseline_comparison.get("deviations", {})
    base_score = baseline_comparison.get("deviation_score", 0.0)

    # 不同偏离模式的权重（对应文档描述的情绪信号）
    mod = 0.0

    # 语速变慢 + 停顿增多 = 高度回避/字斟句酌 → 缝隙可能性大
    if deviations.get("speech_rate_slower") and deviations.get("pause_more"):
        mod += 0.35

    # 音量降低 = 压抑
    if deviations.get("volume_lower"):
        mod += 0.2

    # 音域变窄 = 情绪压抑/疲惫
    if deviations.get("pitch_narrower"):
        mod += 0.2

    # 语速变快 + 音量升高 = 掩饰性的兴奋
    if deviations.get("speech_rate_faster") and deviations.get("volume_higher"):
        mod += 0.15

    # 音高降低（对于男性）= 低落/疲惫
    if deviations.get("pitch_lower"):
        mod += 0.1

    return min(1.0, max(base_score, mod))
