"""
Echo Voice Loop — AI 硬件语音对话核心框架
============================================
语音链路：麦克风 → VAD 自动检测 → Whisper STT → DeepSeek LLM → edge-tts TTS → 扬声器

模块可独立替换。PC 先跑通，代码结构支持零改动迁移到树莓派/嵌入式 Linux。
"""

import os
import sys
import time
import json
import queue
import threading
import tempfile
import subprocess
from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np


# ═══════════════════════════════════════════════════════════════
# 配置（可通过环境变量覆盖）
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # ── 录音 ──
    "silence_threshold":      500,        # 音量低于此值判定为静音（安静环境可调低至 300）
    "silence_duration":       1.5,        # 连续静音多少秒后停止录音
    "min_recording_duration": 0.5,        # 低于此时长丢弃，防止误触发
    "max_recording_duration": 30,         # 单次录音上限（秒）
    "sample_rate":            16000,      # Whisper 标准采样率，不要修改

    # ── Whisper ──
    "whisper_model":          "base",     # tiny | base | small | medium
    "whisper_language":       "zh",       # 指定语言跳过检测，树莓派节省 ~0.5s

    # ── LLM ──
    "llm_api_url":            "https://api.deepseek.com/v1/chat/completions",
    "llm_model":              "deepseek-chat",
    "llm_api_key":            os.environ.get("DEEPSEEK_API_KEY", ""),
    "llm_max_tokens":         1024,
    "llm_temperature":        0.7,
    "memory_max_turns":       10,         # 滑动窗口保留最近 N 轮对话

    # ── TTS ──
    "tts_voice":              "zh-CN-XiaoxiaoNeural",  # Echo 伴侣默认温柔女声
    "tts_rate":               "+0%",      # 语速调整（edge-tts 格式）

    # ── Echo 集成 ──
    "echo_system_prompt":     None,       # None 使用默认提示词；设为字符串则覆盖
}


# ═══════════════════════════════════════════════════════════════
# 1. AudioRecorder — 基于 VAD 的自动录音模块
# ═══════════════════════════════════════════════════════════════

class AudioRecorder:
    """VAD（语音活动检测）自动录音。说话即开始，静音即停止，无需手动按键。"""

    def __init__(self, config: dict = None):
        cfg = {**CONFIG, **(config or {})}
        self.sample_rate          = cfg["sample_rate"]
        self.silence_threshold    = cfg["silence_threshold"]
        self.silence_duration     = cfg["silence_duration"]
        self.min_recording_duration = cfg["min_recording_duration"]
        self.max_recording_duration = cfg["max_recording_duration"]
        self._pyaudio = None

    @property
    def pyaudio(self):
        if self._pyaudio is None:
            import pyaudio
            self._pyaudio = pyaudio
        return self._pyaudio

    def record(self) -> bytes | None:
        """录一段语音，返回 WAV 字节。静音超时自动停止，短于 min 丢弃返回 None。"""
        pa = self.pyaudio
        p = pa.PyAudio()

        try:
            stream = p.open(
                format=pa.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=int(self.sample_rate * 0.1),  # 100ms chunks
            )
        except OSError as e:
            p.terminate()
            raise RuntimeError(f"无法打开麦克风：{e}") from e

        frames: list[bytes] = []
        silent_chunks = 0
        silent_limit  = int(self.silence_duration / 0.1)
        max_chunks    = int(self.max_recording_duration / 0.1)
        min_chunks    = int(self.min_recording_duration / 0.1)
        has_speech    = False

        try:
            while len(frames) < max_chunks:
                data = stream.read(int(self.sample_rate * 0.1), exception_on_overflow=False)
                frames.append(data)

                # 计算音量
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                volume = np.sqrt(np.mean(audio_chunk ** 2))

                if volume > self.silence_threshold:
                    has_speech = True
                    silent_chunks = 0
                else:
                    silent_chunks += 1

                # 有语音后连续静音 → 停止
                if has_speech and silent_chunks >= silent_limit:
                    break

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        if not has_speech or len(frames) < min_chunks:
            return None

        return self._frames_to_wav(frames)

    def _frames_to_wav(self, frames: list[bytes]) -> bytes:
        """将 PCM 帧封装为 WAV 格式字节。"""
        import io
        import struct
        import wave

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # paInt16 = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
# 2. WhisperSTT — 本地语音识别模块
# ═══════════════════════════════════════════════════════════════

class WhisperSTT:
    """OpenAI Whisper 本地模型，完全离线运行，无需 API。"""

    def __init__(self, config: dict = None):
        cfg = {**CONFIG, **(config or {})}
        self.model_size = cfg["whisper_model"]
        self.language   = cfg["whisper_language"]
        self._model     = None

    @property
    def model(self):
        if self._model is None:
            import whisper
            print(f"[Whisper] 加载模型 '{self.model_size}'...", file=sys.stderr)
            self._model = whisper.load_model(self.model_size)
            print("[Whisper] 模型加载完成", file=sys.stderr)
        return self._model

    def transcribe(self, wav_bytes: bytes) -> str:
        """将 WAV 字节转录为文本。"""
        # 写入临时文件（whisper 需要文件路径）
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            result = self.model.transcribe(
                tmp_path,
                language=self.language,
                fp16=False,
                verbose=False,
            )
            text = result['text'].strip()
            return text
        finally:
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════
# 3. ConversationMemory — 滑动窗口对话记忆
# ═══════════════════════════════════════════════════════════════

class ConversationMemory:
    """维护最近 N 轮对话，防止上下文超出 token 限制。"""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: list[dict] = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self):
        """保留最近 max_turns 轮（每轮 user + assistant）。"""
        max_msgs = self.max_turns * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

    def get_messages(self) -> list[dict]:
        return list(self.messages)

    def clear(self):
        self.messages.clear()


# ═══════════════════════════════════════════════════════════════
# 4. LLMClient — 大模型接口
# ═══════════════════════════════════════════════════════════════

class LLMClient:
    """封装 DeepSeek API（兼容 OpenAI 格式）。
    修改 llm_api_url 可切换任意兼容接口（OpenAI、Ollama 本地等）。
    """

    def __init__(self, config: dict = None):
        cfg = {**CONFIG, **(config or {})}
        self.api_url     = cfg["llm_api_url"]
        self.model       = cfg["llm_model"]
        self.api_key     = cfg["llm_api_key"]
        self.max_tokens  = cfg["llm_max_tokens"]
        self.temperature = cfg["llm_temperature"]
        self.memory      = ConversationMemory(max_turns=cfg["memory_max_turns"])

    def chat(self, user_message: str, system_prompt: str = "") -> str:
        """发送用户消息，返回 AI 回复。自动维护对话历史。"""
        import requests

        self.memory.add("user", user_message)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.memory.get_messages())

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(self.api_url, headers=headers, json=body, timeout=30)
            if not resp.ok:
                raise RuntimeError(f"LLM API 错误 ({resp.status_code}): {resp.text[:300]}")
            data = resp.json()
            reply = data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            raise RuntimeError(f"LLM 网络错误: {e}") from e

        self.memory.add("assistant", reply)
        return reply

    def reset_memory(self):
        self.memory.clear()


# ═══════════════════════════════════════════════════════════════
# 5. EdgeTTS — 语音合成模块
# ═══════════════════════════════════════════════════════════════

class EdgeTTS:
    """微软 edge-tts，免费、无需 API key，支持中文多音色。"""

    def __init__(self, config: dict = None):
        cfg = {**CONFIG, **(config or {})}
        self.voice = cfg["tts_voice"]
        self.rate  = cfg["tts_rate"]

    def speak(self, text: str):
        """将文本合成为语音并播放。"""
        import edge_tts

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            tmp_path = f.name

        try:
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            communicate.save_sync(tmp_path)

            # 播放音频
            self._play_audio(tmp_path)
        finally:
            # 延迟清理，避免播放未完成
            threading.Thread(target=lambda: (time.sleep(2), os.unlink(tmp_path)), daemon=True).start()

    def _play_audio(self, filepath: str):
        """跨平台音频播放。"""
        system = sys.platform
        try:
            if system == 'darwin':
                subprocess.run(['afplay', filepath], check=True)
            elif system == 'win32':
                # Windows: 使用系统默认播放器
                import winsound
                # winsound 只支持 WAV，这里用另一种方式
                subprocess.run(
                    ['powershell', '-c',
                     f'(New-Object Media.SoundPlayer "{filepath}").PlaySync()'],
                    check=True, capture_output=True,
                )
            else:
                # Linux / 树莓派：需要 mpg123
                subprocess.run(['mpg123', '-q', filepath], check=True)
        except FileNotFoundError:
            # 回退：尝试 ffplay
            try:
                subprocess.run(['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', filepath], check=True)
            except FileNotFoundError:
                print(f"[TTS] 无法播放音频，文件已保存至: {filepath}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# 6. EchoVoiceLoop — 主状态机
# ═══════════════════════════════════════════════════════════════

class LoopState(Enum):
    IDLE       = auto()  # 待机，等待语音输入
    LISTENING  = auto()  # 麦克风录音中
    PROCESSING = auto()  # STT + LLM 推理中
    SPEAKING   = auto()  # TTS 播放中
    ERROR      = auto()  # 异常状态


class EchoVoiceLoop:
    """将四个模块串联为完整对话循环，内置五态状态机。

    继承此类并重写 _build_system_prompt() 即可接入 Echo 角色/记忆系统。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是 Echo，一个温柔、体贴的 AI 伴侣。"
        "你说话简洁但温暖，像一位随时在身边的知心朋友。"
        "用中文回复，保持自然的口语风格。"
        "回复控制在 2-4 句话以内，像真实对话而非书面文章。"
    )

    def __init__(self, config: dict = None):
        cfg = {**CONFIG, **(config or {})}
        self.config   = cfg
        self.state    = LoopState.IDLE
        self.recorder = AudioRecorder(cfg)
        self.stt      = WhisperSTT(cfg)
        self.llm      = LLMClient(cfg)
        self.tts      = EdgeTTS(cfg)

    # ── 钩子方法（子类可重写） ──

    def _build_system_prompt(self) -> str:
        """构建系统提示词。接入 Echo 角色/记忆系统只需重写此方法。"""
        override = self.config.get("echo_system_prompt")
        if override:
            return override
        return self.DEFAULT_SYSTEM_PROMPT

    def _on_state_change(self, old: LoopState, new: LoopState):
        """状态变更回调（子类可用于日志/UI 通知）。"""
        pass

    def _on_speech_recognized(self, text: str):
        """语音识别完成回调。"""
        print(f"\n👤 你: {text}")

    def _on_response(self, text: str):
        """LLM 回复回调。"""
        print(f"🤖 Echo: {text}")

    # ── 核心循环 ──

    def run(self):
        """启动主循环。Ctrl+C 退出。"""
        print("=" * 50)
        print("  Echo Voice Loop 已启动")
        print("  直接说话即可，Ctrl+C 退出")
        print("=" * 50)
        self._set_state(LoopState.IDLE)

        try:
            while True:
                self._loop_once()
        except KeyboardInterrupt:
            print("\n\nEcho Voice Loop 已退出。再见 👋")

    def _loop_once(self):
        """单次对话循环。"""
        # ── 1. 待机，等待语音 ──
        self._set_state(LoopState.IDLE)

        # ── 2. 录音 ──
        self._set_state(LoopState.LISTENING)
        try:
            wav_bytes = self.recorder.record()
        except Exception as e:
            self._set_state(LoopState.ERROR)
            print(f"[错误] 录音失败: {e}", file=sys.stderr)
            time.sleep(1)
            return

        if wav_bytes is None:
            # 录音太短或无声，静默回到 IDLE
            return

        # ── 3. STT + LLM ──
        self._set_state(LoopState.PROCESSING)

        try:
            user_text = self.stt.transcribe(wav_bytes)
        except Exception as e:
            self._set_state(LoopState.ERROR)
            print(f"[错误] 语音识别失败: {e}", file=sys.stderr)
            return

        if not user_text:
            return

        self._on_speech_recognized(user_text)

        try:
            system_prompt = self._build_system_prompt()
            reply = self.llm.chat(user_text, system_prompt)
        except Exception as e:
            self._set_state(LoopState.ERROR)
            print(f"[错误] LLM 请求失败: {e}", file=sys.stderr)
            return

        self._on_response(reply)

        # ── 4. TTS 播放 ──
        self._set_state(LoopState.SPEAKING)
        try:
            self.tts.speak(reply)
        except Exception as e:
            self._set_state(LoopState.ERROR)
            print(f"[错误] TTS 播放失败: {e}", file=sys.stderr)

    def _set_state(self, new_state: LoopState):
        old = self.state
        self.state = new_state
        if old != new_state:
            self._on_state_change(old, new_state)

    @property
    def status(self) -> str:
        state_names = {
            LoopState.IDLE:       "待机中",
            LoopState.LISTENING:  "聆听中",
            LoopState.PROCESSING: "思考中",
            LoopState.SPEAKING:   "回复中",
            LoopState.ERROR:      "异常",
        }
        return state_names.get(self.state, "未知")


# ═══════════════════════════════════════════════════════════════
# 7. EchoVoiceLoopExtended — Echo 集成示例
# ═══════════════════════════════════════════════════════════════

class EchoVoiceLoopExtended(EchoVoiceLoop):
    """对接 Echo 伴侣系统的扩展子类。
    只需重写 _build_system_prompt 和 _loop_once 两个方法。
    """

    def _build_system_prompt(self) -> str:
        """从 Echo 角色系统获取提示词。"""
        # 从 Echo 的 memory.js 逻辑映射到 Python 端：
        # 1. 加载 Echo 角色定义
        # 2. 注入 spaced-repetition 记忆引擎的内容
        # 3. 注入今日推荐（getDailyPick）

        base_prompt = self.DEFAULT_SYSTEM_PROMPT

        # 如果安装了 echo.character 模块
        try:
            from echo.character import get_system_prompt
            base_prompt = get_system_prompt()
        except ImportError:
            pass

        return base_prompt

    def _loop_once(self):
        """可在此加入唤醒词检测。"""
        # 示例：等待唤醒词（可选）
        # if not self._wait_for_wakeword():
        #     time.sleep(0.1)
        #     return
        super()._loop_once()

    # 预留：唤醒词检测
    def _wait_for_wakeword(self) -> bool:
        """检测唤醒词（可选扩展）。默认始终返回 True。"""
        # 可接入 openWakeWord 或 Porcupine
        return True


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loop = EchoVoiceLoop()
    loop.run()
