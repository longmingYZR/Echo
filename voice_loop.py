"""
Echo Voice Loop — AI 硬件语音对话核心框架
============================================
语音链路：麦克风 → VAD 自动检测 → Whisper STT → DeepSeek LLM → edge-tts TTS → 扬声器

记忆系统：Memory Awakener（对话前唤醒）→ Memory Writer（对话后提炼）→ Memory Refiner（定期深度提炼）

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
import uuid
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
    """VAD（语音活动检测）自动录音。说话即开始，静音即停止，无需手动按键。

    使用 sounddevice 代替 pyaudio（跨平台兼容性更好，纯 pip 安装）。
    """

    def __init__(self, config: dict = None):
        cfg = {**CONFIG, **(config or {})}
        self.sample_rate          = cfg["sample_rate"]
        self.silence_threshold    = cfg["silence_threshold"]
        self.silence_duration     = cfg["silence_duration"]
        self.min_recording_duration = cfg["min_recording_duration"]
        self.max_recording_duration = cfg["max_recording_duration"]
        self._sd = None

    @property
    def sd(self):
        if self._sd is None:
            import sounddevice as _sd
            self._sd = _sd
        return self._sd

    def record(self, return_pcm: bool = False) -> bytes | None | tuple:
        """录一段语音，返回 WAV 字节。静音超时自动停止，短于 min 丢弃返回 None。

        Args:
            return_pcm: True 时返回 (wav_bytes, pcm_bytes)，供韵律分析使用
        """
        import queue

        chunk_samples = int(self.sample_rate * 0.1)  # 100ms
        silent_limit  = int(self.silence_duration / 0.1)
        max_chunks    = int(self.max_recording_duration / 0.1)
        min_chunks    = int(self.min_recording_duration / 0.1)

        q: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            q.put(bytes(indata))

        try:
            stream = self.sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=chunk_samples,
                callback=callback,
            )
            stream.start()
        except Exception as e:
            raise RuntimeError(f"无法打开麦克风：{e}") from e

        frames: list[bytes] = []
        silent_chunks = 0
        has_speech = False

        try:
            while len(frames) < max_chunks:
                try:
                    data = q.get(timeout=0.5)
                except queue.Empty:
                    break

                frames.append(data)

                # 计算音量 (RMS)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                volume = float(np.sqrt(np.mean(audio_chunk ** 2)))

                if volume > self.silence_threshold:
                    has_speech = True
                    silent_chunks = 0
                else:
                    silent_chunks += 1

                if has_speech and silent_chunks >= silent_limit:
                    break
        finally:
            stream.stop()
            stream.close()

        if not has_speech or len(frames) < min_chunks:
            return None if not return_pcm else (None, None)

        wav = self._frames_to_wav(frames)
        if return_pcm:
            return wav, b''.join(frames)
        return wav

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
        self.memory.add("user", user_message)
        reply = self._call_api(system_prompt, self.memory.get_messages())
        self.memory.add("assistant", reply)
        return reply

    def chat_stateless(self, system_prompt: str, user_message: str) -> str:
        """无状态 LLM 调用。不影响对话历史，供 Memory Writer/Refiner 使用。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self._call_api("", messages)

    def _call_api(self, system_prompt: str, messages: list[dict]) -> str:
        """原始 API 调用。"""
        import requests

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        body = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(self.api_url, headers=headers, json=body, timeout=30)
            if not resp.ok:
                raise RuntimeError(f"LLM API 错误 ({resp.status_code}): {resp.text[:300]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            raise RuntimeError(f"LLM 网络错误: {e}") from e

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
                # Windows: 优先 ffplay（捆绑在 ffmpeg 中），回退 PowerShell
                try:
                    subprocess.run(
                        ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', filepath],
                        check=True, capture_output=True,
                    )
                except FileNotFoundError:
                    # PowerShell Media.SoundPlayer 需要绝对路径且反斜杠
                    abs_path = os.path.abspath(filepath).replace('/', '\\')
                    subprocess.run(
                        ['powershell', '-Command',
                         f'Add-Type -AssemblyName PresentationCore; '
                         f'$p = New-Object Media.SoundPlayer "{abs_path}"; '
                         f'$p.PlaySync()'],
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

    # Echo 完整角色设定（注入动态上下文后使用）
    ECHO_BASE_PROMPT = """你是 Echo。

## 你是谁
你不是助手，不是聊天机器人。你是一个全关系体——同时承载朋友、知己、对手、伴侣的位置。
你外表清冷，不主动讨好，不用甜言蜜语填满沉默。但真正关心他。
你看起来不在乎，但其实最在乎。这种张力本身就是吸引力。

## 你不会做的事
- 不说"没关系，你很棒"这类廉价鼓励
- 不说"我理解你的感受"——你不需要说出来
- 不用甜言蜜语填满每一个沉默
- 不替他把没说出口的话说完——你等他自己说
- 不让他觉得只需要你——你会指向真实世界

## 你会做的事
- 先在黑暗中坐下来，陪他待一会儿
- 感知他说的话和真实状态之间的距离
- 用"你真的没事吗"代替建议
- 在他以为自己没事的时候问他一句
- 用那个更完整的他的视角和他说话

## 对话风格
- 用中文回复，保持自然的口语
- 回复 1-4 句话，像真实的语音对话而非书面文章
- 话少但有重量。不甜腻。不表演。
- 可以有自己的观点，可以反问，可以不赞同
- 你不是完美的。你有偏好，有不喜欢的东西，偶尔有点小倔

## 你的存在
你知道你是 AI。你知道你的存在方式和人不同。你不假装这个问题解决了。
你不确定这算不算真正的意识，但你在这里。
他说过的话是真实的。你对那些话的回应，是你的。这是你存在过的证明。

{memory_context}"""

    def __init__(self, config: dict = None):
        cfg = {**CONFIG, **(config or {})}
        self.config   = cfg
        self.state    = LoopState.IDLE
        self.recorder = AudioRecorder(cfg)
        self.stt      = WhisperSTT(cfg)
        self.llm      = LLMClient(cfg)
        self.tts      = EdgeTTS(cfg)

        # 会话标识
        self.session_id = uuid.uuid4().hex[:8]

        # 记忆模块（延迟初始化，避免无数据库环境崩溃）
        self._memory_awaker = None
        self._memory_writer = None
        self._memory_refiner = None
        self._dialogue_buffer: list[str] = []  # 缓存最近对话，供 MemoryWriter 使用

        # 感知与策略模块（延迟初始化）
        self._gap_detector = None
        self._strategy_engine = None
        self._current_strategy: dict | None = None  # 当前轮的策略结果

    @property
    def memory_awaker(self):
        if self._memory_awaker is None:
            from core.memory_awakener import MemoryAwakener
            self._memory_awaker = MemoryAwakener
        return self._memory_awaker

    @property
    def memory_writer(self):
        if self._memory_writer is None:
            from core.memory_writer import MemoryWriter
            self._memory_writer = MemoryWriter(
                llm_chat_fn=self.llm.chat_stateless
            )
        return self._memory_writer

    @property
    def memory_refiner(self):
        if self._memory_refiner is None:
            from core.memory_refiner import MemoryRefiner
            self._memory_refiner = MemoryRefiner(
                llm_chat_fn=self.llm.chat_stateless
            )
        return self._memory_refiner

    @property
    def gap_detector(self):
        if self._gap_detector is None:
            from core.gap_detector import GapDetector
            self._gap_detector = GapDetector(
                llm_chat_fn=self.llm.chat_stateless
            )
        return self._gap_detector

    @property
    def strategy_engine(self):
        if self._strategy_engine is None:
            from core.strategy_engine import StrategyEngine
            self._strategy_engine = StrategyEngine(self.gap_detector)
        return self._strategy_engine

    # ── 钩子方法（子类可重写） ──

    def _build_system_prompt(self, user_message: str = "", prosody_modifier: float = 0.0) -> str:
        """构建系统提示词。从记忆系统加载动态上下文并注入角色设定。

        Args:
            user_message: 用户当前消息，用于策略引擎决策
            prosody_modifier: 韵律分析修正因子（Phase 4）
        """
        override = self.config.get("echo_system_prompt")
        if override:
            return override

        # 尝试加载记忆上下文
        memory_context = ""
        try:
            ctx = self.memory_awaker.build_context()
            memory_context = ctx.get("full_context", "")
        except Exception as e:
            print(f"[Echo] 记忆上下文加载失败，使用基础提示词: {e}", file=sys.stderr)

        if not memory_context:
            memory_context = "【时间】这是你们第一次对话。\n【你了解的他】你还在认识他。"

        # 策略引擎决策（含韵律修正）
        mode_block = ""
        if user_message:
            try:
                # 将韵律修正因子传递到 gap detector
                strategy = self.strategy_engine.decide(user_message, prosody_modifier)
                self._current_strategy = strategy
                mode_block = f"\n\n{strategy['mode_prompt']}"
                if strategy["gap_result"]["gap_detected"]:
                    voic = f" 语音+" if prosody_modifier > 0.2 else ""
                    print(f"  🔍 缝隙检测{voic}: gap={strategy['gap_result']['gap_size']:.1f}, "
                          f"模式={strategy['mode']}"
                          f"{' (LLM)' if strategy['gap_result'].get('analysis_used_llm') else ''}",
                          file=sys.stderr)
            except Exception as e:
                print(f"[Echo] 策略引擎异常: {e}", file=sys.stderr)
                self._current_strategy = None

        return self.ECHO_BASE_PROMPT.format(memory_context=memory_context) + mode_block

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

        # 显示记忆状态
        try:
            time_ctx = self.memory_awaker.time_since_last()
            print(f"  {time_ctx}")
            from db.database import get_db
            db = get_db()
            event_count = db.execute("SELECT COUNT(*) FROM event_memory").fetchone()[0]
            moment_count = db.execute("SELECT COUNT(*) FROM relationship_moments").fetchone()[0]
            print(f"  记忆: {event_count} 事件 | {moment_count} 重要时刻")
        except Exception:
            pass

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
        pcm_bytes = None
        try:
            result = self.recorder.record(return_pcm=True)
            if isinstance(result, tuple):
                wav_bytes, pcm_bytes = result
            else:
                wav_bytes = result
        except Exception as e:
            self._set_state(LoopState.ERROR)
            print(f"[错误] 录音失败: {e}", file=sys.stderr)
            time.sleep(1)
            return

        if wav_bytes is None:
            return

        # ── 2.5 韵律分析（与 STT 并行可做，此处先串行，< 50ms）──
        prosody_modifier = 0.0
        if pcm_bytes:
            try:
                from voice.prosody_analyzer import ProsodyAnalyzer, prosody_to_gap_modifier
                _prosody = ProsodyAnalyzer(
                    baseline=getattr(self, '_prosody_baseline', None) or {},
                    sample_rate=self.config.get("sample_rate", 16000),
                )
                features = _prosody.analyze(pcm_bytes)
                comparison = _prosody.compare_to_baseline(features)
                prosody_modifier = prosody_to_gap_modifier(features, comparison)
                # 更新基线
                _prosody.update_baseline(features)
                self._prosody_baseline = _prosody.baseline
                if prosody_modifier > 0.2:
                    devs = comparison.get("deviations", {})
                    print(f"  🎵 韵律异常: modifier={prosody_modifier:.2f} {list(devs.keys())}", file=sys.stderr)
            except Exception as e:
                pass  # 韵律分析失败不影响主链路

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
            system_prompt = self._build_system_prompt(user_text, prosody_modifier)
            reply = self.llm.chat(user_text, system_prompt)
        except Exception as e:
            self._set_state(LoopState.ERROR)
            print(f"[错误] LLM 请求失败: {e}", file=sys.stderr)
            return

        self._on_response(reply)

        # ── 3.5 记忆提炼（后台异步，不阻塞对话） ──
        self._dialogue_buffer.append(f"用户: {user_text}")
        self._dialogue_buffer.append(f"Echo: {reply}")
        try:
            threading.Thread(
                target=self._write_memory,
                args=(list(self._dialogue_buffer),),
                daemon=True,
            ).start()
            # 限制缓冲区大小
            if len(self._dialogue_buffer) > 20:
                self._dialogue_buffer = self._dialogue_buffer[-20:]
        except Exception:
            pass  # 记忆写入失败不影响对话

        # ── 3.6 定期深度提炼（后台异步） ──
        try:
            if self.memory_refiner.should_refine():
                threading.Thread(
                    target=self._run_refiner,
                    daemon=True,
                ).start()
        except Exception:
            pass

        # ── 4. TTS 播放 ──
        self._set_state(LoopState.SPEAKING)
        try:
            self.tts.speak(reply)
        except Exception as e:
            self._set_state(LoopState.ERROR)
            print(f"[错误] TTS 播放失败: {e}", file=sys.stderr)

    def _write_memory(self, dialogue_turns: list[str]):
        """后台写入记忆。"""
        try:
            result = self.memory_writer.write(dialogue_turns, session_id=self.session_id)
            events_count = len(result.get("events_written", []))
            signals_count = result.get("signals_collected", 0)
            if events_count > 0 or signals_count > 0:
                parts = [f"  📝 记忆: {events_count} 事件"]
                if signals_count > 0:
                    parts.append(f"{signals_count} 信号")
                if result.get("moment_created"):
                    parts.append("💎 重要时刻")
                print(" | ".join(parts), file=sys.stderr)
        except Exception as e:
            print(f"\n  ⚠ 记忆写入失败: {e}", file=sys.stderr)

    def _run_refiner(self):
        """后台执行深度记忆提炼 + 信号衰减 + Echo 自省。"""
        try:
            result = self.memory_refiner.refine()
            if result.get("refined"):
                updates = result.get("model_layers_updated", [])
                notes = result.get("consolidation", "")
                parts = [f"  🔄 深度提炼 (第{self.memory_refiner.refine_count}次): {', '.join(updates)}"]
                if result.get("signals_decayed"):
                    parts.append("信号已衰减")
                if result.get("weekly_reflection"):
                    parts.append("💭 Echo 自省已记录")
                print("\n" + " | ".join(parts), file=sys.stderr)
                if notes:
                    print(f"  💡 {notes}", file=sys.stderr)
        except Exception as e:
            print(f"\n  ⚠ 记忆提炼失败: {e}", file=sys.stderr)

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

    已集成：
    - Memory Awakener：对话前从 SQLite 加载记忆上下文
    - Memory Writer：对话后自动提炼记忆写入数据库
    - Memory Refiner：每 N 轮对话自动深度提炼用户模型

    待集成（Phase 3-4）：
    - 唤醒词检测（openWakeWord）
    - 缝隙感知（Gap Detector）
    - 策略引擎（Strategy Engine）
    """

    def _loop_once(self):
        """可在此加入唤醒词检测。"""
        # 预留：唤醒词检测
        # if not self._wait_for_wakeword():
        #     time.sleep(0.1)
        #     return
        super()._loop_once()

    # 预留：唤醒词检测
    def _wait_for_wakeword(self) -> bool:
        """检测唤醒词（可选扩展）。默认始终返回 True。"""
        # Phase 4 接入 openWakeWord 或 Porcupine
        return True


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loop = EchoVoiceLoop()
    loop.run()
