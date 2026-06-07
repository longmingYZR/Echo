"""
Echo Voice Loop — 无麦克风调试版
==================================
用于在 PC 上快速验证 STT → LLM → TTS 链路，无需接入麦克风。
按编号选择测试模式。
"""

import os
import sys
from voice_loop import (
    CONFIG,
    LLMClient,
    EdgeTTS,
    ConversationMemory,
    EchoVoiceLoop,
)


def print_banner():
    print("""
╔══════════════════════════════════════╗
║     Echo Voice Loop · 调试模式      ║
╠══════════════════════════════════════╣
║  1 → 仅测试 LLM（验证 API 连通性）  ║
║  2 → 仅测试 TTS（验证音频播放）     ║
║  3 → 完整文字模式（输入 + 语音输出）║
║  0 → 退出                           ║
╚══════════════════════════════════════╝
""")


def test_llm():
    """仅测试 LLM：发送文字 → 打印回复（不播放语音）。含策略引擎。"""
    api_key = CONFIG["llm_api_key"]
    if not api_key:
        print("[错误] 请先设置 DEEPSEEK_API_KEY 环境变量")
        print("  export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx")
        return

    print("\n── LLM 连通性测试（含策略引擎）──")
    print(f"API: {CONFIG['llm_api_url']}")
    print(f"模型: {CONFIG['llm_model']}")

    llm = LLMClient()

    # 初始化感知与策略模块
    try:
        from core.gap_detector import GapDetector
        from core.strategy_engine import StrategyEngine
        from core.memory_awakener import MemoryAwakener
        gap_detector = GapDetector(llm_chat_fn=llm.chat_stateless)
        strategy_engine = StrategyEngine(gap_detector)
        has_strategy = True
        time_ctx = MemoryAwakener.time_since_last()
        print(f"  记忆上下文: 已加载 ({time_ctx})")
    except Exception:
        has_strategy = False

    # 构建基础 system prompt
    try:
        ctx = MemoryAwakener.build_context()
        base_prompt = EchoVoiceLoop.ECHO_BASE_PROMPT.format(
            memory_context=ctx.get("full_context", "")
        )
    except Exception:
        base_prompt = EchoVoiceLoop.DEFAULT_SYSTEM_PROMPT

    print("\n输入文字提问（输入 q 退出）：")
    while True:
        try:
            text = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() == 'q':
            break

        # 策略引擎决策（每轮动态）
        mode_block = ""
        if has_strategy:
            try:
                strategy = strategy_engine.decide(text)
                mode_block = f"\n\n{strategy['mode_prompt']}"
                gap = strategy['gap_result']
                if gap['gap_detected']:
                    print(f"  🔍 缝隙: {gap['gap_size']:.1f} | 模式: {strategy['mode']}"
                          f"{' (LLM)' if gap.get('analysis_used_llm') else ''}")
            except Exception:
                pass

        print("⏳ 思考中...", end='', flush=True)
        try:
            reply = llm.chat(text, base_prompt + mode_block)
            print(f"\r🤖 Echo: {reply}")
        except Exception as e:
            print(f"\r[错误] {e}")


def test_tts():
    """仅测试 TTS：输入文字 → 语音播放。"""
    print("\n── TTS 播放测试 ──")
    print(f"音色: {CONFIG['tts_voice']}")

    tts = EdgeTTS()

    test_texts = [
        "你好，我是Echo，你的AI伴侣。今天感觉怎么样？",
        "记得按时吃饭，照顾好自己。",
    ]

    for i, text in enumerate(test_texts):
        print(f"\n测试 {i+1}: \"{text}\"")
        print("🔊 播放中...", end='', flush=True)
        try:
            tts.speak(text)
            print("\r✅ 播放完成    ")
        except Exception as e:
            print(f"\r[错误] {e}")

    # 自定义输入
    print("\n── 自定义文字转语音 ──")
    print("输入文字播放语音（输入 q 退出）：")
    while True:
        try:
            text = input("\n📝 输入: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() == 'q':
            break

        print("🔊 播放中...", end='', flush=True)
        try:
            tts.speak(text)
            print("\r✅ 完成      ")
        except Exception as e:
            print(f"\r[错误] {e}")


def test_full_text_mode():
    """完整文字模式：键盘输入 → LLM → TTS 播放。含记忆系统。"""
    api_key = CONFIG["llm_api_key"]
    if not api_key:
        print("[错误] 请先设置 DEEPSEEK_API_KEY 环境变量")
        print("  export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx")
        return

    print("\n── 完整文字模式（含记忆系统）──")
    print("键盘输入 → DeepSeek LLM → 语音播放")
    print("每轮对话自动写入记忆数据库")
    print("输入 q 退出\n")

    llm = LLMClient()
    tts = EdgeTTS()

    # 初始化记忆 + 感知 + 策略系统
    try:
        from core.memory_awakener import MemoryAwakener
        from core.memory_writer import MemoryWriter
        from core.memory_refiner import MemoryRefiner
        from core.gap_detector import GapDetector
        from core.strategy_engine import StrategyEngine
        import uuid

        session_id = uuid.uuid4().hex[:8]
        memory_writer = MemoryWriter(llm_chat_fn=llm.chat_stateless)
        memory_refiner = MemoryRefiner(llm_chat_fn=llm.chat_stateless)
        gap_detector = GapDetector(llm_chat_fn=llm.chat_stateless)
        strategy_engine = StrategyEngine(gap_detector)
        dialogue_buffer: list[str] = []

        time_ctx = MemoryAwakener.time_since_last()
        print(f"  {time_ctx}")
        has_full_system = True
    except Exception as e:
        print(f"  ⚠ 完整系统不可用: {e}")
        has_full_system = False

    # 构建基础 system prompt（不含模式，模式每轮动态追加）
    if has_full_system:
        try:
            ctx = MemoryAwakener.build_context()
            base_prompt = EchoVoiceLoop.ECHO_BASE_PROMPT.format(
                memory_context=ctx.get("full_context", "")
            )
        except Exception:
            base_prompt = EchoVoiceLoop.DEFAULT_SYSTEM_PROMPT
    else:
        base_prompt = EchoVoiceLoop.DEFAULT_SYSTEM_PROMPT

    while True:
        try:
            text = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() == 'q':
            break

        # 每轮策略决策
        mode_block = ""
        if has_full_system:
            try:
                strategy = strategy_engine.decide(text)
                mode_block = f"\n\n{strategy['mode_prompt']}"
                gap = strategy['gap_result']
                if gap['gap_detected']:
                    print(f"  🔍 缝隙: {gap['gap_size']:.1f} | 模式: {strategy['mode']}"
                          f"{' (LLM)' if gap.get('analysis_used_llm') else ''}")
            except Exception:
                pass

        print("⏳ 思考中...", end='', flush=True)
        try:
            reply = llm.chat(text, base_prompt + mode_block)
            print(f"\r🤖 Echo: {reply}")
        except Exception as e:
            print(f"\r[错误] LLM: {e}")
            continue

        # 写入记忆
        if has_memory:
            dialogue_buffer.append(f"用户: {text}")
            dialogue_buffer.append(f"Echo: {reply}")
            try:
                result = memory_writer.write(list(dialogue_buffer), session_id=session_id)
                if result.get("events_written"):
                    print(f"  📝 记忆已更新", end='')
                if result.get("moment_created"):
                    print(f" | 💎 重要时刻已记录", end='')
                print()
                if len(dialogue_buffer) > 20:
                    dialogue_buffer = dialogue_buffer[-20:]
            except Exception as e:
                print(f"  ⚠ 记忆写入失败: {e}")

        # 定期提炼
        if has_memory:
            try:
                if memory_refiner.should_refine():
                    print("  🔄 触发记忆深度提炼...", end='', flush=True)
                    result = memory_refiner.refine()
                    if result.get("refined"):
                        print(f"\r  记忆深度提炼完成 ({', '.join(result.get('model_layers_updated', []))})        ")
                    else:
                        print(f"\r  提炼跳过: {result.get('reason', '')}                    ")
            except Exception as e:
                print(f"\n  ⚠ 记忆提炼失败: {e}")

        print("🔊 播放中...", end='', flush=True)
        try:
            tts.speak(reply)
            print("\r✅ 播放完成    ")
        except Exception as e:
            print(f"\r[错误] TTS: {e}")


def main():
    print_banner()

    while True:
        try:
            choice = input("请选择模式 [1/2/3/0]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == '1':
            test_llm()
            print_banner()
        elif choice == '2':
            test_tts()
            print_banner()
        elif choice == '3':
            test_full_text_mode()
            print_banner()
        elif choice == '0':
            print("再见 👋")
            break
        else:
            print("无效选项，请重新选择")


if __name__ == "__main__":
    main()
