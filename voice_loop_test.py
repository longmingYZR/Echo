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
    """仅测试 LLM：发送文字 → 打印回复（不播放语音）。"""
    api_key = CONFIG["llm_api_key"]
    if not api_key:
        print("[错误] 请先设置 DEEPSEEK_API_KEY 环境变量")
        print("  export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx")
        return

    print("\n── LLM 连通性测试 ──")
    print(f"API: {CONFIG['llm_api_url']}")
    print(f"模型: {CONFIG['llm_model']}")

    llm = LLMClient()
    system_prompt = EchoVoiceLoop.DEFAULT_SYSTEM_PROMPT

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

        print("⏳ 思考中...", end='', flush=True)
        try:
            reply = llm.chat(text, system_prompt)
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
    """完整文字模式：键盘输入 → LLM → TTS 播放。"""
    api_key = CONFIG["llm_api_key"]
    if not api_key:
        print("[错误] 请先设置 DEEPSEEK_API_KEY 环境变量")
        print("  export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx")
        return

    print("\n── 完整文字模式 ──")
    print("键盘输入 → DeepSeek LLM → 语音播放")
    print("输入 q 退出\n")

    llm = LLMClient()
    tts = EdgeTTS()
    system_prompt = EchoVoiceLoop.DEFAULT_SYSTEM_PROMPT

    while True:
        try:
            text = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() == 'q':
            break

        print("⏳ 思考中...", end='', flush=True)
        try:
            reply = llm.chat(text, system_prompt)
            print(f"\r🤖 Echo: {reply}")
        except Exception as e:
            print(f"\r[错误] LLM: {e}")
            continue

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
