# Echo — AI 语音伴侣

双语伴 AI 项目，由两个层次组成：

| 层 | 文件 | 说明 |
|------|------|------|
| **Web 交互层** | `index.html` + `memory.js` | 浏览器端语音伴侣，唤醒动画、记忆推荐、意图识别 |
| **硬件语音层** | `voice_loop.py` | Python 语音链路：VAD → Whisper STT → LLM → TTS |

## 快速开始

### Web 端

浏览器打开 `index.html`（需要 Chrome 的 Web Speech API），空格键唤醒。

### 语音链路

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx

# 3. 先文字模式验证（无需麦克风）
python voice_loop_test.py

# 4. 接入麦克风跑完整版
python voice_loop.py
```

## 项目结构

```
echo/
├── index.html              # Web 伴侣界面（语音交互 + 记忆推荐）
├── memory.js               # 记忆引擎（localStorage，时间/关键词推荐）
├── voice_loop.py           # Python 语音链路核心框架
├── voice_loop_test.py      # 无麦克风调试版
├── requirements.txt        # Python 依赖
└── README.md
```

## 树莓派部署

推荐硬件：树莓派 Zero 2W / 4B + ReSpeaker 2-Mic HAT

```bash
# 系统依赖
sudo apt install mpg123 portaudio19-dev

# 调整配置
# whisper_model: tiny (Zero 2W) / base (4B)
# silence_threshold: 400 (ReSpeaker)

# 开机自启 → 见 systemd 配置
```

## 对接 Echo 现有系统

```python
from voice_loop import EchoVoiceLoop

class MyEchoVoice(EchoVoiceLoop):
    def _build_system_prompt(self) -> str:
        from echo.character import get_system_prompt
        return get_system_prompt()

loop = MyEchoVoice()
loop.run()
```
