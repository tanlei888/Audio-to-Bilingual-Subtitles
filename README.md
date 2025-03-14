# Audio-to-Bilingual-Subtitles

一个强大的音频转双语字幕工具，专门针对英语听力考试音频设计。该工具可以自动将英语音频文件转换为中英双语字幕，特别优化了对考试指令和专业术语的翻译。

## ✨ 特点

- 🎯 专为英语考试音频优化
- 🔄 自动识别并转录英语音频
- 🈶 智能生成中英双语字幕
- 🚀 支持GPU加速（如果可用）
- 📝 专业考试术语翻译优化
- 🎛️ 自动音频重采样
- 📊 实时处理进度显示
- 🔍 精确的时间戳控制
- 💾 支持多种音频格式（mp3, wav, m4a, ogg, flac）

## 🛠️ 安装要求

- Python 3.8 或更高版本
- CUDA支持（可选，用于GPU加速）
- FFmpeg（用于音频处理）
- 指尖AI API密钥（用于翻译服务）

## 📦 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/tanlei888/Audio-to-Bilingual-Subtitles.git
cd Audio-to-Bilingual-Subtitles
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置AI翻译服务：
   - 访问 [指尖AI官网](https://zhijianai.com.cn) 注册账号
   - 获取您的API密钥
   - 在 `audio_to_srt.py` 文件中找到以下代码并更新：
   ```python
   url = "https://api2.zhijianai.com.cn/v1/chat/completions"
   headers = {
       "Content-Type": "application/json",
       "Authorization": f"Bearer YOUR_API_KEY_HERE"  # 替换为您的API密钥
   }
   ```

5. 首次运行时，程序会自动下载Whisper模型（约140MB）：
   - 模型文件将保存在 `models` 目录下
   - 下载完成后会自动缓存，后续运行无需重新下载
   - 如果下载速度较慢，可以手动下载模型文件：
     1. 访问 https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt
     2. 将下载的 `base.pt` 文件放在 `models` 目录下

## 🚀 使用方法

1. 将音频文件放在程序同一目录下

2. 运行程序：
```bash
python audio_to_srt.py
```

3. 程序会自动：
   - 扫描当前目录下的所有支持的音频文件
   - 转换音频为WAV格式
   - 使用Whisper模型转录音频
   - 生成双语字幕文件（.srt格式）

## 📝 输出格式

生成的SRT文件格式如下：
```
0
00:00:00,620 --> 00:00:02,710
大学英语等级考试
College English test Band

1
00:00:02,920 --> 00:00:09,550
第二部分听力理解A部分指导
Four Part two Listening Comprehension Section A Directions.
```

## ⚙️ 配置说明

- 程序会自动检测并使用GPU（如果可用）
- 默认使用Whisper的"base"模型
- 音频会自动重采样为16kHz（Whisper模型要求）
- 翻译使用指尖AI接口，确保专业术语的准确性
- 可以根据需要调整翻译接口的temperature参数（当前设置为0.3）

## 📋 支持的音频格式

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- OGG (.ogg)
- FLAC (.flac)

## 🔧 故障排除

1. 如果遇到权限问题：
   - Windows：以管理员身份运行程序
   - Linux/Mac：确保有适当的文件权限

2. 如果遇到内存问题：
   - 尝试处理更短的音频文件
   - 确保系统有足够的可用内存

3. 如果翻译质量不理想：
   - 检查API密钥是否正确配置
   - 确保网络连接稳定
   - 尝试调整temperature参数
   - 确保音频质量清晰

## 📈 性能优化

- 使用GPU可显著提升处理速度
- 程序使用多线程处理翻译任务
- 自动缓存Whisper模型，加快后续使用

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

MIT License

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 提供强大的语音识别能力
- [librosa](https://librosa.org/) - 提供音频处理支持
- [指尖AI](https://zhijianai.com.cn) - 提供AI翻译支持 