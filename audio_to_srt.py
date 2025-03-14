import whisper
from googletrans import Translator
import datetime
import os
import glob
import time
from tqdm import tqdm
import shutil
import tempfile
import uuid
import soundfile as sf
import librosa
import logging
import sys
import torch
import wave
import ctypes
import platform
import requests
import json
import concurrent.futures

def is_admin():
    """检查程序是否以管理员权限运行"""
    try:
        if platform.system() == 'Windows':
            return ctypes.windll.shell32.IsUserAnAdmin()
        else:
            return os.getuid() == 0  # Unix-like systems
    except:
        return False

def check_file_permissions(file_path, check_write=False):
    """检查文件权限
    
    Args:
        file_path: 要检查的文件路径
        check_write: 是否检查写入权限
    
    Returns:
        (bool, str): (是否有权限, 错误信息)
    """
    try:
        if not os.path.exists(file_path):
            return False, "文件不存在"
            
        # 检查读取权限
        if not os.access(file_path, os.R_OK):
            return False, "没有读取权限"
            
        # 如果需要检查写入权限
        if check_write and not os.access(file_path, os.W_OK):
            return False, "没有写入权限"
            
        # 尝试打开文件
        if check_write:
            mode = 'r+'
        else:
            mode = 'r'
            
        with open(file_path, mode) as f:
            pass
            
        return True, "权限检查通过"
    except PermissionError as e:
        return False, f"权限错误: {str(e)}"
    except Exception as e:
        return False, f"其他错误: {str(e)}"

def ensure_directory_permissions(directory):
    """确保目录具有正确的权限"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # 检查目录权限
        if not os.access(directory, os.R_OK | os.W_OK | os.X_OK):
            raise PermissionError(f"没有足够的目录权限: {directory}")
            
        # 尝试创建测试文件
        test_file = os.path.join(directory, 'test_permissions.tmp')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            raise PermissionError(f"无法在目录中创建文件: {str(e)}")
            
        return True
    except Exception as e:
        logging.error(f"目录权限检查失败: {str(e)}")
        return False

# 设置日志记录
try:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'audio_processing.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8', mode='a')
        ]
    )
except Exception as e:
    print(f"警告: 无法创建日志文件，将只输出到控制台: {str(e)}")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# 创建工作目录
try:
    WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work_dir')
    if not ensure_directory_permissions(WORK_DIR):
        alternative_work_dir = os.path.join(os.path.expanduser('~'), 'audio_work_dir')
        logging.warning(f"无法使用默认工作目录，尝试使用替代目录: {alternative_work_dir}")
        if ensure_directory_permissions(alternative_work_dir):
            WORK_DIR = alternative_work_dir
        else:
            raise PermissionError("无法创建具有适当权限的工作目录")
except Exception as e:
    logging.error(f"初始化工作目录失败: {str(e)}")
    sys.exit(1)

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    td = datetime.timedelta(seconds=float(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def format_time(seconds):
    """Format seconds into hours, minutes, seconds"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}小时{minutes}分{seconds}秒"
    elif minutes > 0:
        return f"{minutes}分{seconds}秒"
    else:
        return f"{seconds}秒"

def get_common_translations():
    """返回常见短语的映射表"""
    return {
        # 考试相关
        "college english test": "大学英语考试",
        "band four": "四级",
        "band six": "六级",
        "cet-4": "大学英语四级",
        "cet-6": "大学英语六级",
        "part one": "第一部分",
        "part two": "第二部分",
        "part three": "第三部分",
        "part four": "第四部分",
        "section a": "A部分",
        "section b": "B部分",
        "section c": "C部分",
        
        # 考试指令
        "directions": "指导语",
        "in this section": "在本部分中",
        "you will hear": "你将听到",
        "you'll hear": "你将听到",
        "at the end of": "在...结束时",
        "questions": "问题",
        "please note": "请注意",
        "please remember": "请记住",
        
        # 听力类型
        "news reports": "新闻报道",
        "conversation": "对话",
        "passage": "短文",
        "lecture": "讲座",
        "report": "报道",
        
        # 常见指令
        "answer the questions": "回答问题",
        "choose the best answer": "选择最佳答案",
        "according to": "根据",
        "what you have just heard": "你刚才听到的内容",
        "both the conversation and the questions": "对话和问题都",
        "will be spoken only once": "只播放一遍",
        "mark the corresponding letter": "标记相应的字母",
        "on answer sheet": "在答题纸上",
        
        # 结束语
        "that's the end of": "以上就是",
        "listening comprehension": "听力理解",
        "that is the end of": "以上就是",
        "this is the end of": "以上就是",
        
        # 数字和时间
        "one": "一",
        "two": "二",
        "three": "三",
        "four": "四",
        "five": "五",
        "six": "六",
        "first": "第一",
        "second": "第二",
        "third": "第三",
        "fourth": "第四",
        
        # 常见连接词
        "however": "然而",
        "therefore": "因此",
        "moreover": "此外",
        "in addition": "另外",
        "for example": "例如",
        "such as": "比如",
        
        # 考试场景
        "the following": "下列",
        "based on": "基于",
        "you have just heard": "你刚刚听到的",
        "with a single line through the center": "中间有一条线穿过",
        "after you hear": "当你听到",
        "from the four choices marked": "从标有",
        "and": "和",
        "then": "然后",
    }

def optimize_translation(english_text):
    """优化翻译结果"""
    if not english_text:
        return ""
        
    # 转换为小写以进行匹配
    lower_text = english_text.lower().strip()
    
    # 获取常见翻译映射
    common_translations = get_common_translations()
    
    # 检查完整短语匹配
    if lower_text in common_translations:
        return common_translations[lower_text]
    
    # 分词处理
    words = lower_text.split()
    result = []
    i = 0
    
    while i < len(words):
        matched = False
        # 尝试最长匹配
        for j in range(len(words), i, -1):
            phrase = " ".join(words[i:j])
            if phrase in common_translations:
                result.append(common_translations[phrase])
                i = j
                matched = True
                break
        
        if not matched:
            # 如果没有匹配到，尝试使用Google翻译
            try:
                translator = Translator()
                # 翻译单个词或短语
                current_phrase = words[i]
                translation = translator.translate(current_phrase, dest='zh-cn').text
                result.append(translation)
                i += 1
            except Exception as e:
                logging.error(f"翻译失败: {e}")
                # 如果翻译失败，保留原文
                result.append(words[i])
                i += 1
    
    # 优化结果
    final_text = "".join(result)
    
    # 后处理规则
    replacements = {
        "的的": "的",
        "了的": "了",
        "地的": "地",
        "和的": "和",
    }
    
    for old, new in replacements.items():
        final_text = final_text.replace(old, new)
    
    return final_text

def translate_with_ai(text):
    """使用 AI 接口翻译文本"""
    url = "https://example.com.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer sk-gdmFNgzMGvpYXLm19f7bC1Eb29D54aA49d97BfE866C9998a"
    }
    
    # 构建提示词，要求 AI 进行准确的翻译
    prompt = f"""请将以下英文文本翻译成中文。要求：
1. 保持原文的语气和语境
2. 确保翻译准确、自然、符合中文表达习惯
3. 如果是考试指令，使用标准的考试用语
4. 直接返回翻译结果，不要包含任何解释或额外内容

英文文本：{text}"""

    data = {
        "model": "glm-4-flash",
        "messages": [
            {"role": "system", "content": "你是一个专业的英语翻译专家，专门负责考试音频的翻译工作。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,  # 降低随机性，保持翻译一致性
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        # 提取翻译结果
        translation = result['choices'][0]['message']['content'].strip()
        return translation
    except Exception as e:
        logging.error(f"AI 翻译请求失败: {str(e)}")
        return None

def batch_translate_segments(segments):
    """并行批量翻译字幕段落"""
    def translate_segment(segment):
        text = segment['text'].strip()
        translation = translate_with_ai(text)
        return segment, translation
    
    # 使用线程池并行处理翻译
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有翻译任务
        future_to_segment = {executor.submit(translate_segment, segment): segment 
                           for segment in segments}
        
        # 收集结果
        results = []
        for future in concurrent.futures.as_completed(future_to_segment):
            try:
                segment, translation = future.result()
                if translation:
                    results.append((segment, translation))
                else:
                    # 如果翻译失败，使用原文
                    results.append((segment, "[翻译失败]"))
            except Exception as e:
                logging.error(f"处理翻译结果时出错: {str(e)}")
                segment = future_to_segment[future]
                results.append((segment, "[翻译错误]"))
    
    # 按原始顺序排序结果
    return sorted(results, key=lambda x: segments.index(x[0]))

def create_srt(segments, output_file):
    """Create SRT file from segments with Chinese text above English text, starting index from 0.
    Only include timestamps where there is actual speech, preserving silence gaps between segments."""
    total_segments = len(segments)
    print(f"\n开始批量翻译 {total_segments} 个字幕段落...")
    
    # 批量翻译所有段落
    translated_segments = batch_translate_segments(segments)
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (segment, chinese_text) in enumerate(translated_segments):
            # 显示进度
            progress = (i + 1) / total_segments * 100
            print(f"\r写入字幕: {progress:.1f}% ({i + 1}/{total_segments})", end="", flush=True)
            
            # 获取实际的开始和结束时间
            start_time = segment['start']
            end_time = segment['end']
            
            # 如果这段文本是空的或者只包含空白字符，跳过这个片段
            if not segment['text'].strip():
                continue
                
            # 写入SRT条目，序号从0开始
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
            f.write(f"{chinese_text}\n")  # 中文在上
            f.write(f"{segment['text'].strip()}\n\n")  # 英文在下
    
    print("\n字幕文件创建完成！")

def verify_wav_file(file_path):
    """验证WAV文件的完整性"""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # 获取WAV文件的基本信息
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            
            # 计算预期的文件大小
            expected_size = frames * channels * sample_width
            actual_size = os.path.getsize(file_path)
            
            logging.info(f"WAV文件信息:")
            logging.info(f"- 声道数: {channels}")
            logging.info(f"- 采样宽度: {sample_width}")
            logging.info(f"- 采样率: {frame_rate}")
            logging.info(f"- 总帧数: {frames}")
            logging.info(f"- 预期数据大小: {expected_size}")
            logging.info(f"- 实际文件大小: {actual_size}")
            
            # 读取一些数据来验证文件可读性
            wav_file.readframes(min(1000, frames))
            
            return True
    except Exception as e:
        logging.error(f"WAV文件验证失败: {str(e)}")
        return False

def convert_audio_to_wav(input_path, output_path):
    """Convert audio file to WAV format"""
    try:
        logging.info(f"加载音频文件: {input_path}")
        logging.info(f"文件大小: {os.path.getsize(input_path)/1024/1024:.2f} MB")
        
        # 使用librosa加载音频文件
        y, sr = librosa.load(input_path, sr=None)
        
        # 确保输出路径使用正确的路径分隔符
        output_path = os.path.normpath(output_path)
        logging.info(f"转换为WAV格式: {output_path}")
        
        # 保存为WAV格式
        sf.write(output_path, y, sr)
        
        # 验证文件是否成功创建
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logging.info(f"WAV文件创建成功，大小: {file_size/1024/1024:.2f} MB")
            
            # 验证WAV文件的完整性
            if verify_wav_file(output_path):
                logging.info("WAV文件验证通过")
                return True
            else:
                logging.error("WAV文件验证失败")
                return False
        else:
            logging.error(f"WAV文件创建失败: {output_path}")
            return False
    except Exception as e:
        logging.error(f"音频转换失败: {str(e)}")
        return False

def copy_to_work_dir(file_path):
    """将文件复制到工作目录，使用ASCII文件名"""
    try:
        # 生成唯一的ASCII文件名
        temp_name = f'audio_{uuid.uuid4().hex[:8]}.wav'
        temp_file = os.path.join(WORK_DIR, temp_name)
        temp_file = os.path.normpath(temp_file)
        
        # 转换并复制文件
        logging.info(f"开始转换音频文件...")
        if not convert_audio_to_wav(file_path, temp_file):
            raise Exception("音频文件转换失败")
        
        # 再次验证文件
        if not os.path.exists(temp_file):
            raise FileNotFoundError(f"转换后的文件不存在: {temp_file}")
        
        if not verify_wav_file(temp_file):
            raise Exception("WAV文件验证失败")
        
        return temp_file
    except Exception as e:
        logging.error(f"复制文件到工作目录失败: {str(e)}")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logging.info("清理失败的临时文件")
            except:
                pass
        raise

def process_audio(audio_path, output_path, total_files, current_file_index):
    """Process audio file and create bilingual SRT"""
    abs_audio_path = os.path.abspath(audio_path)
    abs_audio_path = os.path.normpath(abs_audio_path)
    
    # 检查输入文件权限
    has_permission, error_msg = check_file_permissions(abs_audio_path)
    if not has_permission:
        raise PermissionError(f"无法访问输入文件: {error_msg}")
    
    # 检查输出目录权限
    output_dir = os.path.dirname(output_path)
    if not ensure_directory_permissions(output_dir):
        raise PermissionError(f"无法写入输出目录: {output_dir}")
    
    if not os.path.exists(abs_audio_path):
        raise FileNotFoundError(f"找不到音频文件: {abs_audio_path}")
    
    if not os.path.isfile(abs_audio_path):
        raise ValueError(f"指定的路径不是文件: {abs_audio_path}")
        
    start_time = time.time()
    logging.info(f"\n处理音频文件 [{current_file_index}/{total_files}]: {os.path.basename(audio_path)}")
    
    work_file = None
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 复制到工作目录
        print(f"\n1. 准备音频文件 [{os.path.basename(audio_path)}]...")
        work_file = copy_to_work_dir(abs_audio_path)
        print("✓ 音频文件准备完成")
        
        # 确保文件存在且可访问
        if not os.path.exists(work_file):
            raise FileNotFoundError(f"工作文件不存在: {work_file}")
            
        # 验证文件大小
        work_file_size = os.path.getsize(work_file)
        logging.info(f"工作文件大小: {work_file_size/1024/1024:.2f} MB")
        
        try:
            print(f"\n2. 加载 Whisper 模型 (使用 {device})...")
            # 使用更快的模型
            model = whisper.load_model("base", device=device, download_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
            print("✓ 模型加载完成")
            
            # 使用绝对路径进行转录
            print("\n3. 开始音频转录...")
            
            # 确保文件存在并可读
            if not os.path.exists(work_file):
                raise FileNotFoundError(f"转录前文件不存在: {work_file}")
                
            # 使用 librosa 加载音频文件，并显示进度
            try:
                print("   正在加载音频...")
                # 使用 librosa 加载音频，显示进度条
                with tqdm(total=100, desc="加载音频", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                    y, sr = librosa.load(work_file, sr=16000)  # Whisper 期望采样率为 16kHz
                    pbar.update(100)
                
                if y is None or len(y) == 0:
                    raise ValueError("音频加载失败")
                    
                print(f"   ✓ 音频加载成功 (采样率: {sr}Hz, 时长: {len(y)/sr:.1f}秒)")
                
                # 执行转录，显示进度
                print("\n4. 正在转录音频...")
                # 计算音频时长
                duration = len(y) / sr
                chunk_duration = 30  # 每30秒显示一次进度
                num_chunks = int(duration / chunk_duration) + 1
                
                with tqdm(total=100, desc="转录进度", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                    result = model.transcribe(
                        y,
                        task="transcribe",
                        language="en",
                        fp16=False if device == "cpu" else True,
                        verbose=False
                    )
                    pbar.update(100)
                
                print("✓ 音频转录完成")
                
                print("\n5. 创建双语字幕文件...")
                
                # 使用绝对路径创建输出文件
                abs_output_path = os.path.abspath(output_path)
                create_srt(result["segments"], abs_output_path)
                
                if not os.path.exists(abs_output_path):
                    raise FileNotFoundError(f"SRT文件创建失败: {abs_output_path}")
                    
            except Exception as e:
                logging.error(f"音频处理过程中出错: {str(e)}")
                raise
                
        except Exception as e:
            logging.error(f"转录或创建SRT文件时出错: {str(e)}")
            raise
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n✓ 字幕文件创建成功: {output_path}")
        print(f"✓ 处理用时: {format_time(processing_time)}")
        
        return processing_time
    except Exception as e:
        logging.error(f"处理文件时出错: {str(e)}")
        logging.error(f"文件路径: {abs_audio_path}")
        logging.error(f"工作文件路径: {work_file}")
        if work_file and os.path.exists(work_file):
            logging.error(f"工作文件存在: 是")
            logging.error(f"工作文件大小: {os.path.getsize(work_file)/1024/1024:.2f} MB")
            try:
                # 尝试读取文件的前几个字节来验证文件完整性
                with open(work_file, 'rb') as f:
                    header = f.read(44)  # WAV header is 44 bytes
                logging.error(f"WAV文件头大小: {len(header)} bytes")
            except Exception as e:
                logging.error(f"读取WAV文件头失败: {e}")
        else:
            logging.error("工作文件不存在")
        raise
    finally:
        # 清理工作文件
        if work_file and os.path.exists(work_file):
            try:
                os.remove(work_file)
                logging.info("工作文件已清理")
            except Exception as e:
                logging.error(f"清理工作文件时出错: {e}")

def get_audio_files():
    """获取当前目录下的所有音频文件"""
    audio_extensions = ['*.mp3', '*.wav', '*.m4a', '*.ogg', '*.flac']
    audio_files = []
    
    current_dir = os.getcwd()
    logging.info(f"搜索目录: {current_dir}")
    
    for ext in audio_extensions:
        found_files = glob.glob(ext)
        if found_files:
            logging.info(f"找到 {ext} 文件: {len(found_files)} 个")
            for file in found_files:
                abs_path = os.path.abspath(file)
                abs_path = os.path.normpath(abs_path)
                if os.path.exists(abs_path):
                    audio_files.append(abs_path)
                else:
                    logging.warning(f"警告: 文件不存在: {abs_path}")
    
    return audio_files

def cleanup_work_dir():
    """清理工作目录"""
    if os.path.exists(WORK_DIR):
        try:
            shutil.rmtree(WORK_DIR)
            os.makedirs(WORK_DIR)
            logging.info("工作目录已清理并重新创建")
        except Exception as e:
            logging.error(f"清理工作目录时出错: {e}")

def main():
    print("音频转双语字幕工具启动...")
    
    # 检查是否具有管理员权限
    if not is_admin():
        logging.warning("程序没有以管理员权限运行，可能会遇到权限问题")
        print("\n警告: 建议以管理员权限运行此程序以避免权限问题")
        print("您可以：")
        print("1. 关闭程序，右键点击程序，选择'以管理员身份运行'")
        print("2. 继续以当前权限运行（可能会遇到问题）")
        choice = input("\n是否继续运行？(y/n): ")
        if choice.lower() != 'y':
            print("程序退出")
            sys.exit(0)
    
    # 创建模型缓存目录
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    logging.info(f"当前工作目录: {os.getcwd()}\n")
    
    # 检查工作目录权限
    if not ensure_directory_permissions(WORK_DIR):
        print("错误: 无法访问工作目录，请检查权限设置")
        input("按回车键退出...")
        sys.exit(1)
    
    # 清理并重新创建工作目录
    cleanup_work_dir()
    
    # 获取当前目录下的所有音频文件
    try:
        audio_files = get_audio_files()
    except PermissionError as e:
        print(f"错误: 无法访问音频文件: {str(e)}")
        print("请确保您有足够的权限访问这些文件")
        input("按回车键退出...")
        return
    
    if not audio_files:
        print("当前目录下没有找到音频文件！")
        print("支持的音频格式：mp3, wav, m4a, ogg, flac")
        print("请确保音频文件与程序在同一目录下。")
        input("按回车键退出...")
        return
    
    total_files = len(audio_files)
    print(f"\n找到 {total_files} 个音频文件：")
    for i, file in enumerate(audio_files, 1):
        file_size = os.path.getsize(file) / (1024 * 1024)  # Convert to MB
        print(f"{i}. {os.path.basename(file)} ({file_size:.2f} MB)")
    
    print("\n开始处理所有音频文件...")
    
    # 记录总处理时间
    total_start_time = time.time()
    processing_times = []
    successful_files = 0
    
    # 处理每个音频文件
    for i, audio_file in enumerate(audio_files, 1):
        # 生成输出文件名（将音频文件扩展名改为.srt）
        output_file = os.path.splitext(audio_file)[0] + '.srt'
        
        try:
            processing_time = process_audio(audio_file, output_file, total_files, i)
            processing_times.append(processing_time)
            successful_files += 1
            
            # 计算预计剩余时间
            if i < total_files:
                avg_time = sum(processing_times) / len(processing_times)
                remaining_files = total_files - i
                estimated_remaining = avg_time * remaining_files
                print(f"预计剩余时间: {format_time(estimated_remaining)}")
                
        except Exception as e:
            logging.error(f"处理文件 {os.path.basename(audio_file)} 时出错: {str(e)}")
            continue
    
    total_time = time.time() - total_start_time
    print(f"\n处理完成！")
    print(f"成功处理文件数: {successful_files}/{total_files}")
    print(f"总处理时间: {format_time(total_time)}")
    if successful_files > 0:
        print(f"平均每个文件处理时间: {format_time(total_time/successful_files)}")
    
    # 最后清理工作目录
    cleanup_work_dir()
    
    input("按回车键退出...")

if __name__ == "__main__":
    main() 