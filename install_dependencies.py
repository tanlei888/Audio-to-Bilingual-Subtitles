import subprocess
import sys
import os

def install_dependencies():
    print("开始安装依赖...")
    
    # 检查是否已安装pip
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        print("错误: 未找到pip。请确保您已安装Python和pip。")
        return False
    
    # 首先升级pip
    try:
        print("升级 pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    except subprocess.CalledProcessError:
        print("警告: 无法升级pip，继续安装依赖...")
    
    # 安装依赖
    try:
        print("安装 openai-whisper...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper==20240930"])
        
        print("安装 googletrans...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "googletrans==3.1.0a0"])
        
        print("安装 pydub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub==0.25.1"])
        
        print("安装 tqdm...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        
        print("\n所有依赖安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"安装依赖时出错: {e}")
        print("\n如果安装失败，请尝试以下步骤：")
        print("1. 升级pip: python -m pip install --upgrade pip")
        print("2. 手动安装依赖：")
        print("   pip install openai-whisper==20240930")
        print("   pip install googletrans==3.1.0a0")
        print("   pip install pydub==0.25.1")
        print("   pip install tqdm")
        return False

if __name__ == "__main__":
    print("音频转双语SRT字幕工具 - 依赖安装程序")
    print("====================================\n")
    
    success = install_dependencies()
    
    if success:
        print("\n现在您可以运行 audio_to_srt.py 来处理音频文件了。")
    else:
        print("\n依赖安装失败。请按照上述步骤手动安装依赖。")
    
    input("\n按回车键退出...") 