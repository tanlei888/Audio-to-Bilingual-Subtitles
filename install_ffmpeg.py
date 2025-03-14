import os
import subprocess
import sys
import urllib.request
import zipfile
import tempfile
import shutil
import winreg

def download_file(url, filename):
    """下载文件并显示进度"""
    print(f"下载 {filename}...")
    urllib.request.urlretrieve(url, filename)

def add_to_path(new_path):
    """添加路径到系统环境变量"""
    # 打开注册表
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 0, winreg.KEY_ALL_ACCESS) as key:
        # 获取当前PATH值
        try:
            current_path = winreg.QueryValueEx(key, 'Path')[0]
        except WindowsError:
            current_path = ''
        
        # 检查路径是否已经在PATH中
        if new_path.lower() not in current_path.lower():
            # 添加新路径
            new_path_value = current_path + ';' + new_path if current_path else new_path
            winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path_value)
            
            # 通知系统环境变量已更改
            subprocess.run(['setx', 'PATH', new_path_value], capture_output=True)
            
            print(f"已将 {new_path} 添加到系统PATH")
            return True
    return False

def main():
    print("FFmpeg 自动安装程序")
    print("===================\n")
    
    # 检查是否已安装
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        print("FFmpeg 已经安装！")
        input("按回车键退出...")
        return
    except:
        print("未检测到 FFmpeg，开始安装...\n")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "ffmpeg.zip")
    
    try:
        # 下载 FFmpeg
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        print("下载 FFmpeg...")
        download_file(url, zip_path)
        
        # 解压文件
        print("\n解压文件...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # 获取解压后的目录名
        ffmpeg_dir = None
        for item in os.listdir(temp_dir):
            if os.path.isdir(os.path.join(temp_dir, item)) and 'ffmpeg' in item.lower():
                ffmpeg_dir = os.path.join(temp_dir, item)
                break
        
        if not ffmpeg_dir:
            raise Exception("无法找到 FFmpeg 目录")
        
        # 创建目标目录
        program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
        target_dir = os.path.join(program_files, 'ffmpeg')
        
        # 如果目标目录已存在，先删除
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        # 移动文件到目标目录
        print("安装 FFmpeg...")
        bin_dir = os.path.join(ffmpeg_dir, 'bin')
        shutil.copytree(bin_dir, target_dir)
        
        # 添加到 PATH
        print("配置环境变量...")
        if add_to_path(target_dir):
            print("\nFFmpeg 安装成功！")
            print(f"安装位置: {target_dir}")
            print("\n请注意：")
            print("1. 您需要重新启动电脑来使环境变量生效")
            print("2. 或者注销并重新登录也可以")
            print("3. 重启后，打开新的命令提示符，输入 ffmpeg -version 测试安装")
        else:
            print("\n警告：环境变量配置可能未成功，请手动添加以下路径到系统PATH：")
            print(target_dir)
        
    except Exception as e:
        print(f"安装过程中出错: {str(e)}")
    finally:
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    # 检查管理员权限
    try:
        is_admin = os.getuid() == 0
    except AttributeError:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    
    if not is_admin:
        print("请以管理员权限运行此程序！")
        print("请右键点击此程序，选择'以管理员身份运行'")
        input("按回车键退出...")
        sys.exit(1)
    
    main() 