import os

def get_dir_size(start_path='.'):
    """Calculate the total size of a directory including all subdirectories and files."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass # Ignore files that cannot be accessed
    return total_size

def format_size(size):
    """Format the size in bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0

def main():
    current_dir = input("请输入要检查的文件夹路径 (留空则默认为当前目录): ").strip()
    if not current_dir:
        current_dir = os.getcwd()
    elif not os.path.exists(current_dir):
        print("错误: 找不到指定的路径。")
        return
    elif not os.path.isdir(current_dir):
        print("错误: 指定的路径不是一个文件夹。")
        return
        
    print(f"\n正在扫描目录: {current_dir}\n")
    
    # Filter only directories within the current directory
    try:
        items = os.listdir(current_dir)
    except PermissionError:
        print("Permission denied to read the current directory.")
        return

    dirs = [item for item in items if os.path.isdir(os.path.join(current_dir, item))]
    
    if not dirs:
        print("No folders found in the current directory.")
        return
        
    print(f"{'Folder Name':<40} | {'Size'}")
    print("-" * 60)
    
    dir_info = []
    for d in dirs:
        dir_path = os.path.join(current_dir, d)
        size = get_dir_size(dir_path)
        dir_info.append((d, size))
        
    dir_info.sort(key=lambda x: x[1], reverse=True)
    
    for d, size in dir_info:
        print(f"{d:<40} | {format_size(size)}")

if __name__ == '__main__':
    main()
