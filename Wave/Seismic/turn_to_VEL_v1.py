import os
import numpy as np
import pandas as pd

# 可批量输入多个地震波 txt 文件（第一列时间，第二列加速度）
FILE_PATHS = [
    r'C:\Scripts\Wave\Seismic\El_Centro_scaled.txt',
]

# True: 自动扫描当前脚本目录下所有 txt（自动排除 *_scaled.txt 与 *_vel.txt）
# False: 使用 FILE_PATHS 中手动给出的路径
AUTO_SCAN_TXT = False


def load_ground_motion(file_path):
    """读取地震波数据：第一列时间，第二列加速度（单位 g）。"""
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['Time', 'Acceleration'])
    df = df[pd.to_numeric(df['Time'], errors='coerce').notna()]
    df = df[pd.to_numeric(df['Acceleration'], errors='coerce').notna()]

    t = df['Time'].astype(float).values
    accel = df['Acceleration'].astype(float).values

    if len(t) < 2:
        raise ValueError('地震动数据点不足，至少需要2个数据点。')

    # 保证时间递增并去除重复时间点
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    accel = accel[sort_idx]

    unique_mask = np.insert(np.diff(t) > 0, 0, True)
    t = t[unique_mask]
    accel = accel[unique_mask]

    if len(t) < 2:
        raise ValueError('时间序列存在重复或无效点，无法积分。')

    return t, accel


def get_input_files():
    """获取待处理的 txt 文件列表。"""
    if not AUTO_SCAN_TXT:
        if not FILE_PATHS:
            raise ValueError('AUTO_SCAN_TXT=False 时，FILE_PATHS 不能为空。')
        return FILE_PATHS

    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_files = []

    for name in os.listdir(script_dir):
        full_path = os.path.join(script_dir, name)
        if not os.path.isfile(full_path):
            continue

        stem, ext = os.path.splitext(name)
        if ext.lower() != '.txt':
            continue

        stem_lower = stem.lower()
        if '_scaled' in stem_lower or '_vel' in stem_lower:
            continue

        txt_files.append(full_path)

    txt_files.sort()
    if not txt_files:
        raise ValueError(f'目录中未找到可处理的 txt 文件：{script_dir}')

    print(f'自动扫描目录：{script_dir}')
    print(f'共找到 {len(txt_files)} 个待处理 txt 文件。')
    return txt_files


def accel_to_velocity(t, accel_g):
    """将加速度时程积分为速度时程。输入加速度单位为 g，输出速度单位为 m/s。"""
    g = 9.81
    accel_ms2 = accel_g * g

    dt = np.diff(t)
    v = np.zeros_like(accel_ms2)
    v[1:] = np.cumsum(0.5 * (accel_ms2[1:] + accel_ms2[:-1]) * dt)
    return v


def main():
    file_paths = get_input_files()

    for file_path in file_paths:
        print(f'\n开始处理：{file_path}')
        if not os.path.exists(file_path):
            print(f'跳过：找不到文件 {file_path}')
            continue

        try:
            t, accel = load_ground_motion(file_path)
            vel = accel_to_velocity(t, accel)

            base, ext = os.path.splitext(file_path)
            vel_file_path = base + '_vel' + ext
            pd.DataFrame({'Time': t, 'Velocity': vel}).to_csv(
                vel_file_path,
                sep='\t',
                index=False,
                header=False,
                float_format='%.8f',
            )
            print(f'已保存时间-速度文件：{vel_file_path}')

        except Exception as exc:
            print(f'处理失败：{file_path} -> {exc}')


if __name__ == '__main__':
    main()