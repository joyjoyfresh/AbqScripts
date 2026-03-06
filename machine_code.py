# -*- coding: utf-8 -*-
"""基于 MAC 地址生成机器码（指定物理网卡，结果稳定）"""
import hashlib
import subprocess
import re


def get_machine_code(adapter_keyword='Intel'):
    """
    获取指定网卡的 MAC 地址并生成机器码

    参数:
        adapter_keyword: 网卡名称关键字，用于匹配目标网卡
    """
    output = subprocess.check_output('getmac /v /fo csv', shell=True).decode('gbk', errors='replace')
    lines = output.strip().split('\n')

    for line in lines[1:]:
        if adapter_keyword in line:
            mac_match = re.search(r'([0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2})', line)
            if mac_match:
                mac_str = mac_match.group(1).upper().replace('-', ':')
                machine_code = hashlib.md5(mac_str.encode()).hexdigest().upper()
                return mac_str, machine_code

    raise RuntimeError('未找到包含 "{}" 的网卡'.format(adapter_keyword))


if __name__ == '__main__':
    mac_addr, code = get_machine_code('Intel')
    print('MAC 地址:  {}'.format(mac_addr))
    print('机器码:    {}'.format(code))
