# -*- coding: utf-8 -*-

# ==== 1. 确保RSG临时目录可import ====
import sys
import os

plugin_name = 'VAB_oblique_v10'
rsg_tmp_dir = os.path.join(
    os.path.expanduser("~"),
    'abaqus_plugins', '_rsgTmpDir', plugin_name
)
if rsg_tmp_dir not in sys.path:
    sys.path.insert(0, rsg_tmp_dir)

# ==== 2. 标准Abaqus插件 import ====
from abaqusGui import getAFXApp, Activator, AFXMode
from abaqusConstants import ALL

# ==== 3. 插件菜单注册 ====
thisPath = os.path.abspath(__file__)
thisDir = os.path.dirname(thisPath)

toolset = getAFXApp().getAFXMainWindow().getPluginToolset()
toolset.registerGuiMenuButton(
    buttonText='VAB_oblique_v10',
    object=Activator(os.path.join(thisDir, 'vAB_oblique_v10DB.py')),
    kernelInitString='import VAB_oblique_v10',
    messageId=AFXMode.ID_ACTIVATE,
    icon=None,
    applicableModules=ALL,
    version='N/A',
    author='N/A',
    description='N/A',
    helpUrl='N/A'
)