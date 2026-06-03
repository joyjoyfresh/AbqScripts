# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2021 replay file
# Internal Version: 2020_03_06-22.50.37 167380
# Run by 12462 on Wed Jun  3 22:30:08 2026
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.3724, 1.37037), width=202.017, 
    height=135.941)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('freefield_selfcheck_v1.py', __main__.__dict__)
#* NameError: global name '__file__' is not defined
#* File "freefield_selfcheck_v1.py", line 154, in <module>
#*     run_selfcheck()  # 调用自检主函数
#* File "freefield_selfcheck_v1.py", line 60, in run_selfcheck
#*     eng = _load_engine()  # 加载引擎模块
#* File "freefield_selfcheck_v1.py", line 29, in _load_engine
#*     here = os.path.dirname(os.path.abspath(__file__))  # 
#* 取当前脚本所在目录
