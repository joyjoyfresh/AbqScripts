# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2021 replay file
# Internal Version: 2020_03_06-22.50.37 167380
# Run by CodexSandboxOffline on Thu Jul  9 17:09:14 2026
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
execfile('test/Hybrid/test_postprocess_all_surface_v2.py', __main__.__dict__)
#* ImportError: No module named util
#* File "test/Hybrid/test_postprocess_all_surface_v2.py", line 2, in <module>
#*     import importlib.util
