# -*- coding: utf-8 -*-
"""F0-8 EQL 失败闭锁策略测试（只加载纯配置/函数，不启动 Abaqus）。"""

from __future__ import print_function

import importlib.util
import os
import sys
import tempfile
import types


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MODEL = os.path.join(REPO, 'Modeling', 'Hybrid', 'slope_frame_ssi_full_v2.py')


def load_model():
    abaqus = types.ModuleType('abaqus')
    abaqus.mdb = object()
    sys.modules['abaqus'] = abaqus
    for name in ('abaqusConstants', 'caeModules', 'mesh'):
        sys.modules[name] = types.ModuleType(name)
    region = types.ModuleType('regionToolset')
    region.Region = object
    sys.modules['regionToolset'] = region
    spec = importlib.util.spec_from_file_location('slope_model_under_test', MODEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Site(object):
    layers = [object()]


def main():
    module = load_model()
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as fh:
        fh.write('0.0 0.0\n0.1 1.0\n')
        wave = fh.name
    try:
        module.eql_cfg['enable'] = True
        module.eql_cfg['fail_closed'] = True
        original = module._run_freefield_eql
        module._run_freefield_eql = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('synthetic EQL failure'))
        failed_closed = False
        try:
            module._run_eql_if_enabled(Site(), object(), [(wave, 0.1, 2)], {}, None)
        except RuntimeError as exc:
            failed_closed = 'fail_closed=True' in str(exc)
        assert failed_closed
        module.eql_cfg['fail_closed'] = False
        site, meta = module._run_eql_if_enabled(Site(), object(), [(wave, 0.1, 2)], {}, None)
        assert meta == {'enable': False}
        module._run_freefield_eql = original
    finally:
        os.remove(wave)
    print('test_eql_fail_closed_v2: 2/2 ok')


if __name__ == '__main__':
    main()
