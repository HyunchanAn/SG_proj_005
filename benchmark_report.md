# Benchmark & E2E Test Report

- **Repository**: SG_proj_005
- **Date**: 2026-07-14 22:43:26

## 1. E2E Testing Summary
❌ **Status**: FAILED

### Test Logs (Snippet)
```text
plugins: anyio-4.12.1, cov-7.1.0, hypothesis-6.155.7, hydra-core-1.3.2, respx-0.23.1
collected 0 items / 2 errors

==================================== ERRORS ====================================
___________________ ERROR collecting tests/test_inference.py ___________________
ImportError while importing test module '/Users/hyunchanan/Documents/GitHub/SG_proj_005/tests/test_inference.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_inference.py:13: in <module>
    from inference_engine import IntegratedEngine
E   ModuleNotFoundError: No module named 'inference_engine'
_____________________ ERROR collecting tests/test_train.py _____________________
ImportError while importing test module '/Users/hyunchanan/Documents/GitHub/SG_proj_005/tests/test_train.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_train.py:10: in <module>
    from train import patched_create_versioned_dir, train
E   ModuleNotFoundError: No module named 'train'
=========================== short test summary info ============================
ERROR tests/test_inference.py
ERROR tests/test_train.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!!
============================== 2 errors in 4.52s ===============================

```

## 2. Models Detected
- `exported_models/weights/torch/model.pt` (125.12 MB)
