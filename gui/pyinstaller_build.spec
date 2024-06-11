# pyinstaller_build.spec

# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_all

block_cipher = None

a = Analysis(
    ['tmat.py'],
    pathex=['.'],
    binaries=[],
    datas=collect_data_files('xmlschema', include_py_files=True) + [
        ('../config/default_branching_computation.json', 'config'),
        ('../config/default_cell_area_computation.json', 'config'),
        ('../config/default_invasion_depth_computation.json', 'config'),
        ('../scripts/compute_branches.py', 'scripts'),
        ('../scripts/compute_cell_area.py', 'scripts'),
        ('../scripts/compute_inv_depth.py', 'scripts'),
        ('../scripts/compute_zproj.py', 'scripts'),
        ('../model_training/invasion_depth_best_hp.json', 'model_training'),
        ('../model_training/invasion_depth_hp_space.json', 'model_training'),
        ('../model_training/invasion_depth_training_values.json', 'model_training'),
        ('../model_training/best_ensemble/best_finetune_weights_0.h5', 'model_training/best_ensemble'),
        ('../model_training/best_ensemble/best_finetune_weights_1.h5', 'model_training/best_ensemble'),
        ('../model_training/best_ensemble/best_finetune_weights_2.h5', 'model_training/best_ensemble'),
        ('../model_training/best_ensemble/best_finetune_weights_3.h5', 'model_training/best_ensemble'),
        ('../model_training/best_ensemble/best_finetune_weights_4.h5', 'model_training/best_ensemble'),
        ('../model_training/best_ensemble/best_model_history_0.csv', 'model_training/best_ensemble'),
        ('../model_training/best_ensemble/best_model_history_1.csv', 'model_training/best_ensemble'),
        ('../model_training/best_ensemble/best_model_history_2.csv', 'model_training/best_ensemble'),
        ('../model_training/best_ensemble/best_model_history_3.csv', 'model_training/best_ensemble'),
        ('../model_training/best_ensemble/best_model_history_4.csv', 'model_training/best_ensemble'),
        ('../model_training/binary_segmentation/checkpoints/checkpoint_1.h5', 'model_training/binary_segmentation/checkpoints'),
        ('../model_training/binary_segmentation/configs/unet_patch_segmentor_1.json', 'model_training/binary_segmentation/configs'),
    ],
    hiddenimports=['xsdata_pydantic_basemodel.hooks', 'xsdata_pydantic_basemodel.hooks.class_type', 'PIL._tkinter_finder', 'tensorflow.python._pywrap_tensorflow_internal'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='tmat',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,
    icon="icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='tmat',
)
