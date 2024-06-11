# pyinstaller_build.spec

# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_all

a = Analysis(
    ['tmat.py'],
    pathex=['tmat.py'],
    datas=collect_data_files('xmlschema', include_py_files=True),
    hiddenimports=['xsdata_pydantic_basemodel.hooks', 'xsdata_pydantic_basemodel.hooks.class_type', 'PIL._tkinter_finder'],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='tmat',
    debug=False,
    console=False,
    icon="icon.ico",
    onefile=True,
)
