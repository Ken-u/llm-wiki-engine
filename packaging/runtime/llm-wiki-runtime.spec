# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path.cwd()

datas = [
    (
        str(ROOT / "app" / "runtime" / "ui_dist"),
        "app/runtime/ui_dist",
    ),
    (
        str(ROOT / "packaging" / "runtime" / "hooks"),
        "hooks",
    ),
]
datas += collect_data_files("litellm")
datas += collect_data_files("lancedb")
datas += collect_data_files("tiktoken")

hiddenimports = []
hiddenimports += collect_submodules("lancedb")
hiddenimports += [
    "litellm",
    "litellm.llms",
    "litellm.llms.openai",
    "litellm.llms.openai.chat",
    "litellm.llms.openai.chat.gpt_transformation",
    "litellm.types",
    "litellm.utils",
    "pandas",
    "pyarrow",
    "pyarrow.lib",
    "tiktoken_ext",
    "tiktoken_ext.openai_public",
]

a = Analysis(
    [str(ROOT / "app" / "runtime_main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "tests",
        "pandas.tests",
        "pyarrow.tests",
        "matplotlib",
        "IPython",
        "notebook",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="llm-wiki-runtime",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
