# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for L.A.M.B.D.A. Build System

Build with:  pyinstaller lambda.spec
Output:      dist/LAMBDA.exe
"""

import os
import glob
import importlib

block_cipher = None

spec_dir = os.path.dirname(os.path.abspath(SPEC))

# Locate Open3D's resources directory (Filament shaders, etc.)
o3d_resources = []
try:
    o3d_path = os.path.dirname(importlib.import_module('open3d').__file__)
    o3d_res = os.path.join(o3d_path, 'resources')
    if os.path.isdir(o3d_res):
        o3d_resources = [(o3d_res, os.path.join('open3d', 'resources'))]
except ImportError:
    pass

# Collect all compiler .py files (excluding my-env, recovery_scripts)
compiler_datas = []
compiler_root = os.path.join(spec_dir, 'compiler')
for root, dirs, files in os.walk(compiler_root):
    # Skip directories we don't want bundled
    dirs[:] = [d for d in dirs if d not in ('my-env', '.mypy_cache', '__pycache__',
                                             'recovery_scripts')]
    for f in files:
        if f.endswith('.py'):
            src = os.path.join(root, f)
            # Destination preserves compiler/ relative structure
            dest = os.path.relpath(root, os.path.dirname(compiler_root))
            compiler_datas.append((src, dest))

# Collect all visualiser .py files
vis_datas = []
vis_root = os.path.join(spec_dir, 'visualiser')
for root, dirs, files in os.walk(vis_root):
    dirs[:] = [d for d in dirs if d not in ('__pycache__',)]
    for f in files:
        if f.endswith('.py'):
            src = os.path.join(root, f)
            dest = os.path.relpath(root, os.path.dirname(vis_root))
            vis_datas.append((src, dest))

a = Analysis(
    ['lambda.py'],
    pathex=[compiler_root],
    binaries=[],
    datas=[
        ('docs/images/lambda_logo.png', 'docs/images'),
        ('docs/images/header.jpg', 'docs/images'),
        ('docs/images/icon.png', 'docs/images'),
        ('docs/images/visualiser_icon.ico', 'docs/images'),
        ('docs/images/visualiser_icon.png', 'docs/images'),
    ] + compiler_datas + vis_datas + o3d_resources,
    hiddenimports=[
        'numpy',
        'open3d',
        'open3d.visualization',
        'open3d.visualization.gui',
        'open3d.visualization.rendering',
        'config',
        'config.mod_config',
        'config.mod_copier',
        'config.tag_rewriter',
        'converters',
        'crosstables',
        'crosstables.builder',
        'crosstables.cross_table_remapper',
        'crosstables.data_types',
        'crosstables.level_graph_navigator',
        'extraction',
        'generation',
        'generation.death_point_generator',
        'graph',
        'graph.game_graph',
        'graph.orphan_connector',
        'levels',
        'levels.levels_config',
        'levels.level_game_parser',
        'parsers',
        'parsers.level_spawn',
        'parsers.game_graph',
        'parsers.cross_table',
        'parsers.patrol_paths',
        'parsers.level_ai',
        'patrols',
        'remapping',
        'remapping.patrol_remapper',
        'serialization',
        'serialization.game_graph_serializer',
        'serialization.all_spawn_writer',
        'serialization.header_serializer',
        'spawn_graph',
        'spawn_graph.builder',
        'utils',
        'utils.logging',
        'utils.guid',
        'utils.binary',
        'build_all_spawn',
        'game_graph_merger',
        'constants',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

# Filter out compiler/my-env from any auto-discovered modules
a.datas = [d for d in a.datas if 'my-env' not in d[0] and 'recovery_scripts' not in d[0]]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

splash = Splash(
    'docs/images/lambda_splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    text_color='white',
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash,
    splash.binaries,
    [],
    name='LAMBDA',
    icon=os.path.join(os.path.dirname(os.path.abspath(SPEC)), 'docs', 'images', 'lambda_icon.ico'),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
