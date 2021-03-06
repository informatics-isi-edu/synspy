# -*- mode: python -*-

block_cipher = None

from os import environ as env
from synspy import __version__ as ver

viewer = Analysis(['synspy/viewer2d.py'],
	               pathex=[],
	               binaries=None,
	               hiddenimports=['six', 'PyQt5', 'vispy.app.backends._pyqt5'],
	               hookspath=['.'],
	               runtime_hooks=[],
	               excludes=[],
	               win_no_prefer_redirects=False,
	               win_private_assemblies=False,
	               cipher=block_cipher)

viewer_pyz = PYZ(viewer.pure, viewer.zipped_data, cipher=block_cipher)

viewer_exe = EXE(viewer_pyz,
		         viewer.scripts,
		         viewer.binaries,
		         viewer.zipfiles,
		         viewer.datas,
		         viewer.dependencies,
		         name='synspy-viewer2d',
		         strip=False,
		         upx=False,
                 debug=env.get("DEBUG", False),
		         console=env.get("DEBUG", False))

launcher = Analysis(['launcher/launcher/__main__.py'],
	               pathex=[],
	               binaries=[('dist/synspy-viewer2d', '.')],
	               hiddenimports=[],
	               hookspath=[],
	               runtime_hooks=[],
	               excludes=[],
	               win_no_prefer_redirects=False,
	               win_private_assemblies=False,
	               cipher=block_cipher)

launcher_pyz = PYZ(launcher.pure, launcher.zipped_data, cipher=block_cipher)

exe = EXE(launcher_pyz,
          launcher.scripts,
          exclude_binaries=True,
          name='Synapse Launcher',
          strip=False,
          upx=False,
          debug=env.get("DEBUG", False),
          console=env.get("DEBUG", False))

coll = COLLECT(exe,
               launcher.binaries,
               launcher.zipfiles,
               launcher.datas,
               launcher.dependencies,
               strip=False,
               upx=False,
               name='Synapse Launcher')

app = BUNDLE(coll,
         name='Synapse Launcher.app',
         icon='launcher/launcher/images/synapse.icns',
         bundle_identifier='org.qt-project.Qt.QtWebEngineCore',
         info_plist={
            'CFBundleDisplayName': 'Synapse Launcher Utility',
            'CFBundleShortVersionString':ver,
            'NSPrincipalClass':'NSApplication',
            'NSHighResolutionCapable': 'True'
         })