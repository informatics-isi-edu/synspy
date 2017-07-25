# -*- mode: python -*-

block_cipher = None

from os import environ as env

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
		         upx=True,
		         debug=env.get("DEBUG", False),
		         console=env.get("DEBUG", False))

launcher = Analysis(['launcher/launcher/__main__.py'],
	               pathex=[],
	               binaries=[('dist/synspy-viewer2d.exe', '.')],
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
          launcher.binaries,
          launcher.zipfiles,
          launcher.datas,
          launcher.dependencies,
          name='synspy-launcher.exe',
          strip=False,
          upx=False,
          debug=env.get("DEBUG", False),
          console=env.get("DEBUG", False),
          icon='launcher/launcher/images/synapse.ico')
