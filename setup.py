# ------------------------------------
# setup-file for the xroms package
# Bjørn Ådlandsvik, <bjorn@imr.no>
# Institute of Marine Research
# 2018-07-23
# ------------------------------------

from distutils.core import setup

setup(name='xroms0',
      version='0.1',
      description='ROMS postprocessing tools in python based on xarray',
      author='Bjørn Ådlandsvik',
      author_email='bjorn@imr.no',
      url='https://github.com/bjornaa/xroms0',
      packages=['xroms0'],
      requires=['numpy', 'scipy', 'xarray']
      )
