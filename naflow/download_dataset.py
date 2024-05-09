import os
import shutil
from pathlib import Path

# import links from https://abdokamel.github.io/sidd/
os.system('wget https://competitions.codalab.org/my/datasets/download/f5c7a97d-5431-4cc0-a101-f2d2fd1665f7 -O SIDD_Medium_Srgb_Parts.zip')
os.system('wget https://competitions.codalab.org/my/datasets/download/7f9926e6-af3d-406c-830a-c5b785548341 -O SIDD_Medium_Srgb_Parts.z01')
os.system('wget https://competitions.codalab.org/my/datasets/download/3a3ec1c4-005f-41f6-8769-c467a99e4bcf -O SIDD_Medium_Srgb_Parts.z02')
os.system('wget https://competitions.codalab.org/my/datasets/download/feb8c3f7-341d-498a-9852-a3459aca4c5c -O SIDD_Medium_Srgb_Parts.z03')
os.system('wget https://competitions.codalab.org/my/datasets/download/0b633399-cfa9-45ff-97e5-d585d2638f48 -O SIDD_Medium_Srgb_Parts.z04')
os.system('wget https://competitions.codalab.org/my/datasets/download/b0b430e3-920e-4778-af5e-4792d7280425 -O SIDD_Medium_Srgb_Parts.z05')
os.system('wget https://competitions.codalab.org/my/datasets/download/f97a3560-fa74-452b-8bf4-292a97121b6a -O SIDD_Medium_Srgb_Parts.z06')
os.system('wget https://competitions.codalab.org/my/datasets/download/3cb70d7e-8361-42d4-a253-58842a39042e -O SIDD_Medium_Srgb_Parts.z07')
os.system('wget https://competitions.codalab.org/my/datasets/download/cb89c789-dd2f-45cc-833b-35fb877f7661 -O SIDD_Medium_Srgb_Parts.z08')
os.system('wget https://competitions.codalab.org/my/datasets/download/a8bd4014-60f0-4e8e-a252-22d02d473fc3 -O SIDD_Medium_Srgb_Parts.z09')
os.system('wget https://competitions.codalab.org/my/datasets/download/cd80db16-11d9-461a-9241-d93a9c1783d3 -O SIDD_Medium_Srgb_Parts.z10')
os.system('wget https://competitions.codalab.org/my/datasets/download/da6fb676-7acd-450e-aff3-693b938e233e -O SIDD_Medium_Srgb_Parts.z11')
os.system('wget https://competitions.codalab.org/my/datasets/download/07378dc1-8434-434a-b902-16a78a2ab53f -O SIDD_Medium_Srgb_Parts.z12')

if not Path('sidd').exists():
    Path('sidd').mkdir()

os.system('zip -FF SIDD_Medium_Srgb_Parts.zip --out trainset.zip')
Path('sidd/trainset').mkdir()
os.system('unzip trainset.zip -d sidd/trainset')

for f in Path(".").glob("SIDD_Medium_Srgb_Parts*"):
    f.unlink()
os.remove('trainset.zip')

os.system('wget --user-agent="Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/121.0" "https://onedrive.live.com/download?resid=85CF5B7F538E2007%2159961&authkey=!AJSw4tY9zCDJCrc" -O validation.tar')
shutil.unpack_archive('validation.tar', 'sidd/.')
os.remove('validation.tar')