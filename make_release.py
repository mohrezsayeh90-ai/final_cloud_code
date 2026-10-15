#!/usr/bin/env python3
"""Create a single-file launcher, a notebook and portable source/results archives."""
import json
from pathlib import Path
import zipfile


def make_single_file(root):
    modules={'parcel_pinn' if p.stem=='__init__' else 'parcel_pinn.'+p.stem:p.read_text()
             for p in sorted((root/'parcel_pinn').glob('*.py'))}
    # Load embedded source with Python's normal import machinery. No files are
    # silently overwritten and no pickle/eval payload is read from user inputs.
    launcher='''#!/usr/bin/env python3
"""Self-contained Gaussian parcel PINN source. Requires dependencies and Validation_Pack.
Usage: python gaussian_parcel_pinn.py --preset demo --validation-pack ./Validation_Pack --out results
The full readable package and scientific documentation are included in the source ZIP.
"""
import importlib.abc
import importlib.util
import sys
SOURCES = '''+repr(modules)+'''

class EmbeddedSource(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in SOURCES:
            return importlib.util.spec_from_loader(fullname, self, is_package=fullname == 'parcel_pinn')
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.__file__ = __file__
        exec(compile(SOURCES[module.__name__], __file__+':'+module.__name__, 'exec'), module.__dict__)

if __name__ == '__main__':
    sys.meta_path.insert(0, EmbeddedSource())
    from parcel_pinn.pipeline import main
    raise SystemExit(main())
'''
    (root/'gaussian_parcel_pinn.py').write_text(launcher)


def make_notebook(root):
    def cell(kind,source):
        c=dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True))
        if kind=='code':c.update(execution_count=None,outputs=[])
        return c
    cells=[cell('markdown',
        '# Gaussian Parcel PINN\n\nحل دوبعدی با کوپل گاوسی، شکل‌ها و انیمیشن. فایل `gaussian_parcel_pinn.py` و بستهٔ بازشدهٔ `Validation_Pack` را کنار این notebook قرار دهید.\n\n'
        'نمای عمق‌دار، نمایش حل دوبعدی است. نتایج کم‌آموزش برای مقاله معتبر فرض نمی‌شوند. آزمون تحلیلی با اعتبارسنجی آزمایشگاهی متفاوت است. راهنمای کامل در README.md آمده است.'),
        cell('code','%pip install -r requirements.txt\n'),
        cell('code',"from pathlib import Path\nimport sys, subprocess, datetime\nfrom IPython.display import display, HTML, Image\n\nPACK = Path('Validation_Pack')  # مسیر بستهٔ اصلی خودتان\nPRESET = 'demo'               # smoke / demo / paper\nOUT = Path('outputs') / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')\nassert (PACK / 'case_catalog.json').is_file(), 'مسیر Validation_Pack را اصلاح کنید'\n"),
        cell('code',"result = subprocess.run([sys.executable, 'gaussian_parcel_pinn.py', '--preset', PRESET, '--validation-pack', str(PACK), '--out', str(OUT)], check=False)\nprint('Exit code:', result.returncode)\n"),
        cell('code',"import json\nsummary = json.loads((OUT / 'summary.json').read_text())\nprint('Verification:', summary['verification_passed'], '/', summary['verification_total'])\nprint('Literature validation:', summary['physical_validation_cases_run'])\nprint('Research criteria:', summary['research_quality_criteria_met'])\nfor c in summary['cases']:\n    print(c['case'], c['residuals'], c['training']['termination'])\n"),
        cell('code',"display(Image(filename=str(OUT / 'figures' / 'architecture.png')))\nfor g in sorted((OUT / 'cases').glob('*/transport.gif')):\n    print(g.parent.name)\n    display(Image(filename=str(g)))\n"),
        cell('markdown','برای مشاهدهٔ همهٔ شکل‌ها و دریافت PDF/SVG، فایل `index.html` در پوشهٔ خروجی را باز کنید. معیارها، provenance و محدودیت‌ها در `summary.json` و `REPORT.md` هستند. برای استفاده در مقاله، مطالعهٔ seed، عرض فیلتر و رزولوشن quadrature را اجرا و مرجع فیزیکی سازگار تهیه کنید.')]
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                                            language_info=dict(name='python',version='3.12')),nbformat=4,nbformat_minor=5)
    for i,c in enumerate(cells):c['id']=f'parcel-cell-{i}'
    (root/'GaussianParcelPINN.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2))


def build(root, results='evaluated_2d_final'):
    root=Path(root).resolve();make_single_file(root);make_notebook(root)
    release=root/'deliverables';release.mkdir(exist_ok=True)
    files=[root/'run.py',root/'gaussian_parcel_pinn.py',root/'GaussianParcelPINN.ipynb',root/'study.py',root/'render.py',
           root/'requirements.txt',root/'README.md',root/'.gitignore',root/'make_release.py']
    for folder in ['parcel_pinn','tests','docs','configs']:
        files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    with zipfile.ZipFile(release/'GaussianParcelPINN_code.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,Path('GaussianParcelPINN')/p.relative_to(root))
    result=root/'outputs'/results
    if result.is_dir():
        with zipfile.ZipFile(release/'GaussianParcelPINN_results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in result.rglob('*'):
                if p.is_file():z.write(p,Path('results')/p.relative_to(result))
    return release


if __name__=='__main__':
    print(build(Path(__file__).parent))
