#!/usr/bin/env python3
import json,zipfile,hashlib
from pathlib import Path


def build(root):
    root=Path(root).resolve();out=root/'outputs/couette_validation'
    if not (out/'summary.json').is_file():raise RuntimeError('complete the validation report first')
    def cell(kind,source,i):
        c=dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True),id=f'couette-{i}')
        if kind=='code':c.update(execution_count=None,outputs=[])
        return c
    content=[('markdown','# Couette PINN validation\n\nمقایسهٔ دوران تعادلی CO01 و CO02 با مقاله. روش نهایی PINN با دادهٔ کمکیِ حل مستقل FEM است. مرکز استوانه روی خط تقارن تعیین شده و مهاجرت اولیه در این آزمون بررسی نمی‌شود.'),
             ('code','%pip install -r requirements.txt\n'),
             ('code',"from pathlib import Path\nimport sys, subprocess, datetime, json\nfrom IPython.display import Image, display\nOUT = Path('outputs') / ('couette_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))\nresult = subprocess.run([sys.executable, 'validate_couette.py', '--out', str(OUT)], check=False)\nprint('Exit code:', result.returncode)\n"),
             ('code',"s = json.loads((OUT / 'summary.json').read_text())\nfor row in s['rows']:\n    print(row['case_id'], 'paper:', row['reference_spin'], 'PINN:', row['PINN_spin'], 'error %:', row['PINN_error_percent'])\nprint('All declared criteria:', s['all_declared_criteria_met'])\ndisplay(Image(filename=str(OUT / 'rotation_comparison.png')))\ndisplay(Image(filename=str(OUT / 'free_rotation.gif')))\n"),
             ('markdown','برای دیدن دادهٔ خام، جدول‌ها، PDF/SVG و محدودیت‌های نتیجه، index.html و REPORT.md داخل پوشهٔ خروجی را باز کنید. اعداد مقاله وارد آموزش نمی‌شوند؛ میدان سرعت FEM به‌عنوان دادهٔ کمکی داخلی استفاده می‌شود.')]
    notebook=dict(cells=[cell(kind,source,i) for i,(kind,source) in enumerate(content)],
                  metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.12')),
                  nbformat=4,nbformat_minor=5)
    (root/'CouetteValidation.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2))
    release=root/'deliverables';release.mkdir(exist_ok=True)
    files=[]
    for p in root.rglob('*'):
        relative=p.relative_to(root)
        if p.is_file() and not any(x in relative.parts for x in ['.git','outputs','deliverables','__pycache__','.venv']):files.append(p)
    with zipfile.ZipFile(release/'CouetteValidation_code.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,Path('CouetteValidation')/p.relative_to(root))
    with zipfile.ZipFile(release/'CouetteValidation_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,Path('couette_validation')/p.relative_to(out))
    lines=[]
    for name in ['CouetteValidation_code.zip','CouetteValidation_results.zip']:
        p=release/name
        with zipfile.ZipFile(p) as z:assert z.testzip() is None
        lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+name)
    (release/'CouetteValidation_SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
    print('Code/results archives saved and CRC verified.')


if __name__=='__main__':build(Path(__file__).parent)
