#!/usr/bin/env python3
"""Build the original coupled Gaussian GPU notebook with geometry-only additions."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
from build_gpu_notebook import cell, write_member


def build(root):
    root = Path(root)
    files = sorted((root/'parcel_pinn').glob('*.py'))+sorted((root/'gaussian_constriction').glob('*.py'))+[
        root/p for p in ['run_constriction.py', 'requirements-gpu.txt', 'docs/CONSTRICTION_GPU_FA.md',
        'reference_inputs/Validation_Pack/case_catalog.json', 'reference_inputs/Validation_Pack/sources.json',
        'reference_inputs/Validation_Pack/README.md', 'reference_inputs/Validation_Pack/data/analytic_stokes.csv',
        'reference_inputs/Validation_Pack/data/analytic_two_way_relaxation.csv',
        'reference_inputs/Validation_Pack/data/analytic_gaussian_overlap3d.csv']]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files: write_member(z, p.relative_to(root), p.read_bytes())
    payload = buffer.getvalue(); digest = hashlib.sha256(payload).hexdigest()
    nb = json.loads((root/'GaussianParcelPINN_GPU.ipynb').read_text()); cells = nb['cells']
    cells[0] = cell('markdown', '''
    # Gaussian Parcels · Bilateral Constriction · GPU

    **همان مدل قبلیِ سیال–ابر ذرات گاوسی، با افزودن دو گرفتگی نیم‌دایره‌ای.**
    حل زمانی دوبعدی، دو شبکهٔ سیال و پارسل، درگ استوکس، کوپل دوطرفهٔ گاوسی، مرز تناوبی x و نیروی محرک قبلی حفظ شده‌اند.
    این نسخه پایا یا بدون ذره نیست. طول L/H=3، Re=20، تعداد پارسل، عرض فیلتر، گرانش و چهار حالت استوکس/بارگذاری همان‌اند.
    در Colab، GPU را فعال و Run all کنید؛ کد و داده‌های تحلیلی لازم داخل همین notebook هستند.
    هدف ۵۵ دقیقه و حداکثر ۳۵ دقیقه آموزش است؛ بودجهٔ زمان تضمین دقت نیست.

    شرط اولیهٔ سیال برای رعایت مرز خمیده سازگار می‌شود؛ در شعاع صفر مدل قبلی دقیقاً بازیابی می‌شود.
    جامد از نقاط حل و انتگرال کرنل حذف می‌شود. مدل همچنان رقیق و بدون برخورد است؛ ورود مرکز به جامد با loss جریمه و صریح گزارش می‌شود.
    مسیر یا برخورد ساختگی به خروجی اضافه نمی‌شود. اعتبارسنجی مقاله‌ای هندسهٔ جدید ادعا نمی‌شود.
    ''', 0)
    parameters = ''.join(cells[1]['source']).replace('gaussian_gpu_results', 'gaussian_constricted_results').replace('gaussian_gpu_source_', 'gaussian_constricted_source_')
    parameters += "\n\n# فقط هندسهٔ افزوده‌شده؛ بقیهٔ فیزیک همان مدل قبلی است\nGEOMETRY = {'radius': 0.25, 'center_fraction': 0.4}\n"
    cells[1] = cell('code', parameters, 1)
    source = ''.join(cells[3]['source']); a = source.index('SOURCE_B64 ='); b = source.index('payload = base64.b64decode')
    source = source[:a]+f'SOURCE_B64 = {base64.b64encode(payload).decode()!r}\nSOURCE_SHA256 = {digest!r}\n'+source[b:]
    cells[3] = cell('code', source, 3, hidden=True)
    launch = ''.join(cells[7]['source']).replace("str(WORK / 'run_gpu.py')", "str(WORK / 'run_constriction.py')")
    launch = launch.replace('command = [sys.executable', "geometry_path = WORK / 'geometry_parameters.json'\ngeometry_path.write_text(json.dumps(GEOMETRY))\ncommand = [sys.executable")
    launch = launch.replace("'--out', str(OUT),", "'--out', str(OUT), '--geometry', str(geometry_path),")
    cells[7] = cell('code', launch, 7)
    metrics = ''.join(cells[9]['source'])+"\nif (OUT / 'summary.json').is_file():\n    print('Particles:', summary['particles_modeled'], '| physical time:', summary['physical_time_modeled'], '| two-way Gaussian:', summary['two_way_gaussian_coupling'])\n    for c in summary['cases']:\n        print(c['case'], 'solid centers:', c['max_obstacle_inside_count'], 'kernel error:', c['max_kernel_error'])\n"
    cells[9] = cell('code', metrics, 9)
    cells[12] = cell('markdown', '## حرکت پارسل‌ها و ابرهای گاوسی\nframeها زمان فیزیکی پیش‌بینی‌شده هستند. نفوذ احتمالی مرکز به جامد در گزارش و با دورخط قرمز در انیمیشن مشخص می‌شود؛ موقعیت‌ها clip یا project نمی‌شوند.', 12)
    preview = [cell('markdown', '## هندسهٔ افزوده‌شده\nL/H=3 همان مدل قبلی؛ دو نیم‌دایره با r/H=.25 و x₀/L=.4. مرز x تناوبی و نیروی محرک ثابت مانده‌اند.', 16), cell('code', '''
    sys.path.insert(0, str(WORK))
    from gaussian_constriction.model import Geometry, GeometrySettings
    from gaussian_constriction.figures import domain
    from parcel_pinn.config import Config
    import matplotlib.pyplot as plt
    from IPython.display import display, Image
    import io
    geometry = Geometry(Config().lengths[0], GeometrySettings(**GEOMETRY))
    fig, ax = plt.subplots(figsize=(10, 4)); domain(ax, geometry)
    ax.set(xlim=(0, geometry.length), ylim=(0, 1), aspect='equal', xlabel='x / H', ylabel='y / H', title='Channel geometry')
    buffer = io.BytesIO(); fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
    display(Image(data=buffer.getvalue())); plt.close(fig)
    print('Transient fluid + two-way Gaussian parcels; periodic x; same forcing')
    ''', 17)]
    cells[6:6] = preview
    for i, c in enumerate(cells): c['id'] = f'gaussian-constricted-{i}'
    nb['cells'] = cells; nb['metadata']['colab']['name'] = 'BilateralConstriction_GPU.ipynb'
    path = root/'BilateralConstriction_GPU.ipynb'; path.write_text(json.dumps(nb, ensure_ascii=False, indent=2))
    extras = [root/p for p in ['build_constriction_notebook.py', 'build_gaussian_constriction_notebook.py', 'build_gpu_notebook.py',
        'GaussianParcelPINN_GPU.ipynb', 'BilateralConstriction_GPU.ipynb', 'tests/test_gaussian_constriction.py',
        'docs/CONSTRICTION_CHECKS.md', 'docs/CONSTRICTION_NOTEBOOK_QA.json']]
    archive = root/'deliverables/BilateralConstriction_GPU_code.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files+extras:
            if p.is_file(): write_member(z, Path('GaussianConstrictedPINN')/p.relative_to(root), p.read_bytes())
    archive.with_suffix('.zip.sha256').write_text(f'{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n')
    print(f'{path}: {path.stat().st_size:,} bytes; embedded source SHA256={digest}')
    return nb


if __name__ == '__main__': build(Path(__file__).resolve().parent)
