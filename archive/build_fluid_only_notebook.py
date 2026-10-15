#!/usr/bin/env python3
"""Build a standalone notebook, preserving the tested GPU setup/watchdog from the base notebook."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
from build_gpu_notebook import cell, write_member


def build(root):
    root = Path(root)
    files = sorted((root/'parcel_pinn').glob('*.py'))+sorted((root/'constricted_flow').glob('*.py'))+[
        root/'run_constriction.py', root/'requirements-gpu.txt', root/'docs/CONSTRICTION_GPU_FA.md']
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in files: write_member(z, path.relative_to(root), path.read_bytes())
    data = buffer.getvalue(); digest = hashlib.sha256(data).hexdigest()
    base = json.loads((root/'GaussianParcelPINN_GPU.ipynb').read_text())
    cells = base['cells']
    cells[0] = cell('markdown', '''
    # Bilateral Constriction · GPU

    حل **جریان پایای دوبعدی سیال** در کانالی با دو نیم‌دایرهٔ روبه‌روی هم، مطابق تصویر نمادین شما.
    **فعلاً هیچ ذره یا پارسل گاوسی مدل نمی‌شود.** ورودی/خروجی باز، دیواره‌های بدون لغزش و تقارن خط وسط اعمال می‌شوند.
    سه حالت Re=10,20,50 و یک کنترل تحلیلی کانال مستقیم اجرا می‌شوند.

    در Colab، GPU را از Runtime → Change runtime type فعال و سپس Run all کنید.
    این notebook کد لازم را همراه خود دارد؛ clone، ZIP داده یا فایل خارجی لازم نیست.
    هدف پیش‌فرض حداکثر ۵۵ دقیقه از اولین سلول، شامل حداکثر ۳۵ دقیقه آموزش است؛ برای شکل‌ها و ZIP بودجهٔ جدا رزرو می‌شود.
    سقف زمان تضمین دقت یا تکمیل همهٔ خروجی‌ها روی هر سخت‌افزار نیست؛ وضعیت واقعی در گزارش باقی می‌ماند.

    انیمیشن‌ها اسکن یک میدان **پایا** هستند و گذرای سیال یا حرکت ذره را نشان نمی‌دهند.
    کنترل پوازی بررسی حد کانال مستقیم است؛ اعتبارسنجی مقاله‌ای هندسهٔ جدید ادعا نمی‌شود.
    ''', 0)
    # Keep the original timer, dependency checks and watchdog; replace only the case-specific paths.
    parameters = ''.join(cells[1]['source']).replace('gaussian_gpu_results', 'constriction_gpu_results').replace('gaussian_gpu_source_', 'constriction_gpu_source_')
    parameters += '''\n\n# H=1: نسبت‌های هندسهٔ نمادین؛ قابل تغییر\nGEOMETRY = {\n    'length': 2.5,\n    'center_fraction': 0.4,\n    'radius': 0.25,\n    'reynolds': [10.0, 20.0, 50.0],\n}\n'''
    cells[1] = cell('code', parameters, 1)
    payload = ''.join(cells[3]['source'])
    start = payload.index('SOURCE_B64 ='); end = payload.index('payload = base64.b64decode')
    payload = payload[:start]+f'SOURCE_B64 = {base64.b64encode(data).decode()!r}\nSOURCE_SHA256 = {digest!r}\n'+payload[end:]
    cells[3] = cell('code', payload, 3, hidden=True)
    cells[6] = cell('markdown', '''
    ## هندسه و آموزش
    ابتدا هندسه را بررسی کنید. داخل نیم‌دایره‌ها از دامنهٔ حل حذف شده است و شرط بدون لغزش روی قوس‌ها اعمال می‌شود.
    آموزش چهار حالت مستقل با Adam، float32 و بودجهٔ زمانی محدود انجام می‌شود.
    کنترل کانال مستقیم حداکثر ۹۰ ثانیه آموزش می‌گیرد؛ زمان باقی‌مانده بین حالت‌های گرفتگی‌دار تقسیم می‌شود.
    خطای زیاد مانع رسم نتایج نیست؛ کیفیت بر اساس residual مستقل و خطای دبی در گزارش ثبت می‌شود.
    ''', 6)
    launch = ''.join(cells[7]['source'])
    launch = launch.replace("command = [sys.executable, '-u', str(WORK / 'run_gpu.py'),\n           '--out', str(OUT), '--validation-pack', str(WORK / 'reference_inputs/Validation_Pack'),",
                            "config_path = WORK / 'geometry_config.json'\nconfig_path.write_text(json.dumps(GEOMETRY, indent=2))\ncommand = [sys.executable, '-u', str(WORK / 'run_constriction.py'),\n           '--out', str(OUT), '--config', str(config_path),")
    if "str(WORK / 'run_gpu.py')" in launch or '--validation-pack' in launch:
        raise RuntimeError('Base notebook launch format changed; update this builder.')
    cells[7] = cell('code', launch, 7)
    cells[8] = cell('markdown', '## گزارش و معیارها\nخطای دبی، معادلات و شرایط مرزی مستقل از loss آموزش گزارش می‌شوند. باقی‌ماندهٔ معادلات درصد خطا در برابر مقاله نیست.', 8)
    cells[9] = cell('code', '''
    from IPython.display import display, HTML, Image, FileLink
    import pandas as pd
    if (OUT / 'summary.json').is_file():
        summary = json.loads((OUT / 'summary.json').read_text())
        print('Workflow complete:', summary['workflow_complete'])
        print('Screening criteria:', summary['research_quality_criteria_met'])
        print('Particles modeled:', summary['particles_modeled'])
        print('Physical time modeled:', summary['physical_time_modeled'])
        print('Literature validation:', summary['physical_validation_cases_run'])
        print('Plot errors:', summary['plot_errors'])
        if (OUT / 'paper_metrics.csv').stat().st_size > 2:
            display(pd.read_csv(OUT / 'paper_metrics.csv'))
        for case in summary['cases']:
            if case['analytic_straight_control'] is not None:
                print('Straight-limit implementation check:', case['analytic_straight_control'])
    else:
        print('گزارش نهایی ساخته نشده؛ checkpointها و logها را بررسی کنید:', OUT)
    ''', 9)
    cells[10] = cell('markdown', '## شکل‌ها و مقایسهٔ حالت‌ها\nساختار شبکه، میدان‌ها، خطوط جریان، دبی، فشار، تنش دیواره و باقی‌مانده‌ها در PNG/PDF/SVG ذخیره می‌شوند. نمای عمق‌دار، حل سه‌بعدی نیست.', 10)
    cells[12] = cell('markdown', '## اسکن میدان پایا\nمقطع مشاهده در طول کانال جابه‌جا می‌شود؛ این نمایش زمان فیزیکی یا حرکت ذره نیست.', 12)
    cells[13] = cell('code', '''
    for animation in sorted((OUT / 'cases').glob('*/section_scan.gif')):
        print(animation.parent.name)
        display(Image(filename=str(animation)))
    ''', 13)
    cells[14] = cell('markdown', '''
    ## دریافت خروجی کامل
    ZIP شامل شکل‌ها، GIF/MP4، میدان‌های CSV/NPZ، جدول LaTeX، checkpointها، تنظیمات، logها و کد اجرا است.
    `index.html` گالری کامل آفلاین و `REPORT.md` گزارش علمی همراه با محدودیت‌ها است.
    برای هندسهٔ گرفتگی‌دار، اعتبارسنجی مقاله‌ای ادعا نمی‌شود. اسکن‌ها مربوط به حل پایا هستند.
    ''', 14)
    preview = [cell('markdown', '## پیش‌نمایش هندسه\nپیش‌فرض: L/H=2.5، r/H=.25، x₀/L=.4؛ ارتفاع گلوگاه .5H است. این ابعاد تخمینی و قابل تغییرند.', 16),
               cell('code', '''
    sys.path.insert(0, str(WORK))
    from constricted_flow.model import Settings, Geometry
    from constricted_flow.report import domain
    import matplotlib.pyplot as plt
    from IPython.display import display, Image
    import io
    geometry_settings = Settings(**GEOMETRY).validate()
    geometry = Geometry(geometry_settings.length, geometry_settings.center_fraction, geometry_settings.radius)
    fig, ax = plt.subplots(figsize=(10, 4))
    domain(ax, geometry)
    ax.set_title('Channel geometry')
    preview_buffer = io.BytesIO()
    fig.savefig(preview_buffer, format='png', dpi=110, bbox_inches='tight')
    display(Image(data=preview_buffer.getvalue()))
    plt.close(fig)
    print('Minimum gap / H:', 1 - 2 * geometry.radius)
    ''', 17)]
    cells[6:6] = preview
    for i, c in enumerate(cells): c['id'] = f'constriction-gpu-{i}'
    base['metadata']['colab']['name'] = 'BilateralConstriction_GPU.ipynb'
    base['cells'] = cells
    path = root/'BilateralConstriction_GPU.ipynb'
    path.write_text(json.dumps(base, ensure_ascii=False, indent=2))
    release = root/'deliverables'/'BilateralConstriction_GPU_code.zip'
    extras = [root/'build_constriction_notebook.py', root/'build_gpu_notebook.py', root/'GaussianParcelPINN_GPU.ipynb', path,
              root/'tests/test_constricted_flow.py', root/'docs/CONSTRICTION_CHECKS.md', root/'docs/CONSTRICTION_NOTEBOOK_QA.json']
    with zipfile.ZipFile(release, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files+extras:
            if p.is_file(): write_member(z, Path('BilateralConstriction_GPU')/p.relative_to(root), p.read_bytes())
    release.with_suffix('.zip.sha256').write_text(f'{hashlib.sha256(release.read_bytes()).hexdigest()}  {release.name}\n')
    print(f'{path}: {path.stat().st_size:,} bytes. Embedded SHA256={digest}')
    return base


if __name__ == '__main__': build(Path(__file__).resolve().parent)
