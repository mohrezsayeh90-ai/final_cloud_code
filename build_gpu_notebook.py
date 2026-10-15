#!/usr/bin/env python3
"""Rebuild the self-contained GPU notebook from the readable repository source."""
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile


def write_member(archive, name, payload):
    info = zipfile.ZipInfo(str(name), date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def cell(kind, source, number, hidden=False):
    result = dict(cell_type=kind, id=f'gaussian-gpu-{number}',
                  metadata={'jupyter': {'source_hidden': True}} if hidden else {},
                  source=textwrap.dedent(source).strip().splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build(root):
    root = Path(root)
    # Only the reference files actually used by V01--V10 are embedded. The full
    # original uploaded archive and literature catalogue remain on GitHub.
    files = sorted((root/'parcel_pinn').glob('*.py')) + [root/p for p in [
        'run_gpu.py', 'render.py', 'requirements-gpu.txt', 'docs/METHOD.md', 'docs/GPU_NOTEBOOK_FA.md',
        'reference_inputs/Validation_Pack/case_catalog.json',
        'reference_inputs/Validation_Pack/sources.json',
        'reference_inputs/Validation_Pack/README.md',
        'reference_inputs/Validation_Pack/data/analytic_stokes.csv',
        'reference_inputs/Validation_Pack/data/analytic_two_way_relaxation.csv',
        'reference_inputs/Validation_Pack/data/analytic_gaussian_overlap3d.csv',
    ]]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            write_member(archive, path.relative_to(root), path.read_bytes())
    payload = buffer.getvalue()
    encoded, digest = base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest()
    cells = []
    def add(kind, text, hidden=False):
        cells.append(cell(kind, text, len(cells), hidden))
    add('markdown', '''
    # Gaussian Parcel PINN · GPU

    این notebook مستقل، **مدل اصلی پارسل گاوسی در کانال دوبعدی** را اجرا می‌کند؛ آزمون استوانه اجرا نمی‌شود.
    در Colab از **Runtime → Change runtime type → GPU** استفاده کنید و سپس **Run all** را بزنید.
    در Kaggle نیز GPU را فعال کنید. نیازی به clone مخزن، فایل ZIP ورودی یا اتصال Google Drive نیست؛ کد و مراجع تحلیلی لازم داخل همین notebook هستند.

    **زمان:** هدف پیش‌فرض، حداکثر ۵۵ دقیقه از شروع سلول اول، شامل نصب و تولید خروجی است؛ حداکثر ۳۵ دقیقه برای آموزش چهار حالت.
    زمان باقی‌مانده برای ارزیابی، شکل‌ها، انیمیشن و ZIP نگه داشته می‌شود. محدودکنندهٔ زمان می‌تواند کار ناقص را متوقف کند؛
    سرعت GPU، نصب وابستگی‌ها و قطع سرویس قابل تضمین نیست. رسیدن به این سقف، تضمین دقت یا تولید همهٔ خروجی‌ها روی هر دستگاه نیست.

    گاوسی در این مدل **کرنل تبادل نیرو** است؛ پهنا ثابت است و حرکت براونی یا تکامل پهنای ابر مدل نشده است.
    نمودارهای تحلیلی، آزمون اجزای مدل هستند و اعتبارسنجی مقاله‌ای مدل کوپل‌شده محسوب نمی‌شوند.
    خطا و وضعیت آموزش در گزارش باقی می‌مانند؛ عنوان شکل‌ها کوتاه و بدون عبارت همگرا/ناهمگراست.
    ''')
    add('code', '''
    import time, os
    from pathlib import Path
    from datetime import datetime, timezone
    import subprocess, sys, json

    NOTEBOOK_STARTED = time.monotonic()
    TOTAL_MINUTES = 55       # سقف کل؛ بیشتر از 55 مجاز نیست
    TRAIN_MINUTES = 35       # سهم کل آموزش؛ بین چهار حالت تقسیم می‌شود
    PRECISION = 'float32'    # روی T4 مناسب‌تر؛ float64 کندتر است
    OUTPUT_BASE = Path.cwd() / 'gaussian_gpu_results'
    # برای ماندگاری، می‌توانید OUTPUT_BASE را به مسیر Drive متصل‌شده تغییر دهید.
    assert 0 < TOTAL_MINUTES <= 55 and TRAIN_MINUTES > 0
    RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    WORK = Path.cwd() / f'gaussian_gpu_source_{RUN_ID}'
    OUT = OUTPUT_BASE / f'run_{RUN_ID}'
    NOTEBOOK_DEADLINE = NOTEBOOK_STARTED + TOTAL_MINUTES * 60
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    print('Output:', OUT)
    ''')
    add('markdown', '## کد همراه notebook\nسلول بعدی بستهٔ کد را با SHA-256 بررسی و در پوشه‌ای جدید استخراج می‌کند. فایل‌های قبلی بازنویسی نمی‌شوند.')
    add('code', f'''
    import base64, hashlib, io, zipfile
    SOURCE_B64 = {encoded!r}
    SOURCE_SHA256 = {digest!r}
    payload = base64.b64decode(SOURCE_B64)
    assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA256, 'Source checksum mismatch'
    WORK.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.testzip() is None
        for item in archive.infolist():
            assert (WORK / item.filename).resolve().is_relative_to(WORK.resolve())
        archive.extractall(WORK)
    print('Source verified:', SOURCE_SHA256)
    ''', hidden=True)
    add('markdown', '''
    ## آماده‌سازی GPU
    PyTorch موجود در محیط حفظ می‌شود تا نسخهٔ CUDA با نسخهٔ CPU جایگزین نشود.
    اگر GPU در دسترس نباشد، این notebook صریحاً متوقف می‌شود؛ آموزش مخفیانه روی CPU ادامه پیدا نمی‌کند.
    ''')
    add('code', '''
    import importlib.util, importlib.metadata
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU را از تنظیمات runtime فعال کنید و دوباره Run all بزنید.')
    print('GPU:', torch.cuda.get_device_name(0))
    print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)
    assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 2)

    # Colab/Kaggle معمولاً این بسته‌ها را دارند؛ فقط وابستگی ناسازگار/غایب نصب می‌شود.
    constraints = {
        'numpy': ('numpy>=1.26,<3', (1, 26), (3, 0)),
        'scipy': ('scipy>=1.12,<2', (1, 12), (2, 0)),
        'pandas': ('pandas>=2.2,<3', (2, 2), (3, 0)),
        'matplotlib': ('matplotlib>=3.8,<4', (3, 8), (4, 0)),
        'Pillow': ('Pillow>=10,<13', (10, 0), (13, 0)),
        'psutil': ('psutil>=5.9,<8', (5, 9), (8, 0)),
    }
    needed = []
    for distribution, (requirement, lower, upper) in constraints.items():
        try:
            version = tuple(int(p) for p in importlib.metadata.version(distribution).split('.')[:2])
            if not lower <= version < upper:
                needed.append(requirement)
        except (importlib.metadata.PackageNotFoundError, ValueError):
            needed.append(requirement)
    if needed:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *needed],
                       check=True, timeout=min(240, max(1, NOTEBOOK_DEADLINE-time.monotonic()-120)))
    print('Dependencies ready. Elapsed:', round((time.monotonic()-NOTEBOOK_STARTED)/60, 2), 'min')
    ''')
    add('markdown', '''
    ## آموزش و تولید خروجی
    چهار حالت شامل اینرسی کم، حالت پایه، بارگذاری بیشتر و اینرسی بیشتر است.
    هر حالت شبکه‌های مستقل سیال و مسیر پارسل دارد؛ دادهٔ مرجع وارد آموزش کانال نمی‌شود.
    آموزش روی GPU با Adam انجام می‌شود؛ مشتقات معادلات بدون AMP و TF32 محاسبه می‌شوند.
    بهترین checkpoint بر اساس batch ثابت نگه داشته می‌شود، حتی اگر loss خراب شود یا بودجهٔ زمان تمام شود.
    گزارش و شکل‌ها روی CPU تولید می‌شوند و برای این مرحله زمان جدا رزرو شده است.
    ''')
    add('code', '''
    import signal, psutil
    remaining_seconds = NOTEBOOK_DEADLINE - time.monotonic() - 20
    if remaining_seconds < 180:
        raise RuntimeError('زمان کافی برای اجرای کامل باقی نمانده؛ runtime را آماده و notebook را دوباره اجرا کنید.')
    command = [sys.executable, '-u', str(WORK / 'run_gpu.py'),
               '--out', str(OUT), '--validation-pack', str(WORK / 'reference_inputs/Validation_Pack'),
               '--device', 'cuda', '--precision', PRECISION,
               '--max-minutes', str(min(55, remaining_seconds/60)),
               '--train-minutes', str(TRAIN_MINUTES)]
    # نمایش زندهٔ وضعیت؛ ناظر مستقل کل پردازش را پیش از پایان بودجه متوقف می‌کند.
    import threading
    proc = subprocess.Popen(command, cwd=WORK, env=os.environ.copy(),
                            start_new_session=os.name == 'posix',
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    def show_progress():
        for line in proc.stdout:
            print(line, end='', flush=True)
    progress = threading.Thread(target=show_progress, daemon=True)
    progress.start()
    hard_timeout = threading.Event()
    def stop_all_workers():
        if proc.poll() is None:
            hard_timeout.set()
            # کارگرها session مستقل دارند؛ psutil در Colab/Kaggle معمولاً نصب است.
            # مسیر عادی توسط supervisor داخلی هر worker را به‌صورت مستقل محدود می‌کند.
            try:
                import psutil
                for child in psutil.Process(proc.pid).children(recursive=True):
                    try: child.kill()
                    except psutil.NoSuchProcess: pass
            except (ImportError, psutil.NoSuchProcess):
                pass
            if os.name == 'posix':
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            else:
                proc.kill()
    watchdog = threading.Timer(remaining_seconds + 5, stop_all_workers)
    watchdog.daemon = True
    watchdog.start()
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        stop_all_workers()
        returncode = proc.wait(timeout=5)
        print('اجرای دستی متوقف شد؛ فایل‌های ساخته‌شده حفظ شده‌اند.')
    finally:
        watchdog.cancel()
        progress.join(timeout=2)
    print('Exit code:', returncode, '| Elapsed:', round((time.monotonic()-NOTEBOOK_STARTED)/60, 2), 'min')
    print('Hard timeout:', hard_timeout.is_set())
    print('خروج غیرصفر یعنی یک مرحله ناقص/خطادار بوده؛ جزئیات را در گزارش ببینید.')
    ''')
    add('markdown', '## معیارها و جدول نتایج\nپایان اجرا به معنای اعتبار علمی نتیجه نیست؛ خطای باقیماندهٔ معادلات با درصد اختلاف با مقاله متفاوت است.')
    add('code', '''
    from IPython.display import display, HTML, Image, FileLink
    import pandas as pd
    if (OUT / 'summary.json').is_file():
        summary = json.loads((OUT / 'summary.json').read_text())
        print('Workflow complete:', summary['workflow_complete'])
        print('Screening criteria:', summary['research_quality_criteria_met'])
        print('Numerical verification:', summary['verification_passed'], '/', summary['verification_total'])
        print('Literature validation:', summary['physical_validation_cases_run'])
        print('Plot errors:', summary['plot_errors'])
        metrics = OUT / 'paper_metrics.csv'
        if metrics.is_file() and metrics.stat().st_size > 2:
            display(pd.read_csv(metrics))
        display(pd.DataFrame(summary['verification'])[['case_id', 'outcome', 'details']])
    else:
        print('گزارش نهایی ساخته نشده است؛ checkpointها و logهای پوشهٔ خروجی را بررسی کنید:', OUT)
    ''')
    add('markdown', '## شکل‌ها\nهمهٔ شکل‌های ثابت از manifest نمایش داده می‌شوند؛ فایل‌های PNG، PDF و SVG در ZIP هستند. نمای عمق‌دار فقط نمایش حل دوبعدی است.')
    add('code', '''
    import html
    manifest = OUT / 'figure_manifest.json'
    if manifest.is_file():
        for figure in json.loads(manifest.read_text()):
            path = OUT / figure['files'][0]
            if path.suffix == '.png' and path.is_file():
                display(HTML('<h4>' + html.escape(figure['title']) + '</h4>'))
                display(Image(filename=str(path), width=900))
                display(HTML('<p>' + html.escape(figure['caption']) + '</p>'))
    ''')
    add('markdown', '## انیمیشن حرکت و توزیع گاوسی\nمسیرها از پیش‌بینی شبکه می‌آیند؛ ابرهای گاوسی با پهنای ثابت و نرمال‌سازی واقعی کوپل نمایش داده می‌شوند.')
    add('code', '''
    for folder in sorted((OUT / 'cases').glob('*')):
        for name in ['transport.gif', 'gaussian_transport.gif']:
            path = folder / name
            if path.is_file():
                print(folder.name, name)
                display(Image(filename=str(path)))
    ''')
    add('markdown', '''
    ## دریافت خروجی کامل
    ZIP شامل شکل‌ها، انیمیشن‌ها، CSV، میدان‌ها، وزن شبکه‌ها، تنظیمات، logها و کد دقیق اجرای شماست.
    `index.html` گالری کامل آفلاین و `REPORT.md` گزارش است. `completion.json` زمان و مراحل ناقص را ثبت می‌کند.
    آزمون‌های V01–V10 بررسی عددی اجزای مدل‌اند؛ دادهٔ مقاله‌ای نامرتبط به‌عنوان validation این مدل گزارش نمی‌شود.
    ''')
    add('code', '''
    archive = OUT.parent / f'{OUT.name}_results.zip'
    if archive.is_file():
        display(FileLink(str(archive)))
        print('ZIP:', archive)
        try:
            from google.colab import files
            files.download(str(archive))
        except ImportError:
            print('در Jupyter/Kaggle فایل ZIP را از پنل فایل‌ها دریافت کنید.')
        except Exception as exc:
            print('دانلود خودکار انجام نشد؛ ZIP را از پنل فایل‌ها بگیرید:', exc)
    if (OUT / 'completion.json').is_file():
        print(json.dumps(json.loads((OUT / 'completion.json').read_text()), ensure_ascii=False, indent=2))
    ''')
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python'), accelerator='GPU',
        colab=dict(name='GaussianParcelPINN_GPU.ipynb', provenance=[])), nbformat=4, nbformat_minor=5)
    path = root/'GaussianParcelPINN_GPU.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    release = root/'deliverables'/'GaussianParcelPINN_GPU_code.zip'
    release.parent.mkdir(exist_ok=True)
    extras = [root/p for p in ['README.md', 'tests/test_physics.py', 'tests/test_gpu_workflow.py',
                               'docs/GPU_EXECUTED_CHECKS.md', 'docs/GPU_NOTEBOOK_QA.json'] if (root/p).is_file()]
    with zipfile.ZipFile(release, 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in files+[path, root/'build_gpu_notebook.py']+extras:
            write_member(archive, Path('GaussianParcelPINN_GPU')/source.relative_to(root), source.read_bytes())
    release.with_suffix('.zip.sha256').write_text(f'{hashlib.sha256(release.read_bytes()).hexdigest()}  {release.name}\n')
    print(f'{path}: {path.stat().st_size:,} bytes, embedded source SHA256={digest}')
    return notebook


if __name__ == '__main__':
    build(Path(__file__).resolve().parent)
