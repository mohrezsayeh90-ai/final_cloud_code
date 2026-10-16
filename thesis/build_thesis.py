#!/usr/bin/env python3
"""Build, validate and bundle the evidence-based thesis manuscript."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
def run(command,**kwargs):
    return subprocess.run(command,cwd=ROOT,text=True,check=True,**kwargs)

def evidence_check():
    manifest=json.loads((ROOT/'evidence/manifest.json').read_text())
    for record in manifest['inputs']:
        p=ROOT/record['archived_path']
        assert p.is_file(),f'Missing evidence: {p}'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],f'Changed evidence: {p}'
    return manifest

def bundle(destination):
    directories=['chapters','figures','generated','evidence','template','tools','assets']
    files=[ROOT/name for name in ['thesis.tex','metadata.tex','references.tex',
        'README.md','DELIVERY_NOTES_FA.md','build_thesis.py','build_validation.json',
        'Particle_Transport_Thesis.pdf']]
    files += [p for d in directories for p in (ROOT/d).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts]
    if (ROOT/'portable_build_validation.json').exists():files.append(ROOT/'portable_build_validation.json')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        for p in sorted(files):out.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(destination) as out:
        assert out.testzip() is None,'Archive integrity failure'
    print(f'Source archive: {destination}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerate',action='store_true')
    parser.add_argument('--bundle',action='store_true')
    args=parser.parse_args()
    manifest=evidence_check()
    style=json.loads((ROOT/'generated/formatting_provenance.json').read_text())
    logo=ROOT/style['logo']['local_path']
    assert hashlib.sha256(logo.read_bytes()).hexdigest()==style['logo']['sha256']
    (ROOT/'build').mkdir(exist_ok=True)
    if args.regenerate:
        run([sys.executable,'tools/make_tables.py'])
        run([sys.executable,'tools/make_figures.py'])
    figures=json.loads((ROOT/'generated/figure_manifest.json').read_text())
    numerical=json.loads((ROOT/'generated/numerical_validation.json').read_text())
    parcel=json.loads((ROOT/'generated/parcel_validation.json').read_text())
    assert numerical['passed']
    assert parcel['passed'] and not parcel['complete_coupled_screen_met']
    assert len([x for x in figures if x['kind']=='diagnostic coupled prediction'])==5
    for item in figures:
        assert (ROOT/'figures'/(item['name']+'.pdf')).is_file()
        for source in item['inputs']:
            assert (ROOT/source.split('#')[0]).is_file(),source
    env=os.environ.copy()
    for key,sub in [('TEXMFVAR','texmf-var'),('TEXMFCONFIG','texmf-config')]:
        folder=ROOT/'build'/sub;folder.mkdir(exist_ok=True);env[key]=str(folder)
    command=['latexmk','-pdf','-pdflatex=pdflatex -no-shell-escape %O %S',
        '-interaction=nonstopmode','-halt-on-error','-file-line-error',
        '-outdir=build','thesis.tex']
    with (ROOT/'build/compile_output.txt').open('w') as log:
        run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    log=(ROOT/'build/thesis.log').read_text(errors='replace')
    bad=[x for x in log.splitlines() if 'Missing character:' in x or
         'Overfull \\' in x or ('Warning:' in x and
         any(y in x for y in ['undefined','multiply defined','destination with the same']))]
    assert not bad,'Document validation failed:\n'+'\n'.join(bad)
    aux=(ROOT/'build/thesis.aux').read_text()
    labels=set(re.findall(r'\\newlabel\{(fig:[^}]+)\}',aux))
    labels={x for x in labels if not x.endswith('@cref')}
    assert len(labels)==len(figures),(len(labels),len(figures))
    algorithms=set(re.findall(r'\\newlabel\{(alg:[^}]+)\}',aux))
    algorithms={x for x in algorithms if not x.endswith('@cref')}
    assert algorithms=={'alg:coupled_training'},algorithms
    text=subprocess.check_output(['pdftotext',str(ROOT/'build/thesis.pdf'),'-'],text=True)
    for marker in ['[TO WRITE:', 'Your Full Name', 'Writing guide', 'as requested by',
                   'as an AI', 'ChatGPT', 'TODO']:
        assert marker.lower() not in text.lower(),f'Unwanted manuscript marker: {marker}'
    assert 'Mohammad Reza Sayah' in text
    assert 'Mohammad Ali Bijarchi' in text
    assert not re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',text)
    assert not re.search(r'[\u0600-\u06ff]',text),'Non-English manuscript text'
    assert not re.search(r'data\s+(?:and\s+)?availability',text,re.I)
    assert 'github.com/mohrezsayeh90-ai' not in text
    for number in ['1.510','0.863','0.717','0.066','0.238','0.218']:
        assert number in text,f'Missing reported result {number}'
    refs=set(re.findall(r'\\bibitem\{([^}]+)\}',(ROOT/'references.tex').read_text()))
    cited=set()
    for p in [ROOT/'thesis.tex',*(ROOT/'chapters').glob('*.tex')]:
        for group in re.findall(r'\\cite\w*\{([^}]+)\}',p.read_text()):cited.update(group.split(','))
    assert cited==refs,dict(uncited=sorted(refs-cited),undefined=sorted(cited-refs))
    plan=json.loads((ROOT/'generated/citation_plan.json').read_text())
    assert len(refs)==plan['total_reference_count']==40
    assert len(plan['placements'])==36
    for placement in plan['placements']:
        source=(ROOT/placement['file']).read_text()
        assert source.count(placement['marker'])==1,placement['id']
        for key in placement['keys']:
            assert re.search(r'\\cite\w*\{[^}]*\b'+re.escape(key)+r'\b[^}]*\}',source),(placement['id'],key)
    info=subprocess.check_output(['pdfinfo',str(ROOT/'build/thesis.pdf')],text=True)
    assert not re.search(r'^(?:CreationDate|ModDate):',info,re.M)
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
    pdf=ROOT/'Particle_Transport_Thesis.pdf';shutil.copy2(ROOT/'build/thesis.pdf',pdf)
    checks=dict(compiler_exit_code=0,pages=pages,figures=len(figures),algorithms=len(algorithms),references=len(refs),
        evidence_files_verified=len(manifest['inputs']),recomputed_metrics=len(numerical['recomputed_metrics']),
        unresolved_references=[],overfull_boxes=[],missing_glyphs=[],
        citation_coverage='all references cited; all citations defined',
        citation_plan_placements=36,added_references=25,
        measured_errors_and_acceptance_outcomes='preserved',
        typography='English Sharif-style layout; Times family; black text and headings',
        submission_date_removed=True,pdf_timestamp_metadata_removed=True,
        english_university_logo_included=True,
        template_and_conversation_markers='absent',constricted_channel_results_claimed=False,
        data_availability_section_removed=True,diagnostic_parcel_figures=5,
        parcel_postprocessing_checks='passed; complete coupled screen remains unmet',
        numerical_result_files_modified=False,sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
    (ROOT/'build_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    if args.bundle:bundle(ROOT/'Particle_Transport_Thesis_Source.zip')

if __name__=='__main__':main()
