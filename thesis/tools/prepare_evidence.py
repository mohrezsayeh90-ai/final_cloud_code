"""Copy selected numerical evidence without changing the original results."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'basic': (Path('/workspace/basic_particle_problem'), [
        'results/final_study/metrics.json', 'results/final_study/config.json',
        'results/final_study/manifest.json', 'results/final_study/stages.json',
        'results/final_study/failures.json',
        'results/final_study/data/poiseuille.csv',
        'results/final_study/data/poiseuille_pressure.csv',
        'results/final_study/data/field_validation.csv',
        'results/final_study/data/drag_validation.csv',
        'results/final_study/data/mesh_sensitivity.csv',
        'results/final_study/data/seed_summary.csv',
        'results/final_study/data/trajectory_summary.json',
        'results/final_study/data/plotted_fields.npz',
        'results/final_study/data/flow_loss_seed_42.csv',
        'results/final_study/data/force_loss.csv',
        'results/final_study/models/poiseuille.pt',
        'results/final_study/models/force.pt',
        'particle_pinn/models.py', 'particle_pinn/geometry.py',
        'particle_pinn/quality.py', 'run_particle_pinn.py']),
    'gaussian': (Path('/workspace/final_cloud_code'), [
        'outputs/evaluated_2d_final/summary.json',
        'outputs/evaluated_2d_final/config.json',
        'outputs/evaluated_2d_final/runtime.json',
        'outputs/evaluated_2d_final/source_manifest.json',
        'outputs/evaluated_2d_final/reference_manifest.json',
        'outputs/evaluated_2d_final/validation/verification.json',
        'outputs/evaluated_2d_final/validation/V01/comparison.csv',
        'outputs/evaluated_2d_final/validation/V02/comparison.csv',
        'outputs/evaluated_2d_final/validation/V04.csv',
        'outputs/evaluated_2d_final/validation/V03.csv',
        'outputs/evaluated_2d_final/validation/V09.csv',
        'outputs/evaluated_2d_final/validation/V10/comparison.csv',
        'outputs/evaluated_2d_final/validation/literature_audit.csv',
        'outputs/evaluated_2d_final/cases/high_inertia_seed7/fields.npz',
        'outputs/evaluated_2d_final/cases/high_inertia_seed7/particles.csv',
        'outputs/evaluated_2d_final/cases/high_inertia_seed7/diagnostics.csv',
        'outputs/evaluated_2d_final/cases/high_inertia_seed7/case_status.json',
        'outputs/evaluated_2d_final/cases/high_inertia_seed7/training_status.json',
        'parcel_pinn/diagnostics.py',
        'reference_inputs/Validation_Pack/sources.json',
        'parcel_pinn/model.py', 'parcel_pinn/physics.py',
        'parcel_pinn/verification.py', 'parcel_pinn/config.py', 'docs/METHOD.md'])}

def main():
    records = []
    revisions = {}
    for name, (repo, filenames) in SOURCES.items():
        revisions[name] = subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        for filename in filenames:
            source = repo / filename
            destination = ROOT / 'evidence' / name / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            records.append(dict(dataset=name, original_path=filename,
                                archived_path=str(destination.relative_to(ROOT)),
                                sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    (ROOT/'evidence/manifest.json').write_text(json.dumps(dict(
        repository_revisions=revisions, inputs=records,
        operation='Unmodified evidence copies; manuscript figures are redrawn from these arrays.',
        constricted_channel_results_available=False), indent=2)+'\n')
    print(f'Archived {len(records)} numerical and method inputs.')

if __name__ == '__main__':
    main()
