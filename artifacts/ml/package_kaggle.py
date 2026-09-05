"""Create a deterministic archive from explicit owned code/data/report paths."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'artifacts/ml/kaggle_ndvi_gpu_v1.zip'


def main():
    selected=[]
    for pattern in ['src/veg_recovery/**/*.py','tests/ml/*.py','configs/ml/*',
                    'artifacts/ml/kaggle/*','artifacts/ml/baseline_v1/*',
                    'artifacts/ml/baseline_v1/bundle/*','reports/experiments.csv',
                    'reports/ml_ablation.md','reports/data_contract_issues.md']:
        selected.extend(p for p in ROOT.glob(pattern) if p.is_file() and p.name!='package_manifest.json')
    for relative in ['data/train_dataset.csv','data/test_data.csv','docs/case_doc.pdf','docs/criteria.pdf',
                     'artifacts/ml/evaluate_baselines.py','artifacts/ml/package_kaggle.py','artifacts/ml/CONTRACT_CHANGELOG.md',
                     'artifacts/ml/test_results.txt']:
        selected.append(ROOT/relative)
    # Kaggle expands the outer upload archive and may transparently unpack or
    # omit nested *.csv.gz files. Baseline OOF tables are reporting artifacts,
    # not training inputs, so keep them out of the portable training dataset.
    selected=sorted(set(path for path in selected if path.suffix != '.gz'))
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in selected}
    manifest=ROOT/'artifacts/ml/kaggle/package_manifest.json'
    manifest.write_text(json.dumps({'schema_version':'1.0','files':hashes},indent=2)+'\n')
    selected.append(manifest)
    with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in selected:
            info=zipfile.ZipInfo('ndvi_project/'+str(path.relative_to(ROOT)),date_time=(2026,9,5,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o644<<16
            archive.writestr(info,path.read_bytes())
    with zipfile.ZipFile(DEST) as archive:
        assert archive.testzip() is None
        for name,digest in hashes.items():
            assert hashlib.sha256(archive.read('ndvi_project/'+name)).hexdigest()==digest
    digest=hashlib.sha256(DEST.read_bytes()).hexdigest()
    DEST.with_suffix('.zip.sha256').write_text(digest+'  '+DEST.name+'\n')
    print(json.dumps({'archive':str(DEST),'bytes':DEST.stat().st_size,'files':len(selected),'sha256':digest}))


if __name__=='__main__': main()
