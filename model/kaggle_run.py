"""Ограниченный общим временем запуск с сохранением уже готового submission."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--budget-minutes",type=int,default=330)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--iterations",type=int,default=1800)
    p.add_argument("--skip-foundation",action="store_true")
    p.add_argument("--run-dir",default="runs/kaggle")
    args=p.parse_args()
    root=Path(__file__).resolve().parent;os.chdir(root)
    out=Path(args.run_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();deadline=started+args.budget_minutes*60
    working=Path("/kaggle/working") if Path("/kaggle/working").exists() else root
    # Подготовленный локально кандидат сразу доступен, даже если новая сессия прервется.
    fallback=root/"runs/local/submission.csv"
    if fallback.exists():
        shutil.copyfile(fallback,working/"submission_cpu_reference.csv")
    try:
        import torch
        gpu=torch.cuda.is_available()
    except ImportError:gpu=False
    device="GPU" if gpu else "CPU"
    common=[sys.executable,"-u","-m","ndvi.pipeline","--out",str(out),"--device",device,
            "--iterations",str(args.iterations),"--rounds",str(args.rounds),"--threads","4",
            "--tabicl-context","12000","--tabicl-estimators","4"]
    events=[]
    def stage(name,command,cap_minutes,reserve_minutes=30):
        available=int(deadline-time.monotonic()-reserve_minutes*60)
        timeout=min(int(cap_minutes*60),available)
        if timeout<60:
            events.append({"stage":name,"status":"skipped_budget"});return False
        print(f"\n{name}: максимум {timeout//60} минут",flush=True)
        t=time.monotonic()
        with (out/f"{name}.log").open("a",encoding="utf-8") as log:
            # Вывод модели виден в файле; управление не зависит от readline, которое может зависнуть.
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            stop=time.monotonic()+timeout
            while True:
                try:
                    rc=proc.wait(timeout=min(30,max(.1,stop-time.monotonic())))
                    status="ok" if rc==0 else "failed";break
                except subprocess.TimeoutExpired:
                    if time.monotonic()>=stop:
                        proc.terminate()
                        try:proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
                        status="timeout";rc=-1;break
                    tail=(out/f"{name}.log").read_text(errors="replace").splitlines()[-1:]
                    print(name,"прошло",round((time.monotonic()-t)/60,1),"мин."," ".join(tail),flush=True)
        events.append({"stage":name,"status":status,"returncode":rc,"seconds":time.monotonic()-t})
        (out/"progress.json").write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding="utf-8")
        print(name,status,"log:",out/f"{name}.log",flush=True)
        if status!="ok":print((out/f"{name}.log").read_text(errors="replace")[-2500:],flush=True)
        return status=="ok"
    complete=stage("01_tree_cv",common+["--stage","cv"],100)
    if complete:
        complete=stage("02_tree_final",common+["--stage","final"],45)
    if complete and (out/"submission.csv").exists():
        shutil.copyfile(out/"submission.csv",out/"submission_trees.csv")
        shutil.copyfile(out/"submission.csv",working/"submission_kaggle_trees.csv")
        # Фиксируем весь tree-only результат, чтобы обновление смеси было обратимым.
        for name in ["ensemble_weights.json","model_bundle.pkl"]:
            shutil.copyfile(out/"artifacts"/name,out/"artifacts"/("trees_"+name))
        shutil.copyfile(out/"reports/metrics.json",out/"reports/trees_metrics.json")
        shutil.copyfile(out/"artifacts/submission_manifest.json",out/"artifacts/trees_submission_manifest.json")
    foundation_ok=False
    if complete and gpu and not args.skip_foundation:
        foundation_ok=stage("03_tabicl_cv",common+["--stage","foundation"],100,45)
        if foundation_ok:foundation_ok=stage("04_ensemble_final",common+["--stage","final"],45,20)
    report_path=out/"reports/metrics.json";manifest_path=out/"artifacts/submission_manifest.json"
    consistent=report_path.exists() and manifest_path.exists() and json.loads(report_path.read_text()).get("weights")==json.loads(manifest_path.read_text()).get("weights")
    if complete and consistent:
        stage("05_diagnostics",[sys.executable,"-u","-m","ndvi.diagnostics","--run",str(out)],15,5)
    # Сравнение кандидатов только по development; audit не выбирает победителя.
    candidates=[]
    candidate_files=[("cpu_reference",fallback,root/"runs/local/reports/metrics.json",root/"runs/local/artifacts/submission_manifest.json"),
                     ("kaggle",out/"submission.csv",out/"reports/metrics.json",out/"artifacts/submission_manifest.json"),
                     ("kaggle_trees",out/"submission_trees.csv",out/"reports/trees_metrics.json",out/"artifacts/trees_submission_manifest.json")]
    for name,csv,report,manifest in candidate_files:
        if csv.exists() and report.exists() and manifest.exists():
            r=json.loads(report.read_text());m=json.loads(manifest.read_text())
            if m.get("status")=="complete" and m.get("weights")==r.get("weights"):
                candidates.append((r["development_ensemble"]["rmse"],name,csv,report,manifest))
    if candidates:
        score,name,csv,report,manifest=min(candidates)
        shutil.copyfile(csv,working/"submission.csv")
        # Выбранный CSV должен сопровождаться именно его моделью, даже при CPU fallback.
        selected=working/"ndvi_selected"/time.strftime("%Y%m%d_%H%M%S")
        (selected/"artifacts").mkdir(parents=True,exist_ok=True)
        (selected/"reports").mkdir(parents=True,exist_ok=True)
        model_run=root/"runs/local" if name=="cpu_reference" else out
        prefix="trees_" if name=="kaggle_trees" else ""
        for filename in ["model_bundle.pkl","ensemble_weights.json"]:
            shutil.copyfile(model_run/"artifacts"/(prefix+filename),selected/"artifacts"/filename)
        shutil.copyfile(model_run/"artifacts/run_config.json",selected/"artifacts/run_config.json")
        shutil.copyfile(report,selected/"reports/metrics.json")
        shutil.copyfile(manifest,selected/"artifacts/submission_manifest.json")
        shutil.copyfile(csv,selected/"submission.csv")
        chosen_weights=json.loads(report.read_text())["weights"]
        if chosen_weights.get("tabicl",0)>0:
            shutil.copytree(model_run/"artifacts/tabicl_final",selected/"artifacts/tabicl_final")
        (working/"selected_submission.json").write_text(json.dumps({"candidate":name,
            "development_rmse":score,"official_test_score":"unknown","file":str(csv),
            "selected_model_directory":str(selected)},indent=2),encoding="utf-8")
        print("Выбран:",name,"development RMSE:",score,"Файл:",working/"submission.csv",flush=True)
    else:
        print("Новая полная модель не завершена. Используйте submission_cpu_reference.csv из архива.")
    print("Общее время, минут:",round((time.monotonic()-started)/60,1),flush=True)


if __name__=="__main__":main()
