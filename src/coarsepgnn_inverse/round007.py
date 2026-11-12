"""第7轮单个外层任务：只读D0、三套新训练、独立重载及固定开发设计比较。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .evaluation import select_configuration
from .io import digest, read_json, sha256, write_json


def run(plan_path: Path, output: Path) -> dict:
    """共享一个硬截止时间，任一子进程失败立即停止并保留已有文件及真实进度。

    子进程隔离训练、F4和评分对象；原始D0文件从不写入。外层plan、当前代码、
    配置及静态用例SHA均由提交之后生成的授权回执核准，不形成提交自引用。
    """
    started=time.monotonic();plan=read_json(plan_path);deadline=started+plan['timeout_seconds']
    if output.exists():raise FileExistsError(output)
    authorization=read_json(Path(plan['authorization']))
    current=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if current!=authorization['training_code_sha'] or sha256(plan_path)!=authorization['plan_sha256']:
        raise PermissionError('第7轮整轮计划或源码身份不匹配')
    for name,value in authorization['static_inputs'].items():
        if sha256(Path(name))!=value:raise PermissionError('冻结静态输入改变: '+name)
    output.mkdir(parents=True);events=[]
    def execute(name: str, arguments: list[str]) -> None:
        """每个子进程只获得同一截止时间的剩余秒数，完整日志逐阶段保留。"""
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('第7轮共享14400秒已耗尽')
        begin=time.monotonic();command=[sys.executable,*arguments]
        with (output/(name+'.log')).open('x') as stream:
            process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
        event={'stage':name,'command':command,'returncode':process.returncode,'elapsed_seconds':time.monotonic()-begin,
               'remaining_round_seconds':deadline-time.monotonic()}
        events.append(event)
        with (output/'progress.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
        print(event,flush=True);process.check_returncode()
    try:
        execute('unit_tests',['-m','pytest','-q',*plan['tests'],'--junitxml='+str(output/'unit_tests.xml')])
        results=[]
        for item in plan['configurations']:
            name=item['config_id'];root=Path(item['model_dir']);config_path=Path(item['config_path'])
            if name!='D0':
                execute(name+'_train',['-m','coarsepgnn_inverse','train','--config',str(config_path),'--output',str(root),
                                      '--deadline-monotonic',str(deadline)])
                execute(name+'_reload',['-m','coarsepgnn_inverse.reload_validation','--config',str(config_path),
                                       '--training-dir',str(root),'--output',str(root/'independent_validation.json')])
            manifest_sha=sha256(root/'checkpoint_manifest.json')
            if name=='D0' and manifest_sha!=item['manifest_sha256']:raise ValueError('D0生产者清单改变')
            args=['-m','coarsepgnn_inverse','evaluate','--config',str(config_path),'--model-dir',str(root),
                  '--manifest-sha256',manifest_sha,'--output',str(output/name/'evaluation')]
            if name=='D0':args+=['--readonly-producer',plan['readonly_producer']]
            execute(name+'_evaluate',args)
            results.append({**item,'manifest_sha256':manifest_sha,
                            'summary':read_json(output/name/'evaluation'/'evaluation_summary.json')})
        selected=select_configuration(results);write_json(output/'configuration_selection.json',selected)
        winner=next(r for r in results if r['config_id']==selected['selected_config_id'])
        command=['-m','coarsepgnn_inverse.round007','--search-only','--config',winner['config_path'],
                 '--model-dir',winner['model_dir'],'--manifest-sha256',winner['manifest_sha256'],
                 '--panel',plan['development_panel'],'--cases',plan['interface_cases'],'--output',str(output/'search')]
        execute('selected_development_search',command)
        result={'status':'ROUND007_COMPLETED_NOT_SCIENTIFIC_PASS','events':events,
                'selected_config_id':selected['selected_config_id'],'elapsed_seconds':time.monotonic()-started,
                'test_access':False,'formal96_performance_computed':False}
        write_json(output/'round007_result.json',result);return result
    except BaseException as error:
        write_json(output/'round007_failure.json',{'status':'FAILED','error':repr(error),'events':events,
                   'elapsed_seconds':time.monotonic()-started,'configuration_grid_complete':False})
        raise


def main() -> None:
    """调度入口与独立搜索子进程共享源码；没有训练失败自动重试或预算重置。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--search-only',action='store_true');parser.add_argument('--config',type=Path)
    parser.add_argument('--model-dir',type=Path);parser.add_argument('--manifest-sha256')
    parser.add_argument('--panel',type=Path);parser.add_argument('--cases',type=Path)
    args=parser.parse_args()
    if args.search_only:
        from .development_search import evaluate_search
        result=evaluate_search(read_json(args.config),args.model_dir,args.manifest_sha256,args.panel,args.cases,args.output)
    else:result=run(args.plan,args.output)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
