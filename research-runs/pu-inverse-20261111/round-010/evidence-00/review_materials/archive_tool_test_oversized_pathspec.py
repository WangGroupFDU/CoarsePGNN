"""隔离验证归档路径列表超过ARG_MAX时仍按完整清单提交，不运行PU代码。"""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, '/Users/jifengwang/.codex/skills/codex-chatgpt-loop/scripts')
from test_loopctl import LoopTests
import loopctl as loop
from common import read, hashes, atomic, git


def main():
    """用临时合成仓库调用真实commit_evidence，验证大列表、空格及Unicode路径全集不变。"""
    case=LoopTests();case.setUp()
    try:
        case.computed()
        report=case.project/'report.md';report.write_text('合成归档压力检查，不是PU科学实验')
        manifest=loop.archive(case.root,case.state,{'report':str(report),'files':[],'progress':case.progress(),'local_acceptance':[]})
        manifest_path=case.project/case.state['manifest_path'];prefix=manifest_path.parent
        for i in range(12000):
            name='bulk/'+f'{i:05d}_'+('p'*70)+'_空间.txt'
            file=prefix/name;file.parent.mkdir(exist_ok=True);file.write_text('fixture')
            manifest['files'].append({'name':name,'path':str(file.relative_to(case.project)),'storage':'github','required_read':False,**hashes(file)})
        atomic(manifest_path,manifest);case.state['manifest_sha256']=hashes(manifest_path)['sha256']
        expected=[case.state['manifest_path']]+[x['path'] for x in manifest['files']]
        byte_count=sum(len(x.encode())+1 for x in expected)
        assert byte_count>os.sysconf('SC_ARG_MAX')
        result=loop.commit_evidence(case.root,case.state)
        actual=git(case.project,'diff-tree','--no-commit-id','--name-only','-r','-z',result['artifact_sha']).split('\0')
        assert set(expected)==set(x for x in actual if x)
        assert git(case.project,'diff','--cached','--name-only')==''
        print(json.dumps({'status':'PASSED','synthetic_only':True,'paths':len(expected),'path_bytes':byte_count,'ARG_MAX':os.sysconf('SC_ARG_MAX'),'staged_exact_file_set':True,'real_remote_or_UI_calls':False}))
    finally:
        case.doCleanups()


if __name__=='__main__':main()
