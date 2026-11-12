"""只整理已核验的R8文件及原有指标，不启动模型或重新计算科学评价。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET

root=Path.cwd();run=root/'.codex-chatgpt-loop/pu-inverse-20261111'
stage=Path('/tmp/pu_inverse_round008_archive');review=Path('/tmp/pu_inverse_round008_review')

def read(path):
    """读取归档边界JSON；文件缺失或格式错误直接停止。"""
    return json.loads(path.read_text())

def sha(path):
    """计算原文件SHA-256，以匹配正式运行和独立回读清单。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()

inventory=read(run/'inputs/round008_inventory.json')
download=read(run/'inputs/round008_download_receipt.json')
assert download['all_size_sha256_verified'] and download['files']==len(inventory)
for item in inventory:
    p=stage/item['name'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['name']
state=read(run/'state.json');C=state['code_sha']
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==C
assert not subprocess.check_output(['git','status','--short'],text=True).strip()
result=read(stage/'round008_result.json');execution=read(run/'inputs/round008_collect.json')['result']
assert result['status']=='ROUND008_COMPLETED_NOT_SCIENTIFIC_PASS'
assert execution['status']=='passed' and execution['provenance_valid']
assert len(execution['outputs'])==596 and all('sha256' in x for x in execution['outputs'])
for name in ['source_review_bundle.md','source_review_manifest.json']:
    shutil.copyfile(review/name,stage/name)
for name in ['round008_inventory.json','round008_inventory_receipt.json','round008_download_receipt.json','round008_transfer_adjustment.json','round008_deployment_receipt.json','round008_resource_check.json']:
    shutil.copyfile(run/'inputs'/name,stage/name)
for p in sorted(Path('configs/inverse').glob('*round008*.json')):
    dest=stage/'frozen_configs'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
for name in ['round007_development_panel.json','round007_interface_cases.json','round007_readonly_producer.json']:
    shutil.copyfile(Path('configs/inverse')/name,stage/'frozen_configs'/name)
shutil.copyfile(Path('configs/inverse/success_contract.json'),stage/'frozen_success_contract.json')

verification=[]
for path in sorted(stage.glob('*/training/*/resume_transition.json')):
    receipt=read(path);completion=read(path.parent/'completion.json')
    assert receipt['status']=='PASSED' and receipt['formal_state_restored'] and receipt['checkpoint_unchanged']
    assert all(v['status']=='passed' for v in receipt['checks'].values())
    assert all(v<=1e-5 for v in receipt['max_scaled_errors'].values())
    receipt_summary={k:v for k,v in receipt.items() if k not in ('audit_batch','checkpoint_identity')}
    verification.append({'stage':str(path.parent.relative_to(stage)),
        'resume_receipt_sha256':sha(path),'resume_transition':receipt_summary,
        'completion_sha256':sha(path.parent/'completion.json'),
        'best_epoch':completion.get('best_epoch'),'formal_optimizer_updates':completion['formal_optimizer_updates'],
        'actual_epochs':completion['history'][-1]['epoch']})
assert len(verification)==50,len(verification)
(stage/'training_verification_summary.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
main_names={'20261111':'D0','20261112':'main_20261112','20261113':'main_20261113'}
main={seed:read(stage/name/'evaluation/evaluation_summary.json') for seed,name in main_names.items()}
paired=read(stage/'no_target_paired_validation.json')
search=read(stage/'search/development_search_summary.json')
ranking=read(stage/'last_ranking/shared_pool_summary.json')
selection=read(stage/'configuration_selection.json')
search_summary={}
for method in ['inverse','retrieval','legal_random','combination_ratio_search']:
    rows=[r for r in search['per_request'] if r['method']==method]
    search_summary[method]={'requests':len(rows),'single_proxy_hits':sum(r['hit_at_20_single_model'] for r in rows),
        'proposal_slots':sum(r['accounting']['proposal_slots'] for r in rows),
        'logical_search_vectors':sum(r['accounting']['logical_search_vectors'] for r in rows)}
suite=ET.parse(stage/'unit_tests.xml').getroot()
tests=sum(int(s.attrib.get('tests',0)) for s in suite.findall('testsuite'))
failures=sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suite.findall('testsuite'))
assert tests==56 and failures==0
summary={'round':8,'code_sha':C,'status':'ENGINEERING_COMPLETE_SCIENTIFIC_NOT_PASSED',
 'formal_round_wall_seconds':execution['elapsed_seconds'],'cumulative_compute_seconds':state['compute_seconds'],
 'cumulative_rounds':state['round'],'remaining_rounds':10-state['round'],
 'tests_passed':tests,'new_fitting_stages':result['new_fitting_stages'],
 'formal_optimizer_updates':result['updates']['formal_optimizer_updates'],
 'diagnostic_optimizer_updates':result['updates']['diagnostic_optimizer_updates'],
 'selected_config_id':selection['selected_config_id'],'selection_lock_sha256':sha(stage/'selection_lock.json'),
 'three_seed_validation':{seed:{'forward':v['forward'],'inverse_nll':v['inverse']['four_target_mean_nll'],
   'single_target_mean_improvement_nats':paired[seed]['single']['four_target_mean_improvement_nats'],
   'joint_state_mean_improvement_nats':paired[seed]['joint']['mean_improvement_nats']} for seed,v in main.items()},
 'development_panel':{'single_target_requests':16,'two_target_UTS_EAB_requests':8,'results':search_summary,
   'last_ranking_single_proxy_hits':sum(r['hit_at_20_single_model'] for r in ranking['per_request']),
   'contains_forward_guided_proposals':True,'three_proxy_consensus_computed':False},
 'interface_cases_passed':len(search['interface']['cases']),
 'test_access':False,'formal96_performance_computed':False,
 'remote_inventory_files':len(inventory),'remote_inventory_bytes':sum(r['bytes'] for r in inventory),
 'largest_file_bytes':max(r['bytes'] for r in inventory),'all_downloads_size_sha256_verified':True,
 'scientific_limits':['仅验证集结果；拉伸强度及断裂伸长率MAE仍较大。','三个种子的目标数值条件增益不稳定，不能声称满足合同。',
 '开发24请求单代理主模型23命中；组合比例搜索24命中，不能宣称主模型优于基线。',
 '实际三/四目标共同状态为0；接口四目标和未来固定挑战面板均不是四性质联合实测验证。']}
(stage/'round_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
rows=[]
for seed,value in main.items():
    f=value['forward'];a=[f['targets'][t]['mae'] for t in ['tg','tm','tensile_strength_ultimate','elongation_at_break']]
    rows.append('| '+seed+' | '+' | '.join(f'{x:.6f}' for x in a)+f" | {f['normalized_score']:.6f} | {paired[seed]['single']['four_target_mean_improvement_nats']:.6f} |")
grid=[]
for name in ['D0','D1','D2','D3']:
    v=read(stage/name/'evaluation/evaluation_summary.json')
    grid.append(f"| {name} | {v['forward']['normalized_score']:.9f} | {v['inverse']['four_target_mean_nll']:.9f} |")
(stage/'report.md').write_text(f'''# 第八轮：端到端开发训练、三种子与配对消融已完成，尚未科学PASS

代码 C8 `{C}`；本轮正式墙钟 {execution['elapsed_seconds']:.6f} 秒，累计 8/10 轮、{state['compute_seconds']:.6f}/259200 秒。全部596项预期输出存在，执行身份有效。新增50个拟合阶段、13375次正式参数更新及100次明确分列的恢复诊断更新；56项工程测试通过。最终测试标签和预注册96请求面板性能仍封存。

## 四配置选择与版本

| 配置 | 验证S（越小越好） | 源float64逆向NLL / nat |
|---|---:|---:|
{chr(10).join(grid)}

完整四配置完成后，按原规则暂选D0；96组开发比较、18个接口用例及真实图/计量/合成来源回读均通过后写入不可覆盖的 selection_lock。D0仍由C6 `94f8f785d5c3bfb4197903d5588e7a5490d3e80d` / A6 `cab04775f48555995ed448deffe0e0c1bfa57029` 生产；C8仅作只读消费。另两个主种子和全部逆向消融由C8生产。没有改选、第五套超参数、重新拟合预处理或重新编码文本。

## 三种子验证结果

| 种子 | Tg MAE / K | Tm MAE / K | UTS MAE / MPa | EAB MAE / 百分点 | S | 无目标对照减主模型NLL / nat |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

这些是固定验证集结果，不是最终测试验收。强度与伸长误差较大；前两个种子的条件目标数值没有带来正的四目标平均增益。不能以第三个种子的正结果代替三种子标准，也不能据此更改冻结合同。单目标分母为239观察，Tg/Tm/UTS/EAB各36/51/73/79观察、18/29/73/79材料；所有来源闭包统计和未见规格分层保留在逐目标结果。63个核准双目标状态的配对改进另行报告，三/四目标实测锚点仍为0。

## 可复核工程与消融

全部新拟合均保存同一最佳epoch的模型和AdamW状态，并保存可续跑last；50份真实恢复诊断、实际更新数及独立重载结果保留。新模型F4及旧D0只读F4逐观察预测/分布重算差异为0。实际K=1、目标数值屏蔽、测试条件屏蔽均通过共享工厂与输入不变性核验，逆向消融没有重训对应前向模型。

D0请求重建依据原C6源码及已归档last.pt，重建35个epoch/245个正式batch；原epoch3审计和最终CPU/loader随机状态完全相同，重建本身0次优化器更新和0次模型前向。其连续目标边界是确定性重建，并非旧运行逐batch独立观察；这一边界在原回执明确保留。所有配对消融使用对应主模型的完整请求轨迹和实际完成更新预算，未用best_epoch代替。

真实训练图张量完全一致；真实图和候选图预测均为54.11520197602607，v扰动进入张量并得到54.1278716258221。NONE及R/v/质量分数回算通过。来源原文可追溯，但尚未结构化的温度、时间、溶剂、催化剂保持null，新组合不冒充实验验证。

## 开发候选与基线

固定开发24请求仅含16单目标与8个UTS/EAB双目标。按各自搜索代理，主模型23/24、合法随机23/24、组合/比例搜索24/24、训练配方检索15/24命中。主模型、随机和组合搜索各12288逻辑评分，检索因历史配方耗尽实际2656；均无免费额外探索。本结果不显示主模型优于最强基线，也不是正式96面板或三代理一致性指标。

相同完整512提案池的末端生成概率排序为17/24命中。该池含主流程前向指导的细化，故只评价最后排序，不声称完全移除了前向作用。K=1与无测试条件权重和源精度真实锚点评分均保留；正式面板上的消融命中和多样性尚待最终窗口。

## 归档与下一轮请求

独立远程清单及本地逐项大小/SHA-256回读覆盖{len(inventory)}个文件、{sum(r['bytes'] for r in inventory)}字节；最大单文件{max(r['bytes'] for r in inventory)}字节，均低于100 MiB。完整清单以文件传输，避免SSH stdout长度上限造成采样；原字节和远程全部产物保留。归档遵循GitHub规则，本轮无需Zenodo。

请审阅C8/A8、选择锁、三种子与配对轨迹和真实接口证据。若工程证据足够，请明确授权剩余第9轮仅实现并执行冻结的最终评价窗口（建议上限14400秒）：不再训练、不增加配置，锁定现有三主种子及消融权重、源精度评分器、候选与基线策略和最终代码；按原data_lock的96请求及A/B协议计算三种子四方法与所需消融，对全部250个测试观察及106个核准双目标状态作同口径核验。按原合同报告全部门槛，无论结果是否成功均不再调参。

建议在第9轮打开测试前一并冻结最终评价和第10轮独立重算两个入口的源码；第10轮不得修改计算实现。第10轮仅对已锁定的最终窗口独立重载归档权重，重算全部留出观察和全部最终候选、合法性及门槛一致性（上限14400秒）；这属于同一冻结窗口的审计，不是另开测试调参窗口。请裁定具体一次性窗口状态和失败恢复规则，并明确K=1、无测试条件及末端排序在正式96面板上的计算/复核口径。当前不请求科学PASS；验证负结果必须保留，成功标准不变。
''')
required={'round_summary.json','training_verification_summary.json','source_review_bundle.md','source_review_manifest.json',
 'round008_result.json','selection_lock.json','configuration_selection.json','final_evaluation_readiness.json',
 'no_target_paired_validation.json','round008_authorization.json','round008_inventory_receipt.json',
 'round008_download_receipt.json','round008_deployment_receipt.json','round008_resource_check.json',
 'frozen_success_contract.json','unit_tests.xml','unit_tests.log',
 'D0_reconstructed_requests/reconstruction_receipt.json','search/development_search_summary.json',
 'search/adapter/adapter_replay.json','search/adapter/graph_tensor_replay.json','search/adapter/stoichiometry_readback.json',
 'search/adapter/synthesis_source_readback.json','search/interfaces/interface_summary.json','last_ranking/shared_pool_summary.json'}
remote={i['name']:i['path'] for i in inventory};files=[]
for p in sorted(stage.rglob('*')):
    if not p.is_file() or p.name=='report.md':continue
    name=str(p.relative_to(stage));need=(name in required or name.startswith('frozen_configs/') or p.name in ['evaluation_summary.json','independent_validation.json','readonly_compatibility.json','variant_validation.json'])
    item={'source':str(p),'name':name,'required_read':need}
    if name in remote:item['source_output']=remote[name]
    files.append(item)
archive={'report':str(stage/'report.md'),'files':files,'local_acceptance':[
 {'id':'A1','status':'pass','evidence':'沿用C4/A4资格锁；全部开发入口实际测试读取为0；原料/来源/图与固定划分未变。'},
 {'id':'A2','status':'pass','evidence':'冻结合同SHA e0d18e2181c800ec077a5c58800e66fefb67d99609bcf83a844eb204a2c292ef 保持。'},
 {'id':'A3','status':'partial','evidence':'50新拟合阶段和三主种子/配对对照完成；验证存在负结果；正式测试与96面板尚未执行，不能科学PASS。'},
 {'id':'A4','status':'pass','evidence':'56工程测试、18接口用例、真实图张量/前向及R/v/NONE计量与来源回读通过；计算命中与实测验证分开。'},
 {'id':'A5','status':'partial','evidence':'596正式预期输出及3114完整产物已独立哈希核验；本轮等待原Pro C8/A8评审，研究未PASS。'}]}
(run/'inputs/archive_round008.json').write_text(json.dumps(archive,ensure_ascii=False,indent=2)+'\n')
print({'files':len(files),'required_read':sum(f['required_read'] for f in files),'raw_bytes':summary['remote_inventory_bytes'],'resume_receipts':len(verification),'remaining_rounds':summary['remaining_rounds']})
