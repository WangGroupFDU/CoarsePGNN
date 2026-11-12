"""最后一次非拟合资格收尾：定点回归、证据分片及不可调参的数据锁。"""
from __future__ import annotations

import collections
import csv
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

from . import TARGET_UNITS
from .io import canonical, digest, read_json, sha256, write_csv, write_json


def read_csv(path: Path) -> list[dict]:
    """读取带BOM且含长来源证据的CSV，明确提高字段上限而不截断来源。"""
    csv.field_size_limit(20_000_000)
    with path.open(encoding="utf-8-sig",newline="") as handle:
        return list(csv.DictReader(handle))


def shard_evidence(source: Path, output: Path, limit: int = 524288) -> dict:
    """按原UTF8换行边界分片并重组验算，绝不重新序列化JSON或丢失字节。

    每片含字节半开区间、行区间、大小和散列；单行超过上限时明确失败。
    仅用于GPT明确要求的文本传输修复，不改变100MiB归档规则或原证据。
    """
    output.mkdir(parents=True)
    blocks,buffer=[],bytearray()
    offset,line_start,line_end=0,1,0
    reconstructed=hashlib.sha256()
    with source.open("rb") as stream:
        for line in stream:
            if len(line)>limit:
                raise ValueError(f"证据单行超过分片上限: {source}:{line_end+1}")
            if buffer and len(buffer)+len(line)>limit:
                blocks.append((bytes(buffer),offset,line_start,line_end))
                offset+=len(buffer);buffer.clear();line_start=line_end+1
            buffer.extend(line);line_end+=1
        if buffer:
            blocks.append((bytes(buffer),offset,line_start,line_end))
    entries=[]
    for index,(block,start,first,last) in enumerate(blocks,1):
        block.decode("utf-8")
        path=output/f"part-{index:03d}.txt"
        with path.open("xb") as handle:
            handle.write(block)
        actual=path.read_bytes();reconstructed.update(actual)
        entries.append({"file":path.name,"order":index,"byte_start":start,"byte_end_exclusive":start+len(block),
                        "line_start":first,"line_end_inclusive":last,"bytes":len(block),"sha256":sha256(path)})
    full_sha=sha256(source)
    if reconstructed.hexdigest()!=full_sha or sum(r["bytes"] for r in entries)!=source.stat().st_size:
        raise AssertionError("原文分片拼接不一致")
    result={"source":str(source),"source_sha256":full_sha,"source_bytes":source.stat().st_size,
            "encoding":"UTF-8","reserialized":False,"max_chunk_bytes":limit,
            "reconstructed_sha256":reconstructed.hexdigest(),"parts":entries}
    write_json(output/"manifest.json",result)
    return result


def protocol_definition(row: dict) -> dict:
    """只从实测协议构建条件与assay身份；无标签、材料ID或来源频次作为模型特征。"""
    from coarsepgnn_hybrid_v3.model import ASSAY_FIELDS
    return {"target":row["target"],
            "condition":{"raw":row["condition_raw"],"normalized":json.loads(row["condition_normalized_json"]),
                         "input_sha256":row["condition_input_sha256"]},
            "assay":{key:row.get(key) for key in ASSAY_FIELDS},
            "scan_identity":row.get("scan_identity"),"peak_identity":row.get("peak_identity"),
            "phase":row.get("transition_scope"),"criterion":row.get("transition_criterion"),
            "thermal_history_stage":row.get("thermal_history_stage")}


def fixed_query_panel(training: list[dict], contract: dict) -> tuple[dict,list[dict],list[str]]:
    """按预注册家族及训练材料数选择A/B，并构造固定96请求，禁止借模型选协议。

    同一完整协议的规范散列决定平局顺序。协议不足两个材料时输出数据失败原因，
    不用另一端点或相域替代；返回注册表、请求和问题列表供封存。
    """
    from .schema import validate_request
    by_target={target:collections.defaultdict(list) for target in TARGET_UNITS}
    definitions={}
    for row in training:
        if row["split"]!="train":
            raise ValueError("协议选择仅接受训练记录")
        target=row["target"]
        if target in {"tg","tm"}:
            expected=("dsc_inflection","soft_segment") if target=="tg" else ("dsc_peak","hard_segment")
            if (row["transition_criterion"],row["transition_scope"])!=expected or ":DSC:" not in row["property_task"]:
                continue
        definition=protocol_definition(row);key=digest(definition)
        definitions[key]=definition;by_target[target][key].append(row)
    registry={};issues=[]
    for target,groups in by_target.items():
        ranked=sorted(groups,key=lambda key:(-len({r['material_id'] for r in groups[key]}),key))
        eligible=[key for key in ranked if len({r['material_id'] for r in groups[key]})>=2]
        if len(eligible)<2:
            issues.append(f"{target}: 指定协议家族只有{len(eligible)}个协议具有至少2个训练材料")
        registry[target]={letter:{"protocol_id":"PROTOCOL_"+key,"canonical_sha256":key,
             "definition":definitions[key],"training_materials":sorted({r['material_id'] for r in groups[key]}),
             "training_observation_ids":sorted(r['observation_id'] for r in groups[key])}
             for letter,key in zip(("A","B"),eligible[:2])}
    requests=[]
    if issues:
        return registry,requests,issues
    targets=list(TARGET_UNITS)
    for size in range(1,5):
        for mask in itertools.combinations(targets,size):
            corners=list(itertools.product((0,2),repeat=size)) if size in (2,4) else \
                    ([(i,) for i in (0,1,2)] if size==1 else [(0,)*size,(2,)*size])
            for corner,letter in itertools.product(corners,("A","B") if size<=2 else ("A",)):
                constraints=[]
                for target,index in zip(mask,corner):
                    tolerance=contract["design_tolerances"][target]
                    center=tolerance["query_centers"][index];delta=tolerance["delta"]
                    protocol=registry[target][letter];definition=protocol["definition"]
                    constraints.append({"target":target,"unit":TARGET_UNITS[target],"lower":center-delta,
                        "upper":center+delta,"protocol_id":protocol["protocol_id"],
                        "condition":definition["condition"],"assay":definition["assay"]})
                request={"schema_version":1,"request_id":f"FINAL_{len(requests)+1:03d}",
                         "constraints":constraints,"nco_oh_ratio":None,"top_k":20,"candidate_budget":512,
                         "protocol_set":letter,"target_count":size}
                requests.append(validate_request(request))
    if len(requests)!=96 or collections.Counter(r['target_count'] for r in requests)!={1:24,2:48,3:8,4:16}:
        raise AssertionError("固定面板分母错误")
    return registry,requests,issues


def affected_graph_replay(config: dict, directory: Path, grades: list[dict]) -> list[dict]:
    """各选择一个恢复的纯酚和混合羟基真实配方，逐模型张量验证既有通用OH构图。

    选取按材料ID排序，与性质值无关。原图文件SHA在读取前校验；完整1162图
    回执只复用，不重复全量featurization。若没有代表性来源则报错并保留缺口。
    """
    import torch
    from coarsepgnn.data import GraphFeaturizer
    from coarsepgnn.schema import package_schema
    from .graph import recipe_to_complete_graph
    observations=read_json(Path(config['preflight'])/'observations.json')
    formulations=read_json(directory/'approved_formulations.json')
    featurizer=GraphFeaturizer.from_sample_schema(package_schema());results=[]
    for name,predicate in [('phenol',lambda r:int(r['phenol_oh_count'] or 0)==2),
                           ('mixed',lambda r:int(r['phenol_oh_count'] or 0)==1 and int(r['alcohol_oh_count'] or 0)==1)]:
        selected={r['grade_id'] for r in grades if r['status']=='APPROVED' and predicate(r)}
        formulas=sorted((f for f in formulations if selected&set(f['grade_ids'])),key=lambda f:f['material_id'])
        if not formulas:
            raise ValueError(f"缺少{name}来源配方供实际图回读")
        formula=formulas[0];row=next(r for r in observations if r['material_id']==formula['material_id'])
        path=Path(row['graph_relpath'])
        if sha256(path)!=row['graph_file_sha256']:
            raise ValueError(f"来源图散列改变: {path}")
        raw=read_json(path)
        rebuilt=recipe_to_complete_graph(raw['components'],formula['nco_oh_ratio'],formula['extender_oh_fraction'])
        before=featurizer.prepare_graph(raw).data.to_dict();after=featurizer.prepare_graph(rebuilt).data.to_dict()
        mismatches=[k for k,v in before.items() if torch.is_tensor(v) and (k not in after or not torch.equal(v,after[k]))]
        results.append({'kind':name,'material_id':row['material_id'],'grade_ids':formula['grade_ids'],
                        'source_graph_sha256':sha256(path),'tensor_fields':sorted(k for k,v in before.items() if torch.is_tensor(v)),
                        'mismatched_tensor_fields':mismatches,'exact_model_input_replay':not mismatches})
        if mismatches:
            raise AssertionError(f"{name}来源图不能精确重放: {mismatches}")
    return results


def close_qualification(config: dict, directory: Path) -> dict:
    """关闭指定三项修正并写数据锁候选；只读训练标签和公开资格表，不打开测试标签。

    data_lock绑定当前C与所有产物散列；包含自身的未来A不能写入自身，需由归档后的
    外部回执与原Pro评审绑定A。状态仍待Pro核准，函数绝不冻结合同或启动拟合。
    """
    previous=Path(config['previous_adjudication']).parent
    adjudication=read_csv(directory/'measurement_adjudication.csv')
    grades=read_csv(directory/'grade_identity_review.csv');old_grades={r['grade_id']:r for r in read_csv(previous/'grade_identity_review.csv')}
    grade_changes=[{'grade_id':r['grade_id'],'old_status':old_grades[r['grade_id']]['status'],
                   'new_status':r['status'],'reason':r['reason'],'names':r['names']} for r in grades
                   if (r['status'],r['reason'])!=(old_grades[r['grade_id']]['status'],old_grades[r['grade_id']]['reason'])]
    write_json(directory/'affected_grade_changes.json',grade_changes)
    replay=affected_graph_replay(config,directory,grades)
    write_json(directory/'restored_hydroxyl_graph_replay.json',replay)
    target_ids={'MEA_0021463de547ba03216a','MEA_91b6c1ec7322be58554c','MEA_bbad6e42b4af1bd65580'}
    thermal=read_csv(directory/'thermal_identity_review.csv')
    events=[r for r in thermal if r['observation_id'] in target_ids]
    if len(events)!=3 or len({(r['phase'],r['peak_identity']) for r in events})!=3 or any(r['same_semantic_key_count']!='1' for r in events):
        raise AssertionError('指定三热事件仍未正确区分')
    old_inputs={r['observation_id']:r for r in read_json(previous/'condition_identity_checks.json')['all_thermal_input_identities']}
    new_inputs=read_json(directory/'condition_identity_checks.json')['all_thermal_input_identities']
    input_changes=[{'observation_id':r['observation_id'],'old':old_inputs[r['observation_id']],'new':r}
                   for r in new_inputs if any(r[k]!=old_inputs[r['observation_id']].get(k) for k in ('input_sha256','prompt_cache_key','property_task','numeric_scan_index'))]
    write_json(directory/'affected_thermal_events.json',{'specified_events':events,'input_changes':input_changes,
        'qualification_decisions':[r for r in adjudication if r['observation_id'] in target_ids]})
    old_u={r['observation_id']:r for r in read_csv(previous/'reported_uncertainty_review.csv')}
    new_u=read_csv(directory/'reported_uncertainty_review.csv')
    u_changes=[{'observation_id':r['observation_id'],'old_type':old_u.get(r['observation_id'],{}).get('type_adjudicated'),**r}
               for r in new_u if r['type_adjudicated']!=old_u.get(r['observation_id'],{}).get('type_adjudicated')]
    unknown=[r for r in new_u if r['reported_type_raw']=='source ± value; SD/SE/range not stated']
    standard=[r for r in new_u if r['reported_type_raw']=='standard_deviation']
    if len(unknown)!=9 or any(r['type_adjudicated']!='unspecified_plus_minus' or r['usable_as_reported_uncertainty']!='False' for r in unknown):
        raise AssertionError('原9条未知±未全部保持类型未知')
    if len(standard)!=2 or any(r['type_adjudicated']!='SD' for r in standard):
        raise AssertionError('原2条standard_deviation未正确正规化')
    write_json(directory/'uncertainty_corrections.json',{'unknown_plus_minus_count':len(unknown),'standard_deviation_count':len(standard),'changes':u_changes})
    training=read_json(directory/'train_labels.json')
    contract=read_json(Path(config['success_contract']))
    registry,requests,issues=fixed_query_panel(training,contract)
    write_json(directory/'protocol_registry.json',registry);write_json(directory/'fixed_96_requests.json',requests)
    denominators=read_json(directory/'evaluation_denominators.json')
    for row in denominators:
        floor=contract['data_policy']['minimum_samples'][row['target']]
        for field,suffix in [('materials','materials'),('groups','closure_groups')]:
            if row[field]<floor[row['split']+'_'+suffix]:
                issues.append(f"{row['target']} {row['split']} {field}: {row[field]} < {floor[row['split']+'_'+suffix]}")
    # 锁定训练配方范围；同一材料多测量不改变极值，也不使用留出配方范围。
    training_materials={r['material_id'] for r in training}
    formulas=[f for f in read_json(directory/'approved_formulations.json') if f['material_id'] in training_materials]
    ranges={key:{'min':min(f[key] for f in formulas),'max':max(f[key] for f in formulas)}
            for key in ('nco_oh_ratio','extender_oh_fraction')}
    whitelist={'model_features':['PSMILES_Morgan2048','SMILES_Morgan2048','two_structure_availability_masks',
        'three_role_indicators','train_fitted_log_molar_mass','functionality_div_2','NONE_indicator'],
        'identity_only':['grade_id'],'excluded_features':['material_id','source_id','sample_id','source_frequency','source_recipe_ratio','node_count'],
        'normalization_fit_grade_ids':read_json(directory/'grade_visibility.json')['inverse_supervised_seen'],
        'supervised_or_label_derived_catalog_features':False}
    write_json(directory/'grade_feature_whitelist.json',whitelist)
    shard_summary={name:shard_evidence(source,directory/name) for name,source in
                   [('round002_training_evidence_shards',previous/'training_evidence.json'),
                    ('round003_training_evidence_shards',directory/'training_evidence.json')]}
    write_json(directory/'evidence_sharding_receipt.json',shard_summary)
    reuse=[]
    for item in config['reuse_evidence']:
        if sha256(Path(item['path']))!=item['sha256']:
            raise ValueError('既有工程证据已变化')
        reuse.append(item)
    write_json(directory/'unchanged_evidence_reuse.json',{'source_artifact_sha':config['previous_artifact_sha'],
        'receipts':reuse,'full_graph_audit_repeated':False,'random_model_forwards_repeated':False,
        'test_access_guard_code_unchanged':True,'reuse_code_sha256':config['unchanged_code']})
    state_old=read_csv(previous/'joint_state_edges.csv');state_new=read_csv(directory/'joint_state_edges.csv')
    old_set={canonical(r) for r in state_old};new_set={canonical(r) for r in state_new}
    write_json(directory/'joint_state_edge_changes.json',{'added':[json.loads(s) for s in sorted(new_set-old_set)],
                                                       'removed':[json.loads(s) for s in sorted(old_set-new_set)]})
    bound_names=['measurement_adjudication.csv','approved_catalog.json','approved_formulations.json','grade_identity_review.csv',
                 'joint_state_review.csv','joint_state_edges.csv','training_measurement_precision.json','grade_visibility.json',
                 'evaluation_denominators.json','split_labels_manifest.json','protocol_registry.json','fixed_96_requests.json',
                 'grade_feature_whitelist.json','evidence_sharding_receipt.json','unchanged_evidence_reuse.json']
    code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    original={r['observation_id']:r for r in read_json(Path(config['preflight'])/'observations.json')}
    qualified=[]
    for row in adjudication:
        if row['decision']!='APPROVED_PRIMARY':
            continue
        parent=original[row['restored_from_mean'] or row['observation_id']]
        item={k:row[k] for k in ('observation_id','material_id','group_id','source_id','split','target','decision')}
        item.update({k:parent[k] for k in ('formulation_key','graph_input_sha256','graph_file_sha256')})
        if (row['split'],row['group_id'])!=(parent['split'],parent['group_id']):
            raise AssertionError('资格收尾改变原集合或闭包')
        qualified.append(item)
    lock={'schema_version':1,'run_id':'pu-inverse-20261111','status':'DATA_CONDITIONS_NOT_MET' if issues else 'PENDING_PRO_APPROVAL',
          'code_sha':code_sha,'previous_artifact_sha':config['previous_artifact_sha'],
          'artifact_binding':'由归档后外部回执绑定本文件SHA与A；禁止自指提交散列',
          'rule_version':config['rule_version'],'rule_code_sha256':sha256(Path('src/coarsepgnn_inverse/adjudication.py')),
          'original_inputs':config['inputs'],'success_contract_numeric_source_sha256':sha256(Path(config['success_contract'])),
          'qualified_observations':qualified,
          'frozen_files':{name:{'sha256':sha256(directory/name),'bytes':(directory/name).stat().st_size} for name in bound_names},
          'split_labels':read_json(directory/'split_labels_manifest.json'),'training_composition_ranges':ranges,
          'cross_split_overlaps':read_json(directory/'post_adjudication_audit.json')['cross_split_overlaps'],
          'data_condition_failures':issues,'optimizer_updates':0,'final_test_window_opened':False,
          'no_test_condition_ablation':'保留用户原计划要求，额外报告，不改动GPT已定稿数值门槛'}
    write_json(directory/'data_lock.json',lock)
    result={'status':lock['status'],'data_lock_sha256':sha256(directory/'data_lock.json'),'issues':issues,
            'grade_changes':len(grade_changes),'thermal_input_changes':len(input_changes),
            'uncertainty_changes':len(u_changes),'fixed_requests':len(requests),'optimizer_updates':0,
            'all_specific_corrections_verified':True,'test_label_file_opened_in_closing_stage':False}
    write_json(directory/'qualification_closure.json',result)
    return result
