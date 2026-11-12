"""冻结权重的设计推理：只读取模型资产与原料目录，不加载任何性质标签。"""
from __future__ import annotations

import gc
from pathlib import Path

import torch
from torch_geometric.data import Batch

from coarsepgnn.data import GraphFeaturizer
from coarsepgnn.schema import package_schema
from coarsepgnn_hybrid_v3.model import ASSAY_FIELDS, StructureOnlyRegressor
from condition_encoding_benchmark.qwen import QwenMRL32Encoder, PromptCache, load_prompt_cache, unique_prompts
from condition_encoding_benchmark.v2_core import PLEConfigV2, PiecewiseLinearEncoderV2, prompts_for_record_v2
from pu_v24_benchmark_harness.common import TargetTransform

from . import TARGET_UNITS
from .conditions import tensorize_conditions
from .development import condition_record
from .graph import recipe_to_complete_graph
from .io import canonical, digest, read_json, sha256, write_json
from .networks import ConditionalForward, ConditionalInverse
from .schema import validate_request
from .search import CandidateSearch, recipe_identity
from .training import verify_preprocessing_assets
from .training_data import ConditionInputs


def request_rows(request: dict) -> list[dict]:
    """把每条请求转换为无标签条件行，保留原文、缺失掩码和assay语义。

    condition必须有raw与规范化fields；缺失条件值应在fields中mask=0，而不是
    省略整个协议。若给出input_sha256，必须和当前原文及规范字段一致。
    """
    normalized=validate_request(request);rows=[]
    for index,item in enumerate(normalized['constraints']):
        condition=item['condition']
        if not isinstance(condition.get('raw'),str) or not isinstance(condition.get('normalized'),dict):
            raise ValueError('设计条件必须包含raw文本和normalized对象')
        if not isinstance(condition['normalized'].get('fields'),dict):
            raise ValueError('normalized必须保留fields及每项缺失mask')
        identity=digest({'raw':condition['raw'],'normalized':condition['normalized']})
        if condition.get('input_sha256',identity)!=identity:
            raise ValueError('请求条件hash与实际原文/规范字段不同')
        rows.append({'observation_id':f'request_{index}_{digest(item)[:20]}','split':'design',
            'target':item['target'],'condition_raw':condition['raw'],
            'condition_normalized_json':canonical(condition['normalized']),'condition_input_sha256':identity,
            **{field:item['assay'].get(field,'') for field in ASSAY_FIELDS}})
    return rows


class FrozenDesignModels:
    """一个种子的四个正向模型和逆模型，强制核验调用方固定的checkpoint清单SHA。"""

    def __init__(self, training_root: Path, *, manifest_sha256: str, contract_path: Path,
                 data_lock_path: Path, device: str='cuda') -> None:
        """从完成产物重建模型，拒绝部分拟合、资产变化和未授权设备降级。

        不重新拟合变换、不读取train/validation/test标签；监督来源由原checkpoint
        身份及冻结清单证明。独立评价调用方还须核验最终锁/三种子清单。
        """
        from .runtime_policy import enforce_deterministic_runtime
        owner_config=read_json(training_root/'run_config.json')['config']
        self.runtime=enforce_deterministic_runtime(owner_config['execution_policy'],owner_config['optimizer'])
        if device!='cuda' or not torch.cuda.is_available():
            raise RuntimeError('本项目正式设计推理要求授权远程CUDA')
        self.root,self.device=training_root,torch.device(device)
        if sha256(training_root/'checkpoint_manifest.json')!=manifest_sha256:
            raise ValueError('checkpoint清单与调用方固定版本不一致')
        self.manifest=read_json(training_root/'checkpoint_manifest.json')
        self.identity=self.manifest['identity'];self.config=read_json(training_root/'run_config.json')['config']
        if 'full_config_sha256' in self.identity and digest(self.config)!=self.identity['full_config_sha256']:
            raise ValueError('训练完整配置身份变化')
        result=read_json(training_root/'training_result.json')
        if result['status']!='DEVELOPMENT_FIT_COMPLETED_NOT_PASS' or result['identity']!=self.identity:
            raise ValueError('只有完整开发拟合的匹配权重可以进入设计推理')
        self.contract=read_json(contract_path);self.lock=read_json(data_lock_path)
        if not self.contract['frozen'] or sha256(contract_path)!=self.identity['contract_sha256']:
            raise ValueError('设计合同不同')
        if sha256(data_lock_path)!=self.identity['data_lock_sha256']:
            raise ValueError('设计原料目录数据锁不同')
        verify_preprocessing_assets(training_root,self.identity['preprocessing_receipt_sha256'])
        catalog_path=data_lock_path.parent/'approved_catalog.json'
        if sha256(catalog_path)!=self.lock['frozen_files']['approved_catalog.json']['sha256']:
            raise ValueError('候选原料目录变化')
        if digest(self.runtime)!=self.identity['execution_settings_sha256']:
            raise ValueError('推理实际执行环境与训练身份不同')
        registry_path=data_lock_path.parent/'protocol_registry.json'
        if sha256(registry_path)!=self.lock['frozen_files']['protocol_registry.json']['sha256']:
            raise ValueError('协议注册表变化')
        self.protocol_registry=read_json(registry_path)
        self.catalog=read_json(catalog_path);self.templates=read_json(training_root/'component_templates.json')
        self.features=torch.load(training_root/'catalog_features.pt',map_location='cpu',weights_only=True)
        self.vocab=read_json(training_root/'assay_vocab.json')
        self.sizes=[len(self.vocab[field]) for field in ASSAY_FIELDS]
        self.bounds=read_json(training_root/'inverse_bound_transforms.json')
        self.transforms={t:TargetTransform(**read_json(training_root/(t+'_transform.json'))) for t in TARGET_UNITS}
        self.ple=PiecewiseLinearEncoderV2(PLEConfigV2.from_dict(read_json(training_root/'ple.json')))
        self.prompt_cache=load_prompt_cache(training_root/'text_cache')
        self.featurizer=GraphFeaturizer.from_sample_schema(package_schema())
        self.forward_models={};self.inverse_model=None;self.encoded_requests={};self.scores={}
        self.generator=torch.Generator(device=self.device).manual_seed(self.config['seed'])

    def load_model(self, stage: str):
        """核验字节hash、内部身份、阶段变换后加载；不接受裸state_dict来源不明文件。"""
        path=self.root/stage/'best.pt'
        if sha256(path)!=self.manifest['checkpoints'][stage]['sha256']:
            raise ValueError('checkpoint文件发生变化: '+stage)
        inverse=stage=='inverse_full'
        transform_name='inverse_bound_transforms.json' if inverse else stage.removesuffix('_full')+'_transform.json'
        expected={**self.identity,'stage':stage,'target_transform_sha256':sha256(self.root/transform_name)}
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['identity']!=expected or saved['identity_sha256']!=digest({'identity':expected,'config':self.config['optimizer']}):
            raise ValueError('checkpoint内部身份不一致: '+stage)
        if inverse:
            model=ConditionalInverse(self.features,self.sizes,hidden=self.config['inverse_hidden'],
                components=self.config['mdn_components'],no_target_values=self.config.get('no_target_values',False),
                no_test_conditions=self.config.get('no_test_conditions',False))
        else:
            model=ConditionalForward(StructureOnlyRegressor(target=stage.removesuffix('_full'),schema=package_schema()),self.sizes)
        model.load_state_dict(saved['model'],strict=True)
        return model.to(self.device).eval()

    def prepare_requests(self, requests: list[dict]) -> int:
        """合并一批协议的缺失prompt，一次加载固定文本模型，避免按每条锚点重复加载。

        验证基础模型逐文件SHA后才编码；只扩展当前推理对象内存缓存，不写回训练资产。
        """
        rows=[row for request in requests for row in request_rows(request)]
        prompts=unique_prompts(p for row in rows for p in prompts_for_record_v2(condition_record(row),all_schemes=False))
        missing=[p for p in prompts if p.sha256 not in self.prompt_cache.vectors]
        if not missing:return 0
        raise ValueError('UNSUPPORTED_PROTOCOL_CACHE: 冻结缓存缺少'+str(len(missing))+'个prompt；本轮不重新编码')

    def encode_request(self, request: dict) -> dict:
        """按固定文本缓存与train-only PLE编码请求，目标数值不进入prompt。

        未知assay按训练词表规定的0索引处理；不重新拟合任何统计量或类别。
        """
        request=validate_request(request);key=digest(request)
        if key in self.encoded_requests:return self.encoded_requests[key]
        missing_count=self.prepare_requests([request]);rows=request_rows(request)
        payload=tensorize_conditions(rows,self.prompt_cache,self.ple)
        conditions=ConditionInputs(payload,rows,self.vocab).take([r['observation_id'] for r in rows])
        result={'rows':rows,'conditions':conditions,'new_text_prompts':missing_count}
        self.encoded_requests[key]=result
        return result

    def inverse_inputs(self, request: dict, ratio: float) -> dict:
        """按冻结训练变换构造[1,J]目标集合，显式区分上下界缺失和真实数值0。"""
        encoded=self.encode_request(request);constraints=request['constraints'];bounds=[];masks=[]
        for item in constraints:
            transform=self.bounds[item['target']];values=[item.get('lower'),item.get('upper')]
            masks.append([v is not None for v in values])
            bounds.append([(v-transform['mean'])/transform['std'] if v is not None else 0.0 for v in values])
        out={key:value.unsqueeze(0).to(self.device) for key,value in encoded['conditions'].items()}
        out.update(property_ids=torch.tensor([[list(TARGET_UNITS).index(c['target']) for c in constraints]],device=self.device),
                   bounds=torch.tensor([bounds],dtype=torch.float32,device=self.device),
                   bound_mask=torch.tensor([masks],dtype=torch.bool,device=self.device),
                   target_mask=torch.ones(1,len(constraints),dtype=torch.bool,device=self.device),
                   nco_oh_ratio=torch.tensor([ratio],dtype=torch.float32,device=self.device))
        return out

    @torch.no_grad()
    def propose(self, request: dict, ratio: float, count: int) -> list[dict]:
        """按指定R抽样原料及v，保留生成概率/MDN权重，不能解释为性质达标概率。"""
        if self.inverse_model is None:
            self.inverse_model=self.load_model('inverse_full')
        context=self.inverse_model.encode(self.inverse_inputs(request,ratio))
        allowed=None
        if 'allowed_grades' in request:
            from . import ROLES
            allowed=self.features['role_masks'].clone().to(self.device)
            for role,names in request['allowed_grades'].items():
                allowed[ROLES.index(role)] &= torch.tensor([g in names for g in self.features['grade_ids']],device=self.device)
        proposals=self.inverse_model.designer.sample_proposals(context,count,self.generator,allowed_masks=allowed)
        return [{'grade_ids':[self.features['grade_ids'][i] for i in p['grade_indices']],
                 'nco_oh_ratio':ratio,'extender_oh_fraction':p['extender_oh_fraction'],
                 'generation':{k:v for k,v in p.items() if k not in ('grade_indices','extender_oh_fraction')},
                 'property_success_probability':None} for p in proposals]

    def graph(self, recipe: dict) -> dict:
        """在D/P/E角色顺序下传递连续R/v构图，NONE严格不创建扩链剂组分。"""
        return recipe_to_complete_graph([self.templates[g] for g in recipe['grade_ids'] if g!='NONE'],
            recipe['nco_oh_ratio'],recipe['extender_oh_fraction'])

    @torch.no_grad()
    def score(self, request: dict, recipes: list[dict]) -> list[dict]:
        """按每条指定协议计算每配方预测；重复缓存命中仍由外层消耗逻辑预算。

        actual_task_calls计实际候选-性质预测数，model_forward_batches另计批次调用。
        构图失败逐项返回原因并消耗提案槽；模型维度等实现错误直接抛出，不替换模型。
        """
        encoded=self.encode_request(request);conditions=encoded['conditions'];request_key=digest(request)
        result=[None]*len(recipes);pending=[];known={}
        for index,recipe in enumerate(recipes):
            key=(request_key,recipe_identity(recipe))
            if key in self.scores:
                result[index]={**self.scores[key],'actual_task_calls':0,'model_forward_batches':0,'cache_hit':True,'graph_constructions':0}
            elif key in known:
                pending.append((index,key,None,known[key]))
            else:
                try:
                    graph=self.featurizer.prepare_graph(self.graph(recipe)).data
                except (ValueError,KeyError) as error:
                    result[index]={'error':str(error),'actual_task_calls':0,'model_forward_batches':0,'cache_hit':False,'graph_constructions':1}
                    continue
                known[key]=index;pending.append((index,key,graph,None))
        unique=[row for row in pending if row[2] is not None]
        for offset in range(0,len(unique),16):
            part=unique[offset:offset+16];batch=Batch.from_data_list([r[2] for r in part]).to(self.device)
            values=[[] for row in part];batches=[0]*len(part)
            for j,item in enumerate(request['constraints']):
                target=item['target']
                if target not in self.forward_models:
                    self.forward_models[target]=self.load_model(target+'_full')
                inputs={'graph':batch,**{k:v[j:j+1].expand(len(part),*v.shape[1:]).to(self.device) for k,v in conditions.items()}}
                raw=self.transforms[target].inverse(self.forward_models[target](inputs).cpu().numpy())
                batches[0]+=1
                for i,value in enumerate(raw): values[i].append(float(value))
            for (index,key,_,_),prediction,batch_count in zip(part,values,batches):
                entry={'predictions':prediction,'actual_task_calls':len(request['constraints']),
                       'model_forward_batches':batch_count,'cache_hit':False,'graph_constructions':1}
                result[index]=entry;self.scores[key]=entry
        for index,key,graph,original in pending:
            if graph is None:
                result[index]={**result[original],'actual_task_calls':0,'model_forward_batches':0,'cache_hit':True,'graph_constructions':0}
        return result

    @torch.no_grad()
    def generation_window_scores(self,request: dict,recipes: list[dict]) -> list[float]:
        """按固定v±0.005窗口的条件配方概率评分，不调用前向性质模型。

        供无前向重排消融在相同候选域内确定生成倾向；R是外部给定/分层条件，
        不虚构P(R)。概率下限与真实锚点评分相同，结果不是性质达标概率。
        """
        from .metrics import inverse_anchor_score
        if self.inverse_model is None:self.inverse_model=self.load_model('inverse_full')
        grade_index={g:i for i,g in enumerate(self.features['grade_ids'])};scores=[]
        for start in range(0,len(recipes),64):
            part=recipes[start:start+64];template=self.inverse_inputs(request,part[0]['nco_oh_ratio'])
            batch={k:v.expand(len(part),*v.shape[1:]) for k,v in template.items()}
            batch['nco_oh_ratio']=torch.tensor([r['nco_oh_ratio'] for r in part],dtype=torch.float32,device=self.device)
            batch['choices']=torch.tensor([[grade_index[g] for g in r['grade_ids']] for r in part],dtype=torch.long,device=self.device)
            batch['v_reference']=torch.tensor([r['extender_oh_fraction'] for r in part],dtype=torch.float64,device=self.device)
            batch['v']=batch['v_reference'].float()
            result=inverse_anchor_score(self.inverse_model,batch)
            scores.extend((-result['nll']).cpu().tolist())
        return scores
