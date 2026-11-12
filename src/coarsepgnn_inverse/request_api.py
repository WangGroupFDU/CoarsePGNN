"""版本化协议解析和可判定错误状态；新协议须先登记，不能悄悄套默认条件。"""
from __future__ import annotations

import copy

from .io import digest
from .schema import validate_request


class DesignInputError(ValueError):
    """保留机器可读状态，同时保持现有ValueError调用接口。"""

    def __init__(self, status: str, reason: str):
        """只携带明确错误语义，不把搜索失败混作物理不可行。"""
        super().__init__(reason)
        self.status=status


def resolve_design_request(request: dict, registry: dict) -> dict:
    """核准协议ID及完整定义，显式记录默认A；再区分非法输入与可证明矛盾。

    registry来自冻结protocol_registry及调用方SHA校验。外部设计只接收已登记协议；
    独立实测锚点评分使用其已核准原始条件，不经过此外部用户输入解析器。
    """
    if not isinstance(request,dict) or not isinstance(request.get('constraints'),list) or not request['constraints']:
        raise DesignInputError('INVALID_REQUEST','constraints必须为非空列表')
    result=copy.deepcopy(request);resolved=[]
    for item in result['constraints']:
        if not isinstance(item,dict) or item.get('target') not in registry:
            raise DesignInputError('INVALID_REQUEST','未知目标或约束格式')
        entries=registry[item['target']];by_id={row['protocol_id']:row for row in entries.values()}
        protocol=item.get('protocol_id');used_default=False
        if protocol is None:
            if 'condition' in item or 'assay' in item:
                raise DesignInputError('INVALID_REQUEST','部分协议不能当作完全缺省协议')
            protocol=entries['A']['protocol_id'];used_default=True
        if protocol not in by_id:
            raise DesignInputError('UNSUPPORTED_PROTOCOL','协议未登记: '+str(protocol))
        record=by_id[protocol];definition=record['definition']
        if digest(definition)!=record['canonical_sha256']:
            raise ValueError('冻结协议注册表身份不一致')
        if item.get('definition_sha256',record['canonical_sha256'])!=record['canonical_sha256']:
            raise DesignInputError('INVALID_REQUEST','请求协议SHA与注册版本不同')
        for key,value in definition.items():
            if key in item and item[key]!=value:
                raise DesignInputError('INVALID_REQUEST','协议ID与实际定义不同: '+key)
            item[key]=copy.deepcopy(value)
        item['protocol_id']=protocol
        resolved.append({'protocol_id':protocol,'used_default':used_default,
                         'default_name':'A' if used_default else None,'definition_sha256':record['canonical_sha256']})
        # 先检查每条约束自身合法，只有随后同协议的交集为空才称约束不可行。
        try: validate_request({**result,'constraints':[item]})
        except (ValueError,KeyError,TypeError) as error: raise DesignInputError('INVALID_REQUEST',str(error)) from error
    limits={}
    for item in result['constraints']:
        key=(item['target'],item['protocol_id']);lower=item.get('lower');upper=item.get('upper')
        low,high=limits.get(key,(None,None))
        low=lower if low is None else low if lower is None else max(low,lower)
        high=upper if high is None else high if upper is None else min(high,upper)
        if low is not None and high is not None and low>high:
            raise DesignInputError('INFEASIBLE_CONSTRAINTS','同一性质及协议的区间没有交集')
        limits[key]=(low,high)
    try: result=validate_request(result)
    except (ValueError,KeyError,TypeError) as error: raise DesignInputError('INVALID_REQUEST',str(error)) from error
    result['resolved_protocols']=resolved
    return result


def validate_grade_domain(request: dict, catalog: list[dict]) -> None:
    """在模型调用前检查用户限定的原料规格，空域与未知规格具有不同状态。"""
    from . import ROLES
    filters=request.get('allowed_grades',{})
    if not isinstance(filters,dict) or not set(filters)<=set(ROLES):
        raise DesignInputError('INVALID_REQUEST','allowed_grades只能按D/P/E完整角色名分组')
    for role in ROLES:
        known={r['grade_id'] for r in catalog if r['role']==role}|({'NONE'} if role==ROLES[2] else set())
        names=filters.get(role,sorted(known))
        if not isinstance(names,list) or any(not isinstance(n,str) for n in names):
            raise DesignInputError('INVALID_REQUEST','规格域必须是字符串列表')
        if not set(names)<=known:raise DesignInputError('UNSUPPORTED_GRADE','存在未知或错误角色规格')
        if not names:raise DesignInputError('INFEASIBLE_DOMAIN','没有合法'+role+'规格')
