"""协议版本、默认来源和不同错误语义的工程边界用例。"""
import pytest

from coarsepgnn_inverse.io import digest
from coarsepgnn_inverse.request_api import resolve_design_request, DesignInputError


def registry():
    """给出不含任何真实标签的两个版本化测试协议。"""
    records={}
    for name,method in [('A','DSC'),('B','DMA')]:
        definition={'target':'tg','condition':{'raw':method,'normalized':{'fields':{}}},
                    'assay':{'property_task':'tg:'+method+':criterion:phase'}}
        key=digest(definition);records[name]={'protocol_id':'PROTOCOL_'+key,'canonical_sha256':key,'definition':definition}
    return {'tg':records}


def base_request():
    """默认A由锁解析并显式标记，目标上下界来自用户而非协议。"""
    return {'schema_version':1,'constraints':[{'target':'tg','unit':'degC','lower':-10,'upper':10}]}


def test_default_protocol_preserves_missing_fields():
    """默认协议只能复制登记字节，不把空fields补成室温或常压。"""
    result=resolve_design_request(base_request(),registry())
    assert result['resolved_protocols'][0]['used_default']
    assert result['constraints'][0]['condition']['normalized']['fields']=={}
    assert result['constraints'][0]['condition']['raw']=='DSC'


def test_inconsistent_intervals_are_not_invalid_units():
    """单条合法但同协议交集为空时标约束不可行；颠倒区间仍为非法输入。"""
    request=base_request();request['constraints'].append({**request['constraints'][0],'lower':20,'upper':30})
    with pytest.raises(DesignInputError) as error: resolve_design_request(request,registry())
    assert error.value.status=='INFEASIBLE_CONSTRAINTS'
    request=base_request();request['constraints'][0]['lower']=20
    with pytest.raises(DesignInputError) as error: resolve_design_request(request,registry())
    assert error.value.status=='INVALID_REQUEST'


def test_unknown_protocol_and_mutated_definition_are_rejected():
    """未知ID不能当默认协议；已知ID与提供的条件不一致时必须拒绝。"""
    request=base_request();request['constraints'][0]['protocol_id']='UNKNOWN'
    with pytest.raises(DesignInputError) as error: resolve_design_request(request,registry())
    assert error.value.status=='UNSUPPORTED_PROTOCOL'
    request['constraints'][0]['protocol_id']=registry()['tg']['A']['protocol_id']
    request['constraints'][0]['condition']={'raw':'not the registered text','normalized':{'fields':{}}}
    with pytest.raises(DesignInputError) as error: resolve_design_request(request,registry())
    assert error.value.status=='INVALID_REQUEST'
