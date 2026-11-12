"""从冻结数据库核对一个训练配方的合成原文，为候选提供有边界的来源依据。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .io import digest, sha256


def read_training_route_example(database: Path, database_sha256: str, formula: dict) -> dict:
    """只读指定train材料及合成证据，核对来源关联和页码；不读取测试性质表。

    返回原文、定位、结构化缺失字段及完整原始行摘要。不把自然语言工艺自动转为
    无来源数值；具体温度、时间等可由证据审阅单独结构化，本入口默认保留null。
    """
    if formula['split']!='train' or not formula.get('synthesis_evidence_id'):
        raise ValueError('合成回读仅允许有证据ID的冻结训练配方')
    if sha256(database)!=database_sha256: raise ValueError('合成来源数据库版本变化')
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        db.row_factory=sqlite3.Row
        material=db.execute('SELECT * FROM materials WHERE material_id=?',(formula['material_id'],)).fetchone()
        evidence=db.execute('SELECT * FROM evidence WHERE evidence_id=?',(formula['synthesis_evidence_id'],)).fetchone()
    if material is None or evidence is None: raise ValueError('合成原材料或证据行缺失')
    material,evidence=dict(material),dict(evidence)
    if material['synthesis_evidence_id']!=evidence['evidence_id'] or material['source_id']!=formula['source_id'] or evidence['source_id']!=formula['source_id']:
        raise ValueError('材料—合成证据—来源关联不闭合')
    if not evidence['locator_text'] or not evidence['text_raw']:
        raise ValueError('合成依据缺少原文或来源定位')
    return {'recipe':{k:formula[k] for k in ('grade_ids','nco_oh_ratio','extender_oh_fraction')},
        'material_id':formula['material_id'],'source_id':formula['source_id'],
        'evidence_id':evidence['evidence_id'],'locator':evidence['locator_text'],
        'source_route':material['synthesis_route'],'quoted_source_evidence':evidence['text_raw'],
        'raw_material_row_sha256':digest(material),'raw_evidence_row_sha256':digest(evidence),
        'database_sha256':database_sha256,'synthesis_status':material['synthesis_status'],
        'temperature_c':None,'time_h':None,'solvent':None,'catalyst':None,
        'structured_fields_status':'NOT_STRUCTURED_FROM_SOURCE_TEXT','experimental_validation_of_new_recipe':False}


def synthesis_basis(recipe: dict, example: dict | None) -> dict:
    """精确匹配才标记来源配方；新组合只引用类似反应，不继承已验证工艺断言。"""
    if example is None:
        return {'status':'SOURCE_INCOMPLETE','temperature_c':None,'time_h':None,'solvent':None,'catalyst':None,
                'reason':'没有经过回读的对应合成来源','experimental_validation_of_new_recipe':False}
    old=example['recipe']
    same=(recipe['grade_ids']==old['grade_ids'] and abs(recipe['nco_oh_ratio']-old['nco_oh_ratio'])<=1e-8 and
          abs(recipe['extender_oh_fraction']-old['extender_oh_fraction'])<=1e-8)
    return {'status':'MATCHED_TRAINING_SOURCE' if same else 'SUGGESTED_ANALOGUE_ONLY',
        'source_material_id':example['material_id'],'source_id':example['source_id'],
        'evidence_id':example['evidence_id'],'locator':example['locator'],
        'source_route':example['source_route'],'quoted_source_evidence':example['quoted_source_evidence'],
        'source_recipe':old,'matches_source_recipe':same,'temperature_c':None,'time_h':None,'solvent':None,'catalyst':None,
        'reason':'来源工艺作为类似反应依据，不能据此断言新组合已合成或已验证' if not same else '配方与训练来源匹配；性质仍按独立请求核验',
        'experimental_validation_of_new_recipe':False}
