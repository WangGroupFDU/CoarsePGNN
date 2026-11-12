"""统一候选预算、固定R、NONE与排名的合成接口测试。"""
from coarsepgnn_inverse.search import CandidateSearch


def test_duplicates_consume_budget_and_all_methods_preserve_given_ratio():
    """重复原料/配比仍占槽；固定R不能在采样或细化时改成默认值。"""
    catalog=[{'grade_id':g,'role':role,'functionality':2,'molar_mass_g_mol':mass}
             for g,role,mass in [('D','diisocyanate',250),('P','polyol',1000),('E','hydroxyl_chain_extender',90)]]
    calls=[]
    def score(request,recipes):
        """记录全部收费评分，即使配方完全重复也不减少账本。"""
        calls.extend(recipes)
        return [{'predictions':[0.],'actual_task_calls':1} for recipe in recipes]
    def inverse(request,ratio,count):
        """特意生成重复NONE配方以覆盖重复计费与严格零比例。"""
        return [{'grade_ids':['D','P','NONE'],'nco_oh_ratio':ratio,'extender_oh_fraction':0.} for _ in range(count)]
    request={'schema_version':1,'nco_oh_ratio':1.23,'candidate_budget':512,'top_k':20,
             'constraints':[{'target':'tg','unit':'degC','lower':-10.,'upper':10.,'protocol_id':'synthetic',
                             'condition':{'raw':'synthetic'},'assay':{'criterion':'synthetic'}}]}
    runner=CandidateSearch(catalog,{'nco_oh_ratio':{'min':.8,'max':1.4}}, {'tg':{'delta':10}},score,inverse)
    for method in ('inverse','legal_random','combination_ratio_search'):
        result=runner.run(request,method,seed=7)
        assert result['accounting']['proposal_slots']==512
        assert result['accounting']['logical_search_vectors']==512
        assert result['accounting']['actual_search_task_calls']==512
        assert result['accounting']['legal_proposals']==512
        assert all(row['recipe']['nco_oh_ratio']==1.23 for row in result['all_proposals'])
        assert all(row['recipe']['extender_oh_fraction']==0 for row in result['all_proposals'] if row['recipe']['grade_ids'][2]=='NONE')
        assert len(result['top_candidates'])<=10
    assert len(calls)==1536
