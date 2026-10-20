from freeze_revision_recipe import choose


def test_validation_objective_precedes_complexity():
    rows=[{'method':'simple','mrr4':.92,'CE_views_per_candidate':1,'feature_count':0},
          {'method':'learned','mrr4':.93,'CE_views_per_candidate':2,'feature_count':20}]
    assert choose(rows)['method']=='learned'


def test_identical_validation_scores_prefer_simpler_recipe():
    rows=[{'method':'learned','mrr4':.92,'CE_views_per_candidate':2,'feature_count':20},
          {'method':'simple','mrr4':.92,'CE_views_per_candidate':1,'feature_count':0}]
    assert choose(rows)['method']=='simple'
