"""Exact scoped-address and revision-set retrieval contracts."""

import pytest

from hidate.expert_metrics import retrieval_metrics


def label(number, *, scope='MainProvision', revision='rev-now', law='law-1',
          role='direct_answer', **extra):
    return dict(law_id=law, legal_scope=scope, article_number=number,
                revision_id=revision, address_role=role, **extra)


def chunk(number, *, scope='MainProvision', revision='rev-now', law='law-1'):
    return dict(source_kind='law', law_id=law, scope=scope, article_num=number,
                slot_id=f'{law}:{scope}:{number}', revision_id=revision)


def gold(*groups, answerability='answerable'):
    return dict(answerability=answerability,
                required_evidence_sets={f'G{i}': g for i, g in enumerate(groups, 1)})


def test_joint_evidence_requires_both_addresses_and_duplicates_do_not_fill_set():
    target = gold([label('1'), label('32'), label('32')])
    output = [chunk('1'), chunk('1'), chunk('99'), chunk('32')]
    incomplete = retrieval_metrics(target, output, k=3)
    assert incomplete == dict(annotated_address_recall_at_k=.5,
        revision_correct_set_coverage_at_k=.5, revision_correct_set_complete_at_k=0.,
        first_complete_set_reciprocal_rank_at_k=0.)
    complete = retrieval_metrics(target, output, k=4)
    assert complete == dict(annotated_address_recall_at_k=1.,
        revision_correct_set_coverage_at_k=1., revision_correct_set_complete_at_k=1.,
        first_complete_set_reciprocal_rank_at_k=.25)


def test_wrong_revision_hits_address_but_not_correct_evidence_set():
    target = gold([label('3')])
    result = retrieval_metrics(target, [chunk('3', revision='rev-old')])
    assert result['annotated_address_recall_at_k'] == 1.
    assert result['revision_correct_set_coverage_at_k'] == 0.
    assert result['revision_correct_set_complete_at_k'] == 0.
    assert result['first_complete_set_reciprocal_rank_at_k'] == 0.
    # A later correct version is not discarded after seeing the wrong version.
    result = retrieval_metrics(target, [chunk('3', revision='rev-old'), chunk('3')])
    assert result['first_complete_set_reciprocal_rank_at_k'] == .5


def test_alternative_sufficient_set_is_not_penalized_for_unretrieved_alternative():
    target = gold([label('1'), label('2')],
                  [label('9'), label('10', role='context_only')])
    result = retrieval_metrics(target, [chunk('99'), chunk('9')])
    assert result['annotated_address_recall_at_k'] == 1.
    assert result['revision_correct_set_coverage_at_k'] == 1.
    assert result['revision_correct_set_complete_at_k'] == 1.
    assert result['first_complete_set_reciprocal_rank_at_k'] == .5


def test_canonical_label_is_exact_and_raw_label_is_retained_without_fuzzy_match():
    original = label('1', scope='SupplProvision:六月十一日')
    target = gold([original])
    canonical_chunk = chunk('1', scope='SupplProvision:六月一一日')
    assert retrieval_metrics(target, [canonical_chunk])['annotated_address_recall_at_k'] == 0.
    original['canonical'] = label('1', scope='SupplProvision:六月一一日')
    assert retrieval_metrics(target, [canonical_chunk])['revision_correct_set_complete_at_k'] == 1.
    assert original['legal_scope'] == 'SupplProvision:六月十一日'
    for other in [chunk('1', scope='MainProvision'),
                  chunk('1', scope='SupplProvision:六月一一日', law='law-2'),
                  chunk('2', scope='SupplProvision:六月一一日')]:
        assert retrieval_metrics(target, [other])['annotated_address_recall_at_k'] == 0.


def test_unnumbered_source_slot_uses_its_exact_suffix():
    target = gold([label('Paragraph_1', scope='SupplProvision:original')])
    output = chunk(None, scope='SupplProvision:original')
    output['slot_id'] = 'law-1:SupplProvision:original:Paragraph_1'
    assert retrieval_metrics(target, [output])['revision_correct_set_complete_at_k'] == 1.


@pytest.mark.parametrize('answerability', [
    'insufficient_material_evidence', 'insufficient_facts'])
def test_insufficient_cases_are_excluded_even_when_context_address_is_returned(answerability):
    target = gold([label('1', role='context_only')], answerability=answerability)
    result = retrieval_metrics(target, [chunk('1')])
    assert len(result) == 4 and all(v is None for v in result.values())


def test_duplicate_output_does_not_pull_fifth_chunk_inside_four_chunk_budget():
    result = retrieval_metrics(gold([label('2')]),
                               [chunk('1')]*4+[chunk('2')], k=4)
    assert all(value == 0. for value in result.values())


def test_answerable_without_a_positive_set_is_an_incomplete_gold_not_a_zero_score():
    with pytest.raises(ValueError, match='required evidence set'):
        retrieval_metrics(gold([label('1', role='context_only')]), [chunk('1')])


def test_temporal_nonapplicability_has_no_provision_target_and_is_not_zero_or_one():
    target = gold()
    target['retrieval_target_type'] = 'temporal_nonapplicability'
    assert target['answerability'] == 'answerable'
    for output in ([], [chunk('1')]):
        result = retrieval_metrics(target, output)
        assert len(result) == 4 and all(value is None for value in result.values())
