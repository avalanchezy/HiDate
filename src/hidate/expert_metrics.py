"""Retrieval of expert-annotated scoped articles and their source revisions.

Labels within a required evidence set are jointly necessary; different sets are
alternative sufficient evidence. These are provision-set metrics, not checks of
the quoted span, the legal conclusion, or the generated answer.
"""

from .discovery import chunk_address


METRIC_NAMES = (
    'annotated_address_recall_at_k',
    'revision_correct_set_coverage_at_k',
    'revision_correct_set_complete_at_k',
    'first_complete_set_reciprocal_rank_at_k',
)


def _gold_identity(label):
    value = label.get('canonical', label)
    address = (value['law_id'], value['legal_scope'], value['article_number'])
    return address, value['revision_id']


def retrieval_metrics(gold, chunks, k=4):
    """Return four metrics, or ``None`` if there is no provision-retrieval target.

    ``gold`` is the imported question record with ``answerability`` and
    ``required_evidence_sets`` (a mapping of set IDs to evidence-label lists).
    A label's optional ``canonical`` dictionary contains its already-audited
    law_id/legal_scope/article_number/revision_id; otherwise its top-level
    fields are used. No alias, fuzzy scope, date, or revision inference occurs.
    ``retrieval_target_type`` defaults to ``provision_set``. An answerable
    ``temporal_nonapplicability`` case has no applicable provision set; all four
    metrics are None rather than zero or a vacuously complete empty set.

    ``chunks`` is the ranked list of actual corpus chunk dictionaries. Only its
    first k positions count: duplicate chunks do not expand the output budget.
    Repeated evidence for one scoped article/revision counts once. A wrong
    version can hit the address metric but cannot hit revision-correct metrics.

    Address recall and revision-correct coverage are the best coverage over
    alternative sufficient sets, so retrieving one complete alternative is not
    penalized for missing another. Complete is one iff any revision-correct set
    is fully retrieved. Reciprocal rank is 1/r at the earliest *output position*
    r that completes any such set. Empty output scores zero for provision-set gold;
    an answerable provision-set record without a required positive set is incomplete gold and
    raises ValueError instead of receiving a fabricated retrieval score.
    ``context_only`` labels are never positive targets.
    """
    if (gold['answerability'] != 'answerable' or
            gold.get('retrieval_target_type', 'provision_set') == 'temporal_nonapplicability'):
        return dict.fromkeys(METRIC_NAMES, None)

    sets = [{_gold_identity(label) for label in labels
             if label.get('address_role') != 'context_only'}
            for labels in gold['required_evidence_sets'].values()]
    sets = [required for required in sets if required]
    if not sets:
        raise ValueError('Answerable gold needs a required evidence set')

    hits = set()
    first_complete = 0.
    for rank, chunk in enumerate(chunks[:k], 1):
        hits.add((chunk_address(chunk), chunk.get('revision_id')))
        if not first_complete and any(required <= hits for required in sets):
            first_complete = 1./rank

    hit_addresses = {address for address, _ in hits}
    address_sets = [{address for address, _ in required} for required in sets]
    address_recall = max(len(required & hit_addresses)/len(required)
                         for required in address_sets)
    revision_coverage = max(len(required & hits)/len(required) for required in sets)
    return dict(zip(METRIC_NAMES, (
        address_recall,
        revision_coverage,
        float(any(required <= hits for required in sets)),
        first_complete,
    )))
