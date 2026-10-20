from hidate.extension import latest_boundary


def test_boundary_uses_enforcement_not_promulgation_or_future_schedule():
    law = {'law_info': {'law_id': 'L'}, 'family_id': 'F', 'revision_info': {'law_title': '法律'}}
    revisions = [
        {'law_revision_id': 'future', 'amendment_enforcement_date': '2027-01-01', 'current_revision_status': 'UnEnforced'},
        {'law_revision_id': 'scheduled', 'amendment_enforcement_date': '2026-09-01',
         'amendment_scheduled_enforcement_date': '2026-09-01', 'current_revision_status': 'CurrentEnforced'},
        {'law_revision_id': 'actual', 'amendment_enforcement_date': '2026-04-01',
         'amendment_promulgate_date': '2024-06-01', 'current_revision_status': 'PreviousEnforced'}]
    result = latest_boundary(law, revisions)
    assert (result['before'], result['after'], result['day_after']) == ('2026-03-31', '2026-04-01', '2026-04-02')
    assert result['revision_id'] == 'actual'
