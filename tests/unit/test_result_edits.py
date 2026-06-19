"""Pure per-field diff that produces Q3 calibration labels. No DB/S3."""
from eden_summary.core.job_store import _EDIT_FIELDS, _diff_fields


def _summary(**over):
    base = {'title': 'T', 'tldr': ['a'], 'decisions': ['d1'],
            'action_items': ['x'], 'risks': ['r1']}
    base.update(over)
    return base


def _flags(rows):
    return {row['field']: row['edited'] for row in rows}


class TestDiffFields:
    def test_covers_exactly_the_judged_fields(self):
        rows = _diff_fields(_summary(), _summary())
        assert {r['field'] for r in rows} == set(_EDIT_FIELDS)
        assert 'title' not in {r['field'] for r in rows}  # title is not a judged claim

    def test_identical_summary_has_no_edits(self):
        rows = _diff_fields(_summary(), _summary())
        assert all(r['edited'] is False for r in rows)

    def test_single_changed_field_flags_only_that_field(self):
        rows = _diff_fields(_summary(), _summary(risks=['r1', 'r2-added']))
        flags = _flags(rows)
        assert flags['risks'] is True
        assert flags['decisions'] is flags['action_items'] is flags['tldr'] is False

    def test_title_change_alone_is_not_an_edit(self):
        # title is excluded from judged fields, so editing only it records no positives
        rows = _diff_fields(_summary(), _summary(title='Different title'))
        assert all(r['edited'] is False for r in rows)

    def test_reorder_counts_as_edit(self):
        # documents the order-sensitive v1 choice
        rows = _diff_fields(_summary(decisions=['d1', 'd2']),
                            _summary(decisions=['d2', 'd1']))
        assert _flags(rows)['decisions'] is True

    def test_dict_action_items_compared_structurally(self):
        stored = _summary(action_items=[{'task': 'ship', 'who': 'Ann', 'deadline': 'Fri'}])
        same = _summary(action_items=[{'who': 'Ann', 'task': 'ship', 'deadline': 'Fri'}])
        changed = _summary(action_items=[{'task': 'ship', 'who': 'Bob', 'deadline': 'Fri'}])
        assert _flags(_diff_fields(stored, same))['action_items'] is False   # key order ignored
        assert _flags(_diff_fields(stored, changed))['action_items'] is True

    def test_original_and_corrected_captured(self):
        rows = _diff_fields(_summary(risks=['old']), _summary(risks=['new']))
        risk_row = next(r for r in rows if r['field'] == 'risks')
        assert risk_row['original'] == ['old']
        assert risk_row['corrected'] == ['new']
