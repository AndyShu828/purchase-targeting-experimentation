from datetime import date

CUTOFF = date(2019, 11, 18)


def test_feature_table_is_one_row_per_population_user_without_labels(make_events):
    from pte.features.build import build_feature_table

    events = make_events(
        [
            {"user_id": 1, "event_type": "view", "event_time": "2019-11-10 10:00:00"},
            {"user_id": 1, "event_type": "purchase", "event_time": "2019-11-20 10:00:00"},
            {"user_id": 2, "event_type": "cart", "event_time": "2019-11-05 10:00:00"},
            {"user_id": 3, "event_type": "view", "event_time": "2019-11-19 10:00:00"},
        ]
    )
    table = build_feature_table(events, CUTOFF)

    assert sorted(r.user_id for r in table.select("user_id").collect()) == [1, 2]
    assert "label" not in table.columns
    # No two groups may emit the same column name, or the joins would shadow one.
    assert len(table.columns) == len(set(table.columns))


def test_label_table_matches_the_feature_population(make_events):
    from pte.features.build import build_label_table

    events = make_events(
        [
            {"user_id": 1, "event_type": "view", "event_time": "2019-11-10 10:00:00"},
            {"user_id": 1, "event_type": "purchase", "event_time": "2019-11-20 10:00:00"},
            {"user_id": 2, "event_type": "cart", "event_time": "2019-11-05 10:00:00"},
        ]
    )
    labels = {r.user_id: r.label for r in build_label_table(events, CUTOFF, 13).collect()}
    assert labels == {1: 1, 2: 0}
