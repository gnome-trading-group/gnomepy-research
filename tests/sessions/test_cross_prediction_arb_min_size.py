from tests.sessions.conftest import SIZE_SCALE, _FakeRegistry


class TestMinSize:
    def test_align_lot_zeroes_sub_minimum_size(self, make_strategy):
        s = make_strategy(registry=_FakeRegistry(lot_size=SIZE_SCALE // 100, min_size=5 * SIZE_SCALE))
        listing = next(iter(s._listing_specs))
        assert s._align_lot(listing, 4 * SIZE_SCALE + 99 * (SIZE_SCALE // 100)) == 0

    def test_align_lot_keeps_size_at_minimum(self, make_strategy):
        s = make_strategy(registry=_FakeRegistry(lot_size=SIZE_SCALE // 100, min_size=5 * SIZE_SCALE))
        listing = next(iter(s._listing_specs))
        assert s._align_lot(listing, 5 * SIZE_SCALE + SIZE_SCALE // 200) == 5 * SIZE_SCALE

    def test_align_lot_unconstrained_without_minimum(self, make_strategy):
        s = make_strategy(registry=_FakeRegistry(lot_size=SIZE_SCALE // 100))
        listing = next(iter(s._listing_specs))
        assert s._align_lot(listing, SIZE_SCALE // 100) == SIZE_SCALE // 100
