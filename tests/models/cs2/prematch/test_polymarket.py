from gnomepy_research.models.cs2.prematch.polymarket import norm


def test_accented_and_plain_spellings_normalise_alike():
    assert norm("Los Niños") == norm("LOS NINOS") == "losninos"
    assert norm("FC Famalicão Esports") == norm("FC Famalicao") == "fcfamalicao"


def test_suffixes_and_punctuation_are_dropped():
    assert norm("Fluxo Esports") == "fluxo"
    assert norm("MIBR fe") == "mibr"
