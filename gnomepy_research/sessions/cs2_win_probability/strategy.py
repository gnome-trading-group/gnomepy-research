from gnomepy_research.sessions.cs2_win_probability.cs2_win_probability import CS2WinProbability

strategy = CS2WinProbability(
    listing_id_yes=0,  # placeholder — override in config via strategy.args
    listing_id_no=0,
    model_path="artifact://xgboost_model/cs2_round_win_prob",
    level="map",
    ct_team_is_team_a=True,
    edge_threshold=0.03,
    half_spread=0.02,
    max_position=100,
    maker_size=10,
)
