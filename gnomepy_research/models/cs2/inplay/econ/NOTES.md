# econ: next-round economy in the map price

## Idea

The benchmark prices a state as `P(A wins round | s) * M(score if A wins) + P(A loses | s) * M(score if A loses)`,
where M sees only the score. The next round's economy is decided by this round: survivors keep their weapons, every
player's money moves by the win reward / loss bonus / kill rewards / plant bonus. This entry replaces M(score) with
M(s, o): the benchmark's score-state M as a base margin plus a boosted correction learned on in-round states of the
2026 timeline maps, each row conditioned on its *realised* round outcome o. Rows come from P(s, o), so a model fitted on
them estimates P(map | s, o) directly, with no need to forecast who survives: given o, who is alive now, the money held
after this round's buy, kills so far and the loss counters describe the next round's economy.

## Money rules learned from the data (`players.money_pre` / `money`)

Zero-kill losers of eliminated rounds (2025-06 onward), income by the team's recent results:
- pistol loss pays $1,900, then $2,400, $2,900, $3,400 (cap): the loss counter starts at 1 at each half start;
- a win lowers the counter by one, it does not reset it (lose, lose, win, lose pays $1,900; win, lose pays $1,400);
- a win pays ~$3,250 (+$250 for bomb objectives); Ts that lose after a plant get a bonus (~$800);
- Ts that survive a timeout loss get nothing;
- kill rewards by weapon as in the guides (AWP $100, SMG $600, shotgun $900, knife $1,500, most others $300).

These drive per-player projections of the next round's pre-buy money under each outcome (capped at $16,000), and
"buy capability" counts: players who could buy rifle + armor next round (survivors who keep a rifle only need a top-up).

## Configurations scored (full development, 2026-04..08, entry minus benchmark log loss)

(see the table at the end; updated as runs finish)

## Point-in-time

All features of a state in round r come from: round r's round-start columns (`players.money`, `equip`, `prim`, `armor`,
`kit`; `rounds` economy), earlier rounds' scores (loss counters, score state), and round r's events with
`seq <= event_seq` (deaths, kills and their weapons, plant). The correction model is trained only on maps that ended
before the scored month. Nothing depends on other states in the batch. The harness audit passes on every run.
