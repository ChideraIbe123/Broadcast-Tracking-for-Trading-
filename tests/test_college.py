import numpy as np
import pandas as pd

from hoops.college.kalshi_map import TeamMatcher, _event_parts
from hoops.college.ratings import fit_ratings


def synth_season(n_teams=40, games_per_team=25, hfa=3.0, sigma=10.0, seed=0, season=2026, start="2025-11-05"):
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 8, n_teams)
    rows = []
    t0 = pd.Timestamp(start)
    for k in range(n_teams * games_per_team // 2):
        h, a = rng.choice(n_teams, 2, replace=False)
        neutral = rng.random() < 0.1
        margin = strength[h] - strength[a] + (0 if neutral else hfa) + rng.normal(0, sigma)
        hs = 70 + rng.normal(0, 5)
        rows.append(dict(game_id=k, season=season, game_date=t0 + pd.Timedelta(days=int(k / (n_teams / 2))), neutral=neutral, completed=True,
                         home_id=str(h), away_id=str(a), home_score=hs + margin / 2, away_score=hs - margin / 2, margin=margin, total=2 * hs,
                         home_display_name=f"T{h}", away_display_name=f"T{a}"))
    return pd.DataFrame(rows), strength


def test_ratings_recover_strengths_and_hfa():
    games, strength = synth_season()
    m = fit_ratings(games)
    r = np.array([m.rating[str(i)] for i in range(len(strength))])
    corr = np.corrcoef(r, strength - strength.mean())[0, 1]
    assert corr > 0.9, corr  # 25 games per team at sigma 10 is noise-limited
    assert 1.5 < m.hfa < 4.5, m.hfa
    assert 8 < m.sigma < 12.5, m.sigma
    assert 0.5 < m.p_home(str(int(np.argmax(strength))), str(int(np.argmin(strength)))) > 0.9


def test_prior_gives_opening_night_opinion():
    prev, strength = synth_season(season=2025, start="2024-11-05")
    prior = fit_ratings(prev)
    cur = prev.iloc[:0].copy()  # no games yet this season
    m = fit_ratings(cur, prior=prior)
    best, worst = str(int(np.argmax(strength))), str(int(np.argmin(strength)))
    assert m.p_home(best, worst) > 0.8
    # shrunk towards the mean relative to last season's full-strength rating
    assert abs(m.rating[best]) < abs(prior.rating[best])


def test_team_matching_and_event_parsing():
    teams = pd.DataFrame(
        [("1", "Utah", "Utah", "Utah Utes", "UTAH"), ("2", "Towson", "Towson", "Towson Tigers", "TOW"), ("3", "Michigan State", "Michigan St", "Michigan State Spartans", "MSU"),
         ("4", "Miami", "Miami", "Miami Hurricanes", "MIA"), ("5", "St. John's", "St. John's", "St. John's Red Storm", "SJU")],
        columns=["team_id", "location", "short", "display", "abbr"],
    )
    tm = TeamMatcher(teams)
    assert tm.match("Utah") == "1" and tm.match("Towson") == "2"
    assert tm.match("Michigan St.") == "3" and tm.match("Michigan State") == "3"
    assert tm.match("Saint John's") == "5"
    assert tm.match("Gonzaga") is None
    assert _event_parts("KXNCAAMBGAME-26NOV05TOWSUTAH") == ("26NOV05", "TOWSUTAH")
    assert _event_parts("KXNCAAMBTOTAL-26NOV02MTSTCSU") == ("26NOV02", "MTSTCSU")
