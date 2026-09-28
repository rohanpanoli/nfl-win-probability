import pandas as pd

def compute_current_team_stats(games, N=5):
    """
    Returns each team's most recent rolling stats based on their
    last N completed games.
    """
    completed = games.dropna(subset=['home_score', 'away_score']).copy()

    home = completed[['game_id', 'season', 'week', 'gameday', 'home_team', 'home_score', 'away_score']].rename(
        columns={'home_team': 'team', 'home_score': 'points_for', 'away_score': 'points_against'}
    )
    away = completed[['game_id', 'season', 'week', 'gameday', 'away_team', 'away_score', 'home_score']].rename(
        columns={'away_team': 'team', 'away_score': 'points_for', 'home_score': 'points_against'}
    )

    team_games = pd.concat([home, away]).sort_values(['team', 'season', 'week'])
    team_games['win'] = (team_games['points_for'] > team_games['points_against']).astype(int)

    # Takes the last N games per team 
    current_stats = (
        team_games.groupby('team')
        .apply(lambda x: pd.Series({
            'roll_pf': x['points_for'].tail(N).mean(),
            'roll_pa': x['points_against'].tail(N).mean(),
            'roll_winpct': x['win'].tail(N).mean()
        }))
        .reset_index()
    )
    return current_stats


def compute_current_elo(games, K=20, base_elo=1500):
    """
    Replays all completed games in order and returns each team's
    Elo rating as of the most recent completed game.
    """
    completed = games.dropna(subset=['home_score', 'away_score']).copy()
    completed['home_win'] = (completed['home_score'] > completed['away_score']).astype(int)

    elo = {team: base_elo for team in pd.concat([completed['home_team'], completed['away_team']]).unique()}

    for _, row in completed.sort_values(['season', 'week']).iterrows():
        h, a = row['home_team'], row['away_team']
        expected_home = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
        actual_home = row['home_win']

        elo[h] += K * (actual_home - expected_home)
        elo[a] += K * ((1 - actual_home) - (1 - expected_home))

    return pd.DataFrame(list(elo.items()), columns=['team', 'elo'])