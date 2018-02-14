import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import defaultdict
import numpy as np
import warnings

def first(season):
    print("Season {0}".format(season))
    scorer = make_scorer(f1_score, pos_label=None, average='weighted')
    season_data = pandas.read_csv("box_scores/{0}.csv".format(season), parse_dates=['GameDate'])

    season_data.columns = ["Date", "Home Team", "Home Points", "Visitor Team", "Visitor Points",
    "HomeTurnovers", "HomeTotalRebounds", "HomeOffensiveRebounds", "HomeDefensiveRebounds", "HomeThreePointersAttempted",
    "HomeThreePointersMade", "HomeThreePointerPercentage", "HomeFreeThrowsAttempted", "HomeFreeThrowsMade", "HomeFreeThrowPercentage",
    "HomeFieldGoalsAttempted", "HomeFieldGoalsMade", "HomeFieldGoalPercentage", "HomeAssists", "HomeSteals", "HomeBlocks", "HomePersonalFouls",
    "VisitorTurnovers", "VisitorTotalRebounds", "VisitorOffensiveRebounds", "VisitorDefensiveRebounds", "VisitorThreePointersAttempted",
    "VisitorThreePointersMade", "VisitorThreePointerPercentage", "VisitorFreeThrowsAttempted", "VisitorFreeThrowsMade", "VisitorFreeThrowPercentage",
    "VisitorFieldGoalsAttempted", "VisitorFieldGoalsMade", "VisitorFieldGoalPercentage", "VisitorAssists", "VisitorSteals", "VisitorBlocks", "VisitorPersonalFouls"]
    season_data["HomeWin"] = season_data["Visitor Points"] < season_data["Home Points"]
    y_true = season_data["HomeWin"].values

    # Base Line Comparison
    n_games = season_data["HomeWin"].count()
    n_homewins = season_data["HomeWin"].sum()

    print(n_games)
    print(n_homewins)

    win_percentage = float(n_homewins) / float(n_games)

    print("Home team wins {0:.2f}% of the time".format(100 * win_percentage))

    season_data["HomeLastWin"] = False
    season_data["VisitorLastWin"] = False
    won_last = defaultdict(int)

    season_data["Winner"] = ""

    season_data["HomeWinStreak"] = 0
    season_data["VisitorWinStreak"] = 0
    win_streak = defaultdict(int)

    season_data["HomeGamesWon"] = 0
    season_data["VisitorGamesWon"] = 0
    games_won = defaultdict(int)

    season_data["HomeGamesPlayed"] = 0
    season_data["VisitorGamesPlayed"] = 0
    season_data["HomeHigherWinningPercentage"] = False
    games_played = defaultdict(int)

    season_data["HomeWinPercentage"] = 0
    season_data["VisitorWinPercentage"] = 0

    season_data["HomeTurnoversPerGame"] = 0
    season_data["VisitorTurnoversPerGame"] = 0
    season_data["HomeLessTurnovers"] = False
    total_turnovers = defaultdict(int)

    season_data["HomeAssistsPerGame"] = 0
    season_data["VisitorAssistsPerGame"] = 0
    season_data["HomeMoreAssists"] = False
    total_assists = defaultdict(int)

    season_data["HomeStealsPerGame"] = 0
    season_data["VisitorStealsPerGame"] = 0
    season_data["HomeMoreSteals"] = False
    total_steals = defaultdict(int)

    season_data["HomeBlocksPerGame"] = 0
    season_data["VisitorBlocksPerGame"] = 0
    season_data["HomeMoreBlocks"] = False
    total_blocks = defaultdict(int)

    season_data["HomeFoulsPerGame"] = 0
    season_data["VisitorFoulsPerGame"] = 0
    season_data["HomeLessFouls"] = False
    total_fouls = defaultdict(int)

    season_data["HomeSeasonThreePointerPercentage"] = 0
    season_data["VisitorSeasonThreePointerPercentage"] = 0
    season_data["HomeHigherThreePointPercentage"] = False
    three_pointers_attempted = defaultdict(int)
    three_pointers_made = defaultdict(int)

    season_data["HomeSeasonFieldGoalPercentage"] = 0
    season_data["VisitorSeasonFieldGoalPercentage"] = 0
    season_data["HomeHigherFieldGoalPercentage"] = False
    field_goals_attempted = defaultdict(int)
    field_goals_made = defaultdict(int)

    season_data["HomeSeasonFreeThrowPercentage"] = 0
    season_data["VisitorSeasonFreeThrowPercentage"] = 0
    season_data["HomeHigherFreeThrowPercentage"] = False
    free_throws_attempted = defaultdict(int)
    free_throws_made = defaultdict(int)

    season_data["HomeDaysLastGame"] = 0
    season_data["VisitorDaysLastGame"] = 0
    date_last_game = defaultdict(int)

    season_data["HomeTeamWonLast"] = False
    last_match_winner = defaultdict(int)

    for index,  row in season_data.iterrows(): # Note this is not efficient
        # HOME TEAM LAST WIN
        game_date = row['Date']
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]

        if row["Home Points"] > row["Visitor Points"]:
            row["Winner"] = home_team
        else:
            row["Winner"] = visitor_team

        row["HomeLastWin"] = won_last[home_team]
        row["VisitorLastWin"] = won_last[visitor_team]

        # HOME TEAM WIN STREAK
        row["HomeWinStreak"] = win_streak[home_team]
        row["VisitorWinStreak"] = win_streak[visitor_team]

        # GAMES WON
        row["HomeGamesWon"] = games_won[home_team]
        row["VisitorGamesWon"] = games_won[visitor_team]

        # AMOUNT OF GAMES PLAYED
        home_games = games_played[home_team]
        visitor_games = games_played[visitor_team]
        row["HomeGamesPlayed"] = games_played[home_team]
        row["VisitorGamesPlayed"] = games_played[visitor_team]

        if home_games != 0 and visitor_games != 0:
            row["HomeWinPercentage"] = float(games_won[home_team]) / float(games_played[home_team])
            row["VisitorWinPercentage"] = float(games_won[visitor_team]) / float(games_played[visitor_team])

        # TURNOVERS PER GAME
        home_turnovers = total_turnovers[home_team]
        visitor_turnovers = total_turnovers[visitor_team]
        if home_turnovers != 0:
            row["HomeTurnoversPerGame"] = home_turnovers / games_played[home_team]
        if visitor_turnovers != 0:
            row["VisitorTurnoversPerGame"] = visitor_turnovers / games_played[visitor_team]
        if home_turnovers != 0 and visitor_turnovers != 0:
            row["HomeLessTurnovers"] = (float(home_turnovers) / float(games_played[home_team])) < (float(visitor_turnovers) / float(games_played[visitor_team]))

        # ASSISTS PER GAME
        home_assists = total_assists[home_team]
        visitor_assists = total_assists[visitor_team]
        if home_assists != 0:
            row["HomeAssistsPerGame"] = home_assists / games_played[home_team]
        if visitor_assists != 0:
            row["VisitorAssistsPerGame"] = visitor_assists / games_played[visitor_team]
        if home_assists != 0 and visitor_assists != 0:
            row["HomeMoreAssists"] = (float(home_assists) / float(games_played[home_team])) > (float(visitor_assists) / float(games_played[visitor_team]))

        # STEALS PER GAME
        home_steals = total_steals[home_team]
        visitor_steals = total_steals[visitor_team]
        if home_steals != 0:
            row["HomeStealsPerGame"] = home_steals / games_played[home_team]
        if visitor_steals != 0:
            row["VisitorStealsPerGame"] = visitor_steals / games_played[visitor_team]
        if home_steals != 0 and visitor_steals != 0:
            row["HomeMoreSteals"] = (float(home_steals) / float(games_played[home_team])) > (float(visitor_steals) / float(games_played[visitor_team]))

        # BLOCKS PER GAME
        home_blocks = total_blocks[home_team]
        visitor_blocks = total_blocks[visitor_team]
        if home_blocks != 0:
            row["HomeBlocksPerGame"] = home_blocks / games_played[home_team]
        if visitor_blocks != 0:
            row["VisitorBlocksPerGame"] = visitor_blocks / games_played[visitor_team]
        if home_blocks != 0 and visitor_blocks != 0:
            row["HomeMoreBlocks"] = (float(home_blocks) / float(games_played[home_team])) > (float(visitor_blocks) / float(games_played[visitor_team]))

        # FOULS PER GAME
        home_fouls = total_fouls[home_team]
        visitor_fouls = total_fouls[visitor_team]
        if home_fouls != 0:
            row["HomeFoulsPerGame"] = home_fouls / games_played[home_team]
        if visitor_fouls != 0:
            row["VisitorFoulsPerGame"] = visitor_fouls / games_played[visitor_team]
        if home_fouls != 0 and visitor_fouls != 0:
            row["HomeLessFouls"] = (float(home_fouls) / float(games_played[home_team])) < (float(visitor_fouls) / float(games_played[visitor_team]))

        # THREE POINTERS
        home_three_pointers_attempted = three_pointers_attempted[home_team]
        visitor_three_pointers_attempted = three_pointers_attempted[visitor_team]

        home_three_pointers_made = three_pointers_made[home_team]
        visitor_three_pointers_made = three_pointers_made[visitor_team]

        if home_three_pointers_attempted != 0:
            row["HomeSeasonThreePointerPercentage"] = float(home_three_pointers_made) / float(home_three_pointers_attempted)
        if visitor_three_pointers_attempted != 0:
            row["VisitorSeasonThreePointerPercentage"] = float(visitor_three_pointers_made) / float(visitor_three_pointers_attempted)
        if home_three_pointers_attempted != 0 and visitor_three_pointers_attempted != 0:
            row["HomeHigherThreePointPercentage"] = (float(home_three_pointers_made) / float(home_three_pointers_attempted) ) > (float(visitor_three_pointers_made) / float(visitor_three_pointers_attempted))

        # FIELD GOALS
        home_field_goals_attempted = field_goals_attempted[home_team]
        visitor_field_goals_attempted = field_goals_attempted[visitor_team]

        home_field_goals_made = field_goals_made[home_team]
        visitor_field_goals_made = field_goals_made[visitor_team]

        if home_field_goals_attempted != 0:
            row["HomeSeasonFieldGoalPercentage"] = float(home_field_goals_made) / float(home_field_goals_attempted)
        if visitor_field_goals_attempted != 0:
            row["VisitorSeasonFieldGoalPercentage"] = float(visitor_field_goals_made) / float(visitor_field_goals_attempted)
        if home_field_goals_attempted != 0 and visitor_field_goals_attempted != 0:
            row["HomeHigherFieldGoalPercentage"] = (float(home_field_goals_made) / float(home_field_goals_attempted) ) > (float(visitor_field_goals_made) / float(visitor_field_goals_attempted))

        # FREE THROWS
        home_free_throws_attempted = free_throws_attempted[home_team]
        visitor_free_throws_attempted = free_throws_attempted[visitor_team]

        home_free_throws_made = free_throws_made[home_team]
        visitor_free_throws_made = free_throws_made[visitor_team]

        if home_free_throws_attempted != 0:
            row["HomeSeasonFreeThrowPercentage"] = float(home_free_throws_made) / float(home_free_throws_attempted)
        if visitor_free_throws_attempted != 0:
            row["VisitorSeasonFreeThrowPercentage"] = float(visitor_free_throws_made) / float(visitor_free_throws_attempted)
        if home_free_throws_attempted != 0 and visitor_free_throws_attempted != 0:
            row["HomeHigherFreeThrowPercentage"] = (float(home_free_throws_made) / float(home_free_throws_attempted)) > (float(visitor_free_throws_made) / float(visitor_free_throws_attempted))

        # DAYS SINCE LAST GAME       
        home_last_game = date_last_game[home_team]
        away_last_game = date_last_game[visitor_team]

        if not isinstance(home_last_game, int):
            home_last_game = home_last_game.date()
            diff = game_date.date() - home_last_game
            row["HomeDaysLastGame"] = diff.days
        else:
            row["HomeDaysLastGame"] = 0

        if not isinstance(away_last_game, int):
            away_last_game = away_last_game.date()
            diff = game_date.date() - away_last_game
            row["VisitorDaysLastGame"] = diff.days
        else:
            row["VisitorDaysLastGame"] = 0

        # HOME TEAM WON LAST
        teams = tuple(sorted([home_team, visitor_team]))
        result = True if last_match_winner[teams] == row["Home Team"] else False
        winner = row["Home Team"] if row ["HomeWin"] else ["Visitor Team"]

        row["HomeTeamWonLast"] = result

        season_data.ix[index] = row

        # HOME TEAM WON LAST
        won_last[home_team] = row["HomeWin"]
        won_last[visitor_team] = not row["HomeWin"]

        # HOME TEAM WIN STREAK
        if row["HomeWin"]:
            games_won[home_team] += 1
            win_streak[home_team] += 1
            win_streak[visitor_team] = 0
        else:
            games_won[visitor_team] += 1
            win_streak[home_team] = 0
            win_streak[visitor_team] += 1

        # AMOUT OF GAMES PLAYED
        games_played[home_team] += 1
        games_played[visitor_team] += 1

        # TURNOVERS
        total_turnovers[home_team] += row["HomeTurnovers"]
        total_turnovers[visitor_team] += row["VisitorTurnovers"]

        # ASSISTS
        total_assists[home_team] += row["HomeAssists"]
        total_assists[visitor_team] += row["VisitorAssists"]

        # BLOCKS
        total_blocks[home_team] += row["HomeBlocks"]
        total_blocks[visitor_team] += row["VisitorBlocks"]

        # STEALS
        total_steals[home_team] += row["HomeSteals"]
        total_steals[visitor_team] += row["VisitorSteals"]

        # FOULS
        total_fouls[home_team] += row["HomePersonalFouls"]
        total_fouls[visitor_team] += row["VisitorPersonalFouls"]

        # THREE POINTERS
        three_pointers_made[home_team] += row["HomeThreePointersMade"]
        three_pointers_made[visitor_team] += row["VisitorThreePointersMade"]
        
        three_pointers_attempted[home_team] += row["HomeThreePointersAttempted"]
        three_pointers_attempted[visitor_team] += row["VisitorThreePointersAttempted"]

        # FIELD GOALS
        field_goals_made[home_team] += row["HomeFieldGoalsMade"]
        field_goals_made[visitor_team] += row["VisitorFieldGoalsMade"]
        
        field_goals_attempted[home_team] += row["HomeFieldGoalsAttempted"]
        field_goals_attempted[visitor_team] += row["VisitorFieldGoalsAttempted"]

        # FREE THROWS
        free_throws_made[home_team] += row["HomeFreeThrowsMade"]
        free_throws_made[visitor_team] += row["VisitorFreeThrowsMade"]
        
        free_throws_attempted[home_team] += row["HomeFreeThrowsAttempted"]
        free_throws_attempted[visitor_team] += row["VisitorFreeThrowsAttempted"]

        # DAYS SINCE LAST GAME
        date_last_game[home_team] = game_date
        date_last_game[visitor_team] = game_date

        # HOME TEAM WON LAST
        last_match_winner[teams] = winner


    season_split = season.split("-")
    last_season_part_one = int(season_split[0]) - 1
    last_season_part_two = int(season_split[1]) - 1

    last_season = "{0}-{1}".format(last_season_part_one, last_season_part_two)

    season_data["HomeTeamRanksHigher"] = season_data.apply(lambda row: home_team_ranks_higher(row, last_season), axis=1)

    season_data["HomeTeamWonLast"] = season_data.apply(home_team_won_last, axis=1)

    return scorer, y_true, season_data

def starter(season):
    scorer, y_true, season_data = first(season)

    encoding = LabelEncoder()
    encoding.fit(season_data["Home Team"].values)

    home_teams = encoding.transform(season_data["Home Team"].values)
    visitor_teams = encoding.transform(season_data["Visitor Team"].values)
    x_teams = np.vstack([home_teams, visitor_teams]).T

    onehot = OneHotEncoder()
    x_teams = onehot.fit_transform(x_teams).todense()

    parameter_space = {
        "max_features": [2, 10, 50, 'auto'],
        "n_estimators": [50, 100, 200],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [1, 2, 4, 6] 
    }

    return encoding, onehot

def secondary(season, encoding, one_hot_encoder):
    scorer, y_true, season_data = first(season)

    home_teams = encoding.transform(season_data["Home Team"].values)
    visitor_teams = encoding.transform(season_data["Visitor Team"].values)
    x_teams = np.vstack([home_teams, visitor_teams]).T
    x_teams = one_hot_encoder.transform(x_teams).todense()

    parameter_space = {
        "max_features": [2, 10, 50, 'auto'],
        "n_estimators": [50, 100, 200],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [1, 2, 4, 6] 
    }
    x_home_higher = season_data[[
        "HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher", "HomeTeamWonLast", "HomeDaysLastGame",
        "HomeWinPercentage", "HomeWinStreak","HomeSeasonFieldGoalPercentage"]].values
    X_all = np.hstack([x_home_higher, x_teams])

    clf = RandomForestClassifier(random_state=14)
    grid = GridSearchCV(clf, parameter_space, scoring=scorer)
    grid.fit(X_all, y_true)
    print("F1: {0:.4f}".format(grid.best_score_))

    y_pred = grid.predict(X_all)
    print(classification_report(y_true, y_pred))
    print("This results in getting {0:.4f}% of predictions correct!\n".format(100 * np.mean(y_pred == y_true)))  

def home_team_ranks_higher(row, season):
    ladder = pandas.read_csv("bball_ref_data/standings-{0}.csv".format(season), index_col="Team")
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    if season == '2013-14':
        if home_team == "Charlotte Hornets":
            home_team = "Charlotte Bobcats"
        if visitor_team == "Charlotte Hornets":
            visitor_team = "Charlotte Bobcats"
    home_rank = ladder.loc[home_team]["Rk"]
    visitor_rank = ladder.loc[visitor_team]["Rk"]

    return home_rank < visitor_rank

def home_team_won_last(row):
    last_match_winner = defaultdict(int)
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]

    teams = tuple(sorted([home_team, visitor_team]))
    result = 1 if last_match_winner[teams] == row["Home Team"] else 0
    winner = row["Home Team"] if row ["HomeWin"] else ["Visitor Team"]

    last_match_winner[teams] = winner

    return result

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        from sklearn.cross_validation import cross_val_score
        from sklearn.grid_search import GridSearchCV

    encoding, one_hot_encoder = starter("2014-15")
    secondary("2015-16", encoding, one_hot_encoder)
    secondary("2016-17", encoding, one_hot_encoder)
    secondary("2017-18", encoding, one_hot_encoder)