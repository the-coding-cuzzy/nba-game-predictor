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
    season_data = pandas.read_csv("bball_ref_data/season-{0}.csv".format(season), parse_dates=['Date'])

    season_data.columns = ["Date", "Time", "Visitor Team", "Visitor Points", "Home Team", "Home Points", "Score Type", "OT?", "Notes"]
    season_data["HomeWin"] = season_data["Visitor Points"] < season_data["Home Points"]
    y_true = season_data["HomeWin"].values

    # Base Line Comparison
    n_games = season_data["HomeWin"].count()
    n_homewins = season_data["HomeWin"].sum()
    win_percentage = float(n_homewins) / float(n_games)

    print("Home team wins {0:.2f}% of the time".format(100 * win_percentage))

    season_data["HomeLastWin"] = False
    season_data["VisitorLastWin"] = False
    won_last = defaultdict(int)

    for index, row in season_data.iterrows(): # Note this is not efficient
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        row["HomeLastWin"] = won_last[home_team]
        row["VisitorLastWin"] = won_last[visitor_team]
        season_data.ix[index] = row
        # Set current win
        won_last[home_team] = row["HomeWin"]
        won_last[visitor_team] = not row["HomeWin"]

    season_data["HomeWinStreak"] = 0
    season_data["VisitorWinStreak"] = 0
    win_streak = defaultdict(int)

    for index, row in season_data.iterrows(): # Note this is not efficient
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        row["HomeWinStreak"] = win_streak[home_team]
        row["VisitorWinStreak"] = win_streak[visitor_team]
        season_data.ix[index] = row
        # Set current win
        if row["HomeWin"]:
            win_streak[home_team] += 1
            win_streak[visitor_team] = 0
        else:
            win_streak[home_team] = 0
            win_streak[visitor_team] += 1

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
    x_home_higher = season_data[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher", "HomeTeamWonLast"]].values
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