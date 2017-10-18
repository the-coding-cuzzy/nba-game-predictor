from nba_py.team import TeamList, TeamDetails, TeamSummary
from nba_py.game import Boxscore
from nba_py.league import GameLog
from nba_py import constants
import json
import csv

def get_basic_data():
    # Youngest Team is New Orleans Pelicans, started in 2002
    seasons = [
        "2002-03", "2003-04", "2004-05", "2005-06", "2006-07", 
        "2007-08", "2008-09", "2009-10", "2010-11", "2011-12",
        "2012-13", "2013-14", "2014-15", "2015-16", "2016-17"
    ]

    team_abrv_to_team_id = {
        'MIL': 1610612749, 'GSW': 1610612744, 'MIN': 1610612750, 'MIA': 1610612748, 'ATL': 1610612737, 'BOS': 1610612738, 
        'DET': 1610612765, 'NYK': 1610612752, 'DEN': 1610612743, 'DAL': 1610612742, 'BKN': 1610612751, 'POR': 1610612757, 
        'OKC': 1610612760, 'TOR': 1610612761, 'CHI': 1610612741, 'SAS': 1610612759, 'CHA': 1610612766, 'UTA': 1610612762, 
        'CLE': 1610612739, 'HOU': 1610612745, 'WAS': 1610612764, 'LAL': 1610612747, 'PHI': 1610612755, 'MEM': 1610612763, 
        'LAC': 1610612746, 'SAC': 1610612758, 'ORL': 1610612753, 'PHX': 1610612756, 'IND': 1610612754, 'NOP': 1610612740
    }
    with open("data/all-seasons.csv", "wb+") as data:
        for season in seasons:
            game_log = GameLog(season=season, player_or_team=constants.Player_or_Team.Team)
            writer = csv.writer(data)
            writer.writerow([
                "TeamId", "OpponentId", "Result", "Turnovers", "TotalRebounds", "OffensiveRebounds","DefensiveRebounds",
                "ThreePointersAttempted", "ThreePointersMade", "ThreePointerPercentage", "FreeThrowsAttempted", "FreeThrowsMade", "FreeThrowPercentage",
                "FieldGoalsAttempted", "FieldGoalsMade", "FieldGoalPercentage", "Assists", "Steals", "Blocks", "PersonalFouls"
            ])
            games = json.loads(game_log.overall().reset_index().to_json(orient='records'))

            for game in games:
                result = 1 if game["WL"] == "W" else 0
                if game["MATCHUP"][-3:] == "NJN":
                    opp = "BKN"
                    team = "BKN"
                elif game["MATCHUP"][-3:] == "NOH":
                    opp = "NOP"
                elif game["MATCHUP"][-3:] == "SEA":
                    opp = "OKC"
                elif game["MATCHUP"][-3:] == "NOK":
                    continue
                else:
                    opp = game["MATCHUP"][-3:]

                if game['TEAM_ABBREVIATION'] == "NJN":
                    team = "BKN"
                elif game['TEAM_ABBREVIATION'] == "NOH":
                    team = "NOP"
                elif game['TEAM_ABBREVIATION'] == "SEA":
                    team = "OKC"
                elif game['TEAM_ABBREVIATION'] == "NOK":
                    continue
                else:
                    team = game['TEAM_ABBREVIATION']
                
                writer.writerow([
                    team_abrv_to_team_id[team], team_abrv_to_team_id[opp], result, game["TOV"], game["REB"], game["OREB"], game["DREB"],
                    game["FG3A"], game["FG3M"], game["FG3_PCT"], game["FTA"], game["FTM"], game["FT_PCT"],
                    game["FGA"], game["FGM"], game["FG_PCT"], game["AST"], game["STL"], game["BLK"], game['PF']
                ])

        # TODO: Use data direct, as its already in a pandas dataframe
        # games = game_log.overall()
        # print(games.columns)
        # print(games.shape)
if __name__ == '__main__':
    get_basic_data()