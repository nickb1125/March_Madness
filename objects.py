import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import itertools
import random
season_rankings = pd.read_csv("data/season_rankings_men.csv").query("Season >= 2003")
season_game_results = pd.read_csv("data/game_results.csv").query("Season >= 2003")
tourney_seeds = pd.read_csv('data/tourney_seeds.csv').query("Season >= 2003")
tourney_results = pd.read_csv('data/tourney_results.csv').query("Season >= 2003")
tourney_slots = pd.read_csv("data/tourney_slots.csv")
teams = pd.read_csv("data/teams.csv")
all_years = tourney_seeds.Season.unique()

cache = dict({year : dict({}) for year in all_years})
class team:
    def __init__(self, team_id : int, year : int) -> None:
        self.team_id = team_id
        self.year = year
        self.team_season_rank = season_rankings.query('TeamID == @team_id & Season == @year')
        self.team_game_results = season_game_results.query('TeamID == @team_id & Season == @year')
    

    def get_preseason_rank(self, system :str) -> int:
        """Get preseason ranking of team."""
        preseason_rank = self.team_season_rank.query("SystemName == @system").query("RankingDayNum == RankingDayNum.max()").reset_index(drop = 1).OrdinalRank[0]
        return preseason_rank


    def get_latest_rank(self, system : str) -> int:
        """Get current ranking of team."""
        latest_rank = self.team_season_rank.query("SystemName == @system").query("RankingDayNum == RankingDayNum.min()").reset_index(drop = 1).OrdinalRank[0]
        return latest_rank

    def get_depth(self):
        """Add player level depth"""
        pass

    def get_MOV(self):
        """Get season long margin of victory"""
        pass

    def get_past_n_MOV(self):
        """Get margin of victory over last n games."""
        pass

    def get_past_n_record(self, n : int) -> float:
        """Get (percentage) record over past 10 games"""
        past_n_record = self.team_game_results.sort_values('DayNum', ascending=False).head(10).Result.mean()
        return past_n_record
    
    def get_past_n_strength_of_schedule(self, n) -> float:
        """Get average overall kempom of past n game matchups"""
        pass
    

    def get_road_record(self) -> float:
        """Get overall road record"""
        road_record = self.team_game_results.query("Loc == 'A'").Result.mean()
        return road_record
    
    def get_tourney_seed(self):
        """Get tourney seed"""
        try:
            return tourney_seeds.query(f"Season == @self.year & TeamID == @self.team_id").reset_index(drop = 1).Seed[0]
        except KeyError:
            return 17 # If either (1) tourney seeds not released or (2) team did not make tourney assign fictional seed "17"
    def collect_features(self, n_games : int, system : str):
        """Collect current team level modeling features"""
        features = dict({
            "road_record" : [self.get_road_record()],
            "past_n_record" : [self.get_past_n_record(n = n_games)],
            "tourney_seed" : [self.get_tourney_seed()]
            })
        cache.get(self.year)[self.team_id] = features
        return pd.DataFrame(features)


class playoff_matchup:
    def __init__(self, team_id_1 :int, team_id_2 : int, n_games : int, system = str) -> None:
        """Init playoff_matchup class."""
        team_ids = [team_id_1, team_id_2]
        self.team_id_1 = min(team_ids) # Ensures lower team_id is the one predicted for
        self.team_id_2 = max(team_ids)
        self.n_games = n_games
        self.system = system
    

    def collect_features(self, year : int):
        """Collect features for playoff modeling"""
        if not (self.team_id_1 in cache.get(year).keys()):
            team_1_class = team(team_id = self.team_id_1, year = year)
            team_1_features = team_1_class.collect_features(n_games = self.n_games, system = self.system)
        if not (self.team_id_2 in cache.get(year).keys()):
            team_2_class = team(team_id = self.team_id_2, year = year)
            team_2_features = team_2_class.collect_features(n_games = self.n_games, system = self.system)
        team_1_features = pd.DataFrame(cache.get(year).get(self.team_id_1)).add_suffix(f'_T1')
        team_2_features = pd.DataFrame(cache.get(year).get(self.team_id_2)).add_suffix(f'_T2')
        features = pd.concat([team_1_features, team_2_features], axis = 1)
        return features
    
    def predict(self, year, model):
        """Predicts probability of smaller team_id winning"""
        game_features = self.collect_features(year = year)
        predictions = model.predict_proba(game_features)
        prob_lower_team_id = predictions[1]
        return prob_lower_team_id



class trainer:
    def __init__(self, n_games : int, system : str):
        """Init class trainer."""
        self.features_men = np.NaN
        self.labels_men = np.NaN
        self.model_men = np.NaN
        self.season_key_men = np.NaN

        self.features_women = np.NaN
        self.labels_women = np.NaN
        self.model_women = np.NaN
        self.season_key_wome = np.NaN

        self.all_years = tourney_results.Season.unique()
        self.n_games = n_games
        self.system = system

    def get_features_labels(self, gender : str):
        """Gets features and labels for all years for a certain gender."""
        all_features = pd.DataFrame()
        all_labels = np.array([])
        season_keys = np.array([])
        for year in tqdm(self.all_years):
            tourney_results_year = tourney_results.query("Season == @year & M_W == @gender")
            for index, row in enumerate(tourney_results_year.itertuples()):
                team_ids = [row.WTeamID, row.LTeamID]
                team_id_1 = min(team_ids)
                if row.WTeamID == team_id_1:
                    label = 1
                else:
                    label = 0
                all_labels = np.append(all_labels, label)
                matchup_class = playoff_matchup(min(team_ids), max(team_ids), n_games = self.n_games, system = self.system)
                this_game_features = matchup_class.collect_features(year = row.Season)
                all_features = pd.concat([all_features.copy(), this_game_features.copy()])
                season_keys = np.append(season_keys, year)
        return dict({"all_features" : all_features, "all_labels" : all_labels, "season_keys" : season_keys})

    def label_and_data_maker(self):
        """Create training and label sets for all NCAA matchups (men and women)"""
        print("Getting Mens Training Data:")
        men = self.get_features_labels(gender = "M")
        print("Getting Womens Training Data:")
        women = self.get_features_labels(gender = "W")

        self.features_men = men['all_features']
        self.labels_men = men['all_labels']
        self.season_key_men = men['season_keys']
        self.features_women = women['all_features']
        self.labels_women = women['all_labels']
        self.season_key_women = women['season_keys']

                
    def train(self, test_year : int = 2022):
        """Trains Models for both MNCAA and WNCAA"""
        model_men = XGBClassifier()
        X_train_men, X_test_men, y_train_men, y_test_men = self.features_men[self.season_key_men != 2022], self.features_men[self.season_key_men == 2022], self.labels_men[self.season_key_men != 2022], self.labels_men[self.season_key_men == 2022]
        model_men.fit(X_train_men, y_train_men)
        self.model_men = model_men

        model_women = XGBClassifier()
        X_train_women, X_test_women, y_train_women, y_test_women = self.features_women[self.season_key_women != 2022], self.features_women[self.season_key_women == 2022], self.labels_women[self.season_key_women != 2022], self.labels_women[self.season_key_women == 2022]
        model_women.fit(X_train_women, y_train_women)
        self.model_women = model_women

        pred_men = self.model_men.predict(X_test_men)
        pred_men = [round(value) for value in pred_men]
        pred_women = self.model_women.predict(X_test_women)
        pred_women = [round(value) for value in pred_women]
        accuracy_men = accuracy_score(y_test_men, pred_men)
        accuracy_women = accuracy_score(y_test_women, pred_women)
        print(f"Mens accuracy for {test_year} is {round(accuracy_men * 100, 2)} percent.")
        print(f"Womens accuracy for {test_year} is {round(accuracy_women * 100, 2)} percent.")

class tournament:
    def __init__(self, year, mens_model, womens_model, n_games, system):
        self.year = year
        self.n_games = n_games
        self.system = system
        self.this_year_tourney_teams_mens = tourney_seeds.query("Season == @year & M_W == 'M'").TeamID
        self.this_year_tourney_teams_womens = tourney_seeds.query("Season == @year & M_W == 'W'").TeamID
        self.mens_model = mens_model
        self.womens_model = womens_model
        self.mens_features = np.NaN
        self.womens_features = np.NaN
        self.predictions = np.NaN
    
    def predict_all_including_non_possible(self):
        """Predicts probabilities for every pair of matchups for all teams in tournement"""
        # get all possible matchups
        all_mens_matchups = list(itertools.combinations(self.this_year_tourney_teams_mens, 2))
        all_womens_matchups = list(itertools.combinations(self.this_year_tourney_teams_womens, 2))

        # get training data for all possible matchups
        mens_features = pd.DataFrame()
        mens_matchup_strings = []
        womens_features = pd.DataFrame()
        womens_matchup_strings = []
        print(f"Getting {self.year} mens tournement predicitions...")
        for matchup in tqdm(all_mens_matchups):
            matchup_list = [matchup[0], matchup[1]]
            matchup_class = playoff_matchup(team_id_1=matchup[0],team_id_2=matchup[1], n_games = self.n_games, system = self.system)
            matchup_features = matchup_class.collect_features(year = self.year)
            mens_features = pd.concat([mens_features.copy(), matchup_features.copy()])
            matchup_string = f"{self.year}_{min(matchup_list)}_{max(matchup_list)}"
            mens_matchup_strings.append(matchup_string)
        print(f"Getting {self.year} womens tournement predicitions.")
        for matchup in tqdm(all_womens_matchups):
            matchup_list = [matchup[0], matchup[1]]
            matchup_class = playoff_matchup(team_id_1=matchup[0],team_id_2=matchup[1], n_games = self.n_games, system = self.system)
            matchup_features = matchup_class.collect_features(year = self.year)
            womens_features = pd.concat([womens_features.copy(), matchup_features.copy()])
            matchup_string = f"{self.year}_{min(matchup_list)}_{max(matchup_list)}"
            womens_matchup_strings.append(matchup_string)
        self.womens_features = womens_features
        self.mens_features = mens_features

        # predict
        mens_predictions = self.mens_model.predict_proba(mens_features)
        womens_predictions = self.womens_model.predict_proba(womens_features)
        mens_predicitons = [predict[1] for predict in mens_predictions]
        womens_predicitons = [predict[1] for predict in womens_predictions]
        self.predictions = pd.DataFrame({"ID" : np.append(mens_matchup_strings, womens_matchup_strings), 
                                    "Pred" : np.append(mens_predicitons, womens_predicitons)})
        return(self.predictions)
        
        

    def predict_exact_tourney(self, gender : str, print_results = True):
        """Predicts binary outputs for each progressive round of tourney"""
        # if not run simulate all do that
        if not isinstance(self.predictions, pd.DataFrame):
            print("Predictions need to be run for this tourney, running now...") 
            predictions = self.predict_all_including_non_possible()
        tourney_seeds_this_year = tourney_seeds.query("Season == @self.year & M_W == @gender")
        tourney_slots_this_year = tourney_slots.query("Season == @self.year & M_W == @gender")
        updating_seed_record = dict({tourney_seeds_this_year.query("TeamID == @team_id").reset_index(drop = 1).ReigonSeed[0] : team_id for team_id in tourney_seeds_this_year.TeamID})
        for game_round in np.sort(tourney_slots.GameRound.unique()):
            slots_this_round = tourney_slots_this_year.query("GameRound == @game_round")
            if print_results:
                print(f"Simulating Round {game_round}...")
            for index, row in enumerate(slots_this_round.itertuples()):
                teams_in_matchup = [updating_seed_record.get(row.StrongSeed), updating_seed_record.get(row.WeakSeed)]
                team_1 = min(teams_in_matchup)
                team_2 = max(teams_in_matchup)
                id_in_probs = f"{self.year}_{team_1}_{team_2}"
                prediction = self.predictions.query("ID == @id_in_probs").reset_index(drop = 1).Pred[0]
                if random.random() <= prediction:
                    updating_seed_record[row.Slot] = team_1
                    team_name_win = teams.query("TeamID == @team_1").reset_index(drop = 1).TeamName[0]
                    team_name_lost = teams.query("TeamID == @team_2").reset_index(drop = 1).TeamName[0]
                    if teams_in_matchup[0] == team_1:
                        win_seed = row.StrongSeed
                        lose_seed = row.WeakSeed
                else:
                    updating_seed_record[row.Slot] = team_2
                    team_name_win = teams.query("TeamID == @team_2").reset_index(drop = 1).TeamName[0]
                    team_name_lost = teams.query("TeamID == @team_1").reset_index(drop = 1).TeamName[0]
                    if teams_in_matchup[0] == team_1:
                        lose_seed = row.StrongSeed
                        win_seed = row.WeakSeed
                if print_results:
                    print(f"Seed {win_seed}. {team_name_win} beats seed {lose_seed} {team_name_lost}")
        return(updating_seed_record)
    
    def get_all_possible_in_slot(self, slot, gender):
        all_slots = tourney_slots.query("Season == @self.year & M_W == @gender")
        tourney_seeds_this_year = tourney_seeds.query("Season == 2023 & M_W == 'M'")        
        try:
            slot_documentation = all_slots.query("Slot == @slot").reset_index(drop = 1).iloc[0]
        except IndexError:
            return(slot)
        strong_slot_seed = slot_documentation.StrongSeed
        weak_slot_seed = slot_documentation.WeakSeed
        all_possible_strong_side = self.get_all_possible_in_slot(strong_slot_seed, gender)
        all_possible_weak_side = self.get_all_possible_in_slot(weak_slot_seed, gender)
        return np.append(all_possible_strong_side, all_possible_weak_side)
    
    def get_one_round_back_ansestors(self, slot, gender):
        all_slots = tourney_slots.query("Season == @self.year & M_W == @gender")
        slot_documentation = all_slots.query("Slot == @slot").reset_index(drop = 1).iloc[0]
        return np.array([slot_documentation.StrongSeed, slot_documentation.WeakSeed])
    
    def get_team_round_probabilities(self, gender):
        slots_this_year = tourney_slots.query("Season == @self.year & M_W == @gender")
        seeds_this_year = tourney_seeds.query("Season == @self.year & M_W == @gender")
        teams_this_year = seeds_this_year.TeamID.unique()
        team_chances = pd.DataFrame({"TeamID" : teams_this_year, "TeamName" : [teams.query("TeamID == @team_id").reset_index(0).iloc[0].TeamName for team_id in teams_this_year]})
        for this_round in [0,1,2,3,4,5,6]:
            slots_this_round = slots_this_year.query("GameRound == @this_round").Slot
            team_round_probs = dict({team_id : np.array([]) for team_id in teams_this_year})
            teams_accounted_for = []
            for slot in slots_this_round:
                ansestor_slot_1, ansestor_slot_2 = self.get_one_round_back_ansestors(slot, gender)
                possible_seeds_ansestor_1 = self.get_all_possible_in_slot(slot = ansestor_slot_1, gender = gender)
                possible_seeds_ansestor_2 = self.get_all_possible_in_slot(slot = ansestor_slot_2, gender = gender)
                possible_team_ids_ansestor_1 = seeds_this_year.query("ReigonSeed in @possible_seeds_ansestor_1").TeamID.tolist()
                possible_team_ids_ansestor_2 = seeds_this_year.query("ReigonSeed in @possible_seeds_ansestor_2").TeamID.tolist()
                possible_team_ids_all = np.append(possible_team_ids_ansestor_1, possible_team_ids_ansestor_2)
                teams_accounted_for.extend(possible_team_ids_all)
                possible_slot_matchups = list(itertools.product(possible_team_ids_ansestor_1, possible_team_ids_ansestor_2))
                possible_game_ids = [f"{self.year}_{min(matchup_teams)}_{max(matchup_teams)}" for matchup_teams in possible_slot_matchups]
                for game_id in possible_game_ids:
                    team_1 = int(game_id[5:9])
                    team_2 = int(game_id[10:14])
                    prob_t1_wins_given_t2 = self.predictions.query("ID == @game_id").reset_index(drop = 1).Pred[0] # P(T1 Wins | T2 Opponent)
                    prob_t2_wins_given_t1 = 1 - prob_t1_wins_given_t2
                    if this_round == 0:
                        prob_t1_makes_it_here = 1
                        prob_t2_makes_it_here = 1
                    else:
                        prob_t1_makes_it_here = team_chances.query("TeamID == @team_1").reset_index(drop=1).iloc[0][f"Round_{this_round - 1}"] # P(T1 Opponent)
                        prob_t2_makes_it_here = team_chances.query("TeamID == @team_2").reset_index(drop=1).iloc[0][f"Round_{this_round - 1}"]
                    prob_t1_wins_and_t2_opp = prob_t1_wins_given_t2*prob_t1_makes_it_here*prob_t2_makes_it_here # P(T1 wins and T2 Opponent) = P(T1 Wins | T1 makes, T2 makes) * P(T2 Opponent) * P(T1 makes)
                    prob_t2_wins_and_t1_opp = prob_t2_wins_given_t1*prob_t2_makes_it_here*prob_t1_makes_it_here 
                    team_round_probs[team_1] = np.append(team_round_probs[team_1], np.array([prob_t1_wins_and_t2_opp]))
                    team_round_probs[team_2] = np.append(team_round_probs[team_2], np.array([prob_t2_wins_and_t1_opp]))
            teams_not_accounted_for = list(set(list(teams_this_year)) - set(list(teams_accounted_for)))
            if this_round == 0:
                for team_id in teams_not_accounted_for:
                    team_round_probs[team_id] = np.append(team_round_probs[team_id], np.array([1])) # these teams are teams in round 0 that dont play (100% chance of advancing)
            cumulative_probs = [np.sum(team_round_probs.get(team_id)) for team_id in teams_this_year]
            team_chances[f"Round_{this_round}"] = cumulative_probs
        
        team_chances = team_chances.rename(columns = {"Round_0" : 'Round of 64', "Round_1" : 'Round of 32', "Round_2" : "Sweet 16", "Round_3" : "Elite Eight",
                                            "Round_4" : "Final Four" , "Round_5" : "Championship", "Round_6" : "Champion"}).sort_values(by = ['Champion'], ascending=False)
        team_chances_numeric = team_chances.select_dtypes(include='number').multiply(100)
        team_chances = pd.concat([team_chances.select_dtypes(exclude='number'), team_chances_numeric], axis=1).reset_index(drop = 1).drop(['TeamID'], axis = 1).round(3)
        return(team_chances)

    
