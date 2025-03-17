import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
import itertools
import random
from scipy.interpolate import UnivariateSpline
import pickle
from pygam import LogisticGAM, s
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

season_game_results = pd.read_csv("data/game_results.csv").query("Season >= 2010")
tourney_seeds = pd.read_csv("data/tourney_seeds.csv").query("Season >= 2010")
tourney_results = pd.read_csv("data/tourney_results.csv").query("Season >= 2010")
tourney_slots = pd.read_csv("data/tourney_slots.csv")
teams = pd.read_csv("data/teams.csv")
all_effects = pd.read_csv("data/all_effects.csv")
all_years = season_game_results.Season.unique()
possible_num_games_back = [15]
possible_exponential_decays = [0.90]

cache = dict({year: 
              dict({
                  n_games : dict({
                      exp_dec : dict() 
                      for exp_dec in possible_exponential_decays}) 
                      for n_games in possible_num_games_back}) 
                      for year in all_years})
#with open("cache.pickle", "rb") as handle:
 #   cache = pickle.load(handle)

class OurMod:
    def __init__(self, train_x, train_y, with_glm = False): 
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.param = {} 
        self.param['eval_metric'] =  'mae'
        self.param['booster'] = 'gbtree'
        self.param['eta'] = 0.05 #change to ~0.02 for final run
        self.param['subsample'] = 0.35
        self.param['colsample_bytree'] = 0.7
        self.param['num_parallel_tree'] = 3 #recommend 10
        self.param['min_child_weight'] = 40
        self.param['gamma'] = 10
        self.param['max_depth'] =  3
        self.xgb = np.NaN
        self.train_x = train_x
        self.train_y = train_y
        self.spline = np.NaN
        self.with_glm = with_glm

    def fit(self):
        # Hyperparamter Tune (With repeated folds)
        xgb_cv = []
        repeat_cv = 3 # recommend 10
        dtrain = xgb.DMatrix(self.train_x.values, label = self.train_y) 
        for i in range(repeat_cv): 
            # print(f"Fold repeater {i}")
            xgb_cv.append(
                xgb.cv(
                    params = self.param,
                    dtrain = dtrain,
                    obj = cauchyobj,
                    num_boost_round = 3000,
                    folds = KFold(n_splits = 5, shuffle = True, random_state = i),
                    early_stopping_rounds = 25,
                    verbose_eval=False
                )
            )
        iteration_counts = [np.argmin(x['test-mae-mean'].values) for x in xgb_cv]
        val_mae = [np.min(x['test-mae-mean'].values) for x in xgb_cv]
        # print(f"iteration_counts : {iteration_counts}, val_mae : {val_mae}")

        # Train model with ideal parameters (for each repeated cv from above)
        oof_preds = []


        self.xgb = []
        xgb_index = 0
        val_true_records = {}
        val_pred_records = {}
        for i in range(repeat_cv):
            # print(f"Fold repeater {i}")
            preds = [self.train_y.copy()]
            kfold = KFold(n_splits = 5, shuffle = True, random_state = i)  
            for train_index, val_index in kfold.split(self.train_x.values,self.train_y):
                dtrain_i = xgb.DMatrix(self.train_x.values[train_index], label = self.train_y[train_index])
                dval_i = xgb.DMatrix(self.train_x.values[val_index], label = self.train_y[val_index])  
                self.xgb.append(xgb.train(
                    params = self.param,
                    dtrain = dtrain_i,
                    num_boost_round = iteration_counts[i],
                    verbose_eval = False
                ))
                preds[val_index] = self.xgb[xgb_index].predict(dval_i)
                xgb_index += 1
            oof_preds.append(np.clip(preds,-30,30))
        
        # Train multiple splines with cross validation (will average results)
        self.spline = []
        for i in range(repeat_cv):
            dat = list(zip(oof_preds[i],np.where(self.train_y>0,1,0)))
            dat = sorted(dat, key = lambda x: x[0])
            datdict = {}
            for k in range(len(dat)):
                datdict[dat[k][0]]= dat[k][1]
            self.spline.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
            spline_fit = self.spline[i](oof_preds[i])
            print(f"logloss of cvsplit {i}: {log_loss(np.where(self.train_y>0,1,0),spline_fit)}")

    def predict(self, x):
        x = xgb.DMatrix(x)
        pred_mov = []
        for i in range(len(self.xgb)):
            pred_mov.append(np.array(self.xgb[i].predict(x)))
        pred_mov = np.column_stack(pred_mov)
        return np.mean(pred_mov, axis = 1)
    
    def predict_proba(self, x):
        predict_mov = self.predict(x)
        all_pred_prob = []
        for i in range(len(self.spline)):
            predicted_probabilities = self.spline[i](predict_mov)
            predicted_probabilities = [x if x < 0.975 else 0.975 for x in predicted_probabilities]
            predicted_probabilities = [x if x > 0.025 else 0.025 for x in predicted_probabilities]
            all_pred_prob.append(np.array(predicted_probabilities))
        all_pred_prob = np.column_stack(all_pred_prob)
        return np.mean(all_pred_prob, axis = 1)

class team:
    def __init__(self, team_id: int, year: int, cache=cache) -> None:
        self.team_id = team_id
        self.year = year
        self.team_game_results = season_game_results.query(
            "TeamID == @team_id & Season == @year"
        )
        self.opp_game_results = season_game_results.query(
            "OppTeamID == @team_id & Season == @year"
        )
        self.cache = cache

    def get_past_n_record(self, n: int, exponential_downweight : float) -> float:
        """Get (percentage) record over past 10 games"""
        past_n_record = self.team_game_results.sort_values("DayNum", ascending=False).head(n).Result
        weights = np.array([exponential_downweight**i for i in range(len(past_n_record))])
        past_n_record = np.array(past_n_record*weights).sum() / weights.sum()

        return past_n_record

    def get_strength_of_schedule(self, n, exponential_downweight : float) -> float:
        """Get average overall kempom of past n game matchups"""
        last_n_opponents = self.team_game_results.sort_values(
            "DayNum", ascending=False
        ).OppTeamID.head(n)
        opp_ranks = all_effects.query("Season==@self.year & TeamID in @last_n_opponents").reset_index(drop = 1).Effect
        weights = np.array([exponential_downweight**i for i in range(len(opp_ranks))])
        return np.array(opp_ranks*weights).sum() / weights.sum()

    def get_road_record(self) -> float:
        """Get overall road record"""
        road_record = self.team_game_results.query("Loc == 'A'").Result.mean()
        return road_record

    def get_tourney_seed(self):
        """Get tourney seed"""
        try:
            return (
                tourney_seeds.query(f"Season == @self.year & TeamID == @self.team_id")
                .reset_index(drop=1)
                .Seed[0]
            )
        except KeyError:
            return 17  # If either (1) tourney seeds not released or (2) team did not make tourney assign fictional seed "17"

    def get_on_court_features(self, n_games_back, exponential_downweight):
        num_prev = len(self.team_game_results)
        weights = np.array([exponential_downweight**i for i in range(n_games_back)])[:num_prev]
        desired_vars = ["FGM",
                            "FGA",
                            "FGM3",
                            "FGA3",
                            "FTM",
                            "FTA",
                            "OR",
                            "DR",
                            "Ast",
                            "TO",
                            "Stl",
                            "Blk",
                            "mov"
                        ]
        court_features_mean = pd.DataFrame(
            {
                items[0]: [items[1]]
                for items in dict(
                    self.team_game_results.
                    sort_values("DayNum", ascending=False)
                    .head(n_games_back)[desired_vars]
                    .add_suffix(f"_mean")
                    .mul(weights, axis=0)
                    .mean()
                ).items()
            }
        )

        opp_court_features_mean = pd.DataFrame(
            {
                items[0]: [items[1]]
                for items in dict(
                    self.opp_game_results.
                    sort_values("DayNum", ascending=False)
                    .head(n_games_back)[desired_vars]
                    .add_suffix(f"_opp_mean")
                    .mul(weights, axis=0)
                    .mean()
                ).items()
            }
        )

        court_features_min = pd.DataFrame(
            {
                items[0]: [items[1]]
                for items in dict(
                    self.team_game_results.
                    sort_values("DayNum", ascending=False)
                    .head(n_games_back)[desired_vars]
                    .add_suffix(f"_min")
                    .mul(weights, axis=0)
                    .min()
                ).items()
            }
        )

        opp_court_features_min = pd.DataFrame(
            {
                items[0]: [items[1]]
                for items in dict(
                    self.opp_game_results.
                    sort_values("DayNum", ascending=False)
                    .head(n_games_back)[desired_vars]
                    .add_suffix(f"_opp_min")
                    .mul(weights, axis=0)
                    .min()
                ).items()
            }
        )

        court_features_max = pd.DataFrame(
            {
                items[0]: [items[1]]
                for items in dict(
                    self.team_game_results.
                    sort_values("DayNum", ascending=False)
                    .head(n_games_back)[desired_vars]
                    .add_suffix(f"_max")
                    .mul(weights, axis=0)
                    .max()
                ).items()
            }
        )

        opp_court_features_max = pd.DataFrame(
            {
                items[0]: [items[1]]
                for items in dict(
                    self.opp_game_results.
                    sort_values("DayNum", ascending=False)
                    .head(n_games_back)[desired_vars]
                    .add_suffix(f"_opp_max")
                    .mul(weights, axis=0)
                    .max()
                ).items()
            }
        )

        court_features = pd.concat([court_features_mean, court_features_min, opp_court_features_min,
                                    opp_court_features_mean, court_features_max, opp_court_features_max], axis=1)
        return court_features
    
    def get_mixed_effect_ranking(self):
        """Get rankings via mixed effects model from season games"""
        try:
            effect = all_effects.query("Season==@self.year & TeamID == @self.team_id").Effect.reset_index(drop = 1)[0]
        except KeyError:
            effect = np.NaN # If team not shown in data
        return effect
    
    def get_historical_mixed_effect_ranking(self, exponential_downweight):
        """Get rankings via mixed effects model from season games"""
        try:
            effect = all_effects.query("Season>=@self.year-3 & TeamID == @self.team_id").Effect.values
            weights = np.array([exponential_downweight**i for i in range(len(effect))])
            effect = np.mean(effect*weights)
        except KeyError:
            effect = np.NaN # If team not shown in data
        return effect


    def collect_features(self, n_games: int, exponential_downweight):
        """Collect current team level modeling features"""
        if self.team_id in self.cache.get(self.year).get(n_games)[exponential_downweight].keys():
            self.cache.get(self.year)[n_games][exponential_downweight][self.team_id]
        features = pd.DataFrame(
            dict(
                {
                    "pasn_n_road_record" : [self.get_past_n_record(n=n_games, exponential_downweight=exponential_downweight)],
                    "past_n_record": [self.get_past_n_record(n=n_games, exponential_downweight=exponential_downweight)],
                    "get_tourney_rank": [self.get_tourney_seed()],
                    "get_team_effect":[self.get_mixed_effect_ranking()],
                    "historical_team_effect":[self.get_historical_mixed_effect_ranking(exponential_downweight=exponential_downweight)],
                    "get_sos" : [self.get_strength_of_schedule(n=n_games, exponential_downweight=exponential_downweight)]
                }
            )
        )
        on_court_features = self.get_on_court_features(n_games_back = n_games, exponential_downweight=exponential_downweight)
        features = pd.concat([features, on_court_features], axis=1)
        self.cache.get(self.year)[n_games][exponential_downweight][self.team_id] = features
        return features


class playoff_matchup:
    def __init__(
        self, team_id_1: int, team_id_2: int, n_games: int, exponential_downweight : float, cache=cache
    ) -> None:
        """Init playoff_matchup class."""
        team_ids = [team_id_1, team_id_2]
        self.team_id_1 = min(team_ids)  # Ensures lower team_id is the one predicted for
        self.team_id_2 = max(team_ids)
        self.n_games = n_games
        self.exponential_downweight = exponential_downweight
        self.cache = cache

    def collect_features(self, year: int):
        """Collect features for playoff modeling"""
        if not (self.team_id_1 in self.cache.get(year)[self.n_games][self.exponential_downweight].keys()):
            team_1_class = team(team_id=self.team_id_1, year=year)
            team_1_features = team_1_class.collect_features(
                n_games=self.n_games,
                exponential_downweight=self.exponential_downweight
            )
        if not (self.team_id_2 in self.cache.get(year)[self.n_games][self.exponential_downweight].keys()):
            team_2_class = team(team_id=self.team_id_2, year=year)
            team_2_features = team_2_class.collect_features(
                n_games=self.n_games,
                exponential_downweight=self.exponential_downweight
            )
        team_1_features = pd.DataFrame(
            self.cache.get(year)[self.n_games][self.exponential_downweight].get(self.team_id_1)
        ).add_suffix(f"_T1")
        team_2_features = pd.DataFrame(
            self.cache.get(year)[self.n_games][self.exponential_downweight].get(self.team_id_2)
        ).add_suffix(f"_T2")
        features = pd.concat([team_1_features, team_2_features], axis=1)
        features["seed_diff"] = features["get_tourney_rank_T1"] - features["get_tourney_rank_T2"]
        features["effect_diff"] = features["get_team_effect_T1"] - features["get_team_effect_T2"]
        return features

    def predict(self, year, model):
        """Predicts probability of smaller team_id winning"""
        game_features = self.collect_features(year=year)
        predictions = model.predict(game_features)
        prob_lower_team_id = predictions[1]
        return prob_lower_team_id


class trainer:
    def __init__(self, n_games: int, exponential_downweight : float, with_glm = False):
        """Init class trainer."""
        self.features_men = np.NaN
        self.labels_men = np.NaN
        self.model_men = np.NaN
        self.season_key_men = np.NaN
        self.with_glm = with_glm
        self.exponential_downweight=exponential_downweight

        self.features_women = np.NaN
        self.labels_women = np.NaN
        self.model_women = np.NaN
        self.season_key_women = np.NaN

        self.all_years = tourney_results.Season.unique()
        self.n_games = n_games

    def get_features_labels(self, gender: str):
        """Gets features and labels for all years for a certain gender."""
        all_features_list = []
        all_labels = np.array([])
        season_keys = np.array([])

        for year in tqdm(self.all_years):
            tourney_results_year = tourney_results.query(
                "Season == @year & M_W == @gender"
            )
            for index, row in enumerate(tourney_results_year.itertuples()):
                team_ids = [row.WTeamID, row.LTeamID]
                team_id_1 = min(team_ids)
                if row.WTeamID == team_id_1:
                    label = row.WScore - row.LScore
                else:
                    label = row.LScore - row.WScore
                all_labels = np.append(all_labels, label)
                matchup_class = playoff_matchup(
                    min(team_ids),
                    max(team_ids),
                    n_games=self.n_games,
                    exponential_downweight=self.exponential_downweight
                )
                this_game_features = matchup_class.collect_features(year=row.Season)
                all_features_list.append(this_game_features.copy())
                season_keys = np.append(season_keys, year)
        all_features = pd.concat(all_features_list)
        return dict(
            {
                "all_features": all_features,
                "all_labels": all_labels,
                "season_keys": season_keys,
            }
        )

    def label_and_data_maker(self):
        """Create training and label sets for all NCAA matchups (men and women)"""
        print("Getting Mens Training Data:")
        men = self.get_features_labels(gender="M")
        print("Getting Womens Training Data:")
        women = self.get_features_labels(gender="W")

        self.features_men = men["all_features"]
        self.labels_men = men["all_labels"]
        self.season_key_men = men["season_keys"]
        self.features_women = women["all_features"]
        self.labels_women = women["all_labels"]
        self.season_key_women = women["season_keys"]

    def train(self, verbose = True):
        """Trains Models for both MNCAA and WNCAA"""
        max_year = int(self.season_key_men.max()) + 1
        all_mses = []

        # Hyperparameter tune
        for year in range(2021, max_year):
            if year == 2020:
                continue
            X_train_men, X_test_men, y_train_men, y_test_men = (
                self.features_men[self.season_key_men < year],
                self.features_men[self.season_key_men == year],
                self.labels_men[self.season_key_men < year],
                self.labels_men[self.season_key_men == year],
            )
            self.model_men = OurMod(X_train_men, y_train_men, with_glm=self.with_glm)
            print("Training Mens Model")
            self.model_men.fit()

            X_train_women, X_test_women, y_train_women, y_test_women = (
                self.features_women[self.season_key_women < year],
                self.features_women[self.season_key_women == year],
                self.labels_women[self.season_key_women < year],
                self.labels_women[self.season_key_women == year],
            )
            model_women = OurMod(X_train_women, y_train_women, with_glm=self.with_glm)
            print("Training Womens Model")
            model_women.fit()

            pred_men_prob = self.model_men.predict_proba(X_test_men).tolist()
            pred_women_prob = model_women.predict_proba(X_test_women).tolist()
            pred_men_binary = [x > 0.5 for x in pred_men_prob]  # prob greater than 0.5
            pred_women_binary = [x > 0.5 for x in pred_women_prob]
            y_test_men_binary = [x > 0 for x in y_test_men]  # mov greater than 0
            y_test_women_binary = [x > 0 for x in y_test_women]
            pred_overall_prob, y_test_overall_binary = pred_men_prob.copy(), y_test_men_binary.copy()
            pred_overall_prob.extend(pred_women_prob)
            y_test_overall_binary.extend(y_test_women_binary)

            # log_loss(np.where(self.train_y>0,1,0),spline_fit)
            
            
            accuracy_men = accuracy_score(y_test_men_binary, pred_men_binary)
            mse_men = mean_squared_error(y_test_men_binary, pred_men_prob)
            log_loss_men = log_loss(y_test_men_binary, pred_men_prob)
            accuracy_women = accuracy_score(y_test_women_binary, pred_women_binary)
            mse_women = mean_squared_error(y_test_women_binary, pred_women_prob)
            log_loss_women = log_loss(y_test_women_binary, pred_women_prob)
            mse_overall = mean_squared_error(y_test_overall_binary, pred_overall_prob)
            all_mses.append(mse_overall)
            if verbose:
                print(
                    f"Mens accuracy for {year} is {round(accuracy_men * 100, 2)} percent. MSE is {round(mse_men, 6)}. Log loss is {round(log_loss_men, 6)}"
                )
                print(
                    f"Womens accuracy for {year} is {round(accuracy_women * 100, 2)} percent. MSE is {round(mse_women, 6)}. Log loss is {round(log_loss_women, 6)}"
                )
                print(
                    f"Overall MSE for {year} is {round(mse_overall, 6)}"
                )
        X_train_men, y_train_men = (
            self.features_men[self.season_key_men < max_year],
            self.labels_men[self.season_key_men < max_year],
        )
        self.model_men = OurMod(X_train_men, y_train_men, with_glm=self.with_glm)
        self.model_men.fit()

        X_train_women, y_train_women = (
            self.features_women[self.season_key_women < max_year],
            self.labels_women[self.season_key_women < max_year],
        )
        self.model_women = OurMod(X_train_women, y_train_women, with_glm=self.with_glm)
        self.model_women.fit()
        return all_mses


class tournament:
    def __init__(self, year, mens_model, womens_model, n_games, exponential_downweight):
        self.year = year
        self.n_games = n_games
        self.this_year_tourney_teams_mens = tourney_seeds.query(
            "Season == @year & M_W == 'M'"
        ).TeamID
        self.this_year_tourney_teams_womens = tourney_seeds.query(
            "Season == @year & M_W == 'W'"
        ).TeamID
        self.mens_model = mens_model
        self.womens_model = womens_model
        self.mens_features = np.NaN
        self.womens_features = np.NaN
        self.predictions = np.NaN
        self.exponential_downweight = exponential_downweight

    def predict_all_including_non_possible(self):
        """Predicts probabilities for every pair of matchups for all teams in tournement"""
        # get all possible matchups
        all_womens_matchups = list(
            itertools.combinations(
                tourney_seeds.query(
                    "Season == @self.year &  M_W == 'W'"
                ).TeamID.unique(),
                2,
            )
        )
        all_mens_matchups = list(
                    itertools.combinations(
                        tourney_seeds.query(
                            "Season == @self.year &  M_W == 'M'"
                        ).TeamID.unique(),
                        2,
                    )
                )

        womens_features = pd.DataFrame()
        womens_matchup_strings = []
        mens_features = pd.DataFrame()
        mens_matchup_strings = []
        for matchup in tqdm(all_mens_matchups):
            matchup_list = [matchup[0], matchup[1]]
            matchup_class = playoff_matchup(
                team_id_1=matchup[0],
                team_id_2=matchup[1],
                n_games=self.n_games,
                exponential_downweight=self.exponential_downweight
                )
            matchup_features = matchup_class.collect_features(year=self.year)
            mens_features = pd.concat([mens_features.copy(), matchup_features.copy()])
            matchup_string = f"{self.year}_{min(matchup_list)}_{max(matchup_list)}"
            mens_matchup_strings.append(matchup_string)
        for matchup in tqdm(all_womens_matchups):
            matchup_list = [matchup[0], matchup[1]]
            matchup_class = playoff_matchup(
                team_id_1=matchup[0],
                team_id_2=matchup[1],
                n_games=self.n_games,
                exponential_downweight=self.exponential_downweight
                )
            matchup_features = matchup_class.collect_features(year=self.year)
            womens_features = pd.concat(
                [womens_features.copy(), matchup_features.copy()]
            )
            matchup_string = f"{self.year}_{min(matchup_list)}_{max(matchup_list)}"
            womens_matchup_strings.append(matchup_string)


        mens_predictions = self.mens_model.predict_proba(mens_features)
        womens_predictions = self.womens_model.predict_proba(womens_features)
        self.predictions = pd.DataFrame(
            {
                "ID": np.append(mens_matchup_strings, womens_matchup_strings),
                "Pred": np.append(mens_predictions, womens_predictions),
            }
        )
        return self.predictions

    def predict_exact_tourney(self, gender: str, print_results=True):
        """Predicts binary outputs for each progressive round of tourney"""
        # if not run simulate all do that
        if not isinstance(self.predictions, pd.DataFrame):
            print("Predictions need to be run for this tourney, running now...")
            predictions = self.predict_all_including_non_possible()
        tourney_seeds_this_year = tourney_seeds.query(
            "Season == @self.year & M_W == @gender"
        )
        tourney_slots_this_year = tourney_slots.query(
            "Season == @self.year & M_W == @gender"
        )
        updating_seed_record = dict(
            {
                tourney_seeds_this_year.query("TeamID == @team_id")
                .reset_index(drop=1)
                .ReigonSeed[0]: team_id
                for team_id in tourney_seeds_this_year.TeamID
            }
        )
        for game_round in np.sort(tourney_slots.GameRound.unique()):
            slots_this_round = tourney_slots_this_year.query("GameRound == @game_round")
            if print_results:
                print(f"Simulating Round {game_round}...")
            for index, row in enumerate(slots_this_round.itertuples()):
                teams_in_matchup = [
                    updating_seed_record.get(row.StrongSeed),
                    updating_seed_record.get(row.WeakSeed),
                ]
                team_1 = min(teams_in_matchup)
                team_2 = max(teams_in_matchup)
                id_in_probs = f"{self.year}_{team_1}_{team_2}"
                prediction = (
                    self.predictions.query("ID == @id_in_probs")
                    .reset_index(drop=1)
                    .Pred[0]
                )
                if random.random() <= prediction:
                    updating_seed_record[row.Slot] = team_1
                    team_name_win = (
                        teams.query("TeamID == @team_1").reset_index(drop=1).TeamName[0]
                    )
                    team_name_lost = (
                        teams.query("TeamID == @team_2").reset_index(drop=1).TeamName[0]
                    )
                    if teams_in_matchup[0] == team_1:
                        win_seed = row.StrongSeed
                        lose_seed = row.WeakSeed
                        win_prob = prediction
                else:
                    updating_seed_record[row.Slot] = team_2
                    team_name_win = (
                        teams.query("TeamID == @team_2").reset_index(drop=1).TeamName[0]
                    )
                    team_name_lost = (
                        teams.query("TeamID == @team_1").reset_index(drop=1).TeamName[0]
                    )
                    if teams_in_matchup[0] == team_1:
                        lose_seed = row.StrongSeed
                        win_seed = row.WeakSeed
                        win_prob = 1 - prediction
                if print_results:
                    print(
                        f"Seed {win_seed}. {team_name_win} beats seed {lose_seed} {team_name_lost} (with probability {win_prob})"
                    )
        return updating_seed_record

    def get_all_possible_in_slot(self, slot, gender):
        """Recursively retrieve all possible teams that could occupy a bracket slot."""
        all_slots = tourney_slots.query("Season == @self.year & M_W == @gender")
        tourney_seeds_this_year = tourney_seeds.query("Season == 2023 & M_W == 'M'")
        try:
            slot_documentation = (
                all_slots.query("Slot == @slot").reset_index(drop=1).iloc[0]
            )
        except IndexError:
            return slot
        strong_slot_seed = slot_documentation.StrongSeed
        weak_slot_seed = slot_documentation.WeakSeed
        all_possible_strong_side = self.get_all_possible_in_slot(
            strong_slot_seed, gender
        )
        all_possible_weak_side = self.get_all_possible_in_slot(weak_slot_seed, gender)
        return np.append(all_possible_strong_side, all_possible_weak_side)

    def get_one_round_back_ansestors(self, slot, gender):
        all_slots = tourney_slots.query("Season == @self.year & M_W == @gender")
        slot_documentation = (
            all_slots.query("Slot == @slot").reset_index(drop=1).iloc[0]
        )
        return np.array([slot_documentation.StrongSeed, slot_documentation.WeakSeed])

    def get_team_round_probabilities(self, gender):
        slots_this_year = tourney_slots.query("Season == @self.year & M_W == @gender")
        seeds_this_year = tourney_seeds.query("Season == @self.year & M_W == @gender")
        teams_this_year = seeds_this_year.TeamID.unique()
        team_chances = pd.DataFrame(
            {
                "TeamID": teams_this_year,
                "TeamName": [
                    teams.query("TeamID == @team_id").reset_index(0).iloc[0].TeamName
                    for team_id in teams_this_year
                ],
            }
        )
        for this_round in [0, 1, 2, 3, 4, 5, 6]:
            slots_this_round = slots_this_year.query("GameRound == @this_round").Slot
            team_round_probs = dict(
                {team_id: np.array([]) for team_id in teams_this_year}
            )
            teams_accounted_for = []
            for slot in slots_this_round:
                ansestor_slot_1, ansestor_slot_2 = self.get_one_round_back_ansestors(
                    slot, gender
                )
                possible_seeds_ansestor_1 = self.get_all_possible_in_slot(
                    slot=ansestor_slot_1, gender=gender
                )
                possible_seeds_ansestor_2 = self.get_all_possible_in_slot(
                    slot=ansestor_slot_2, gender=gender
                )
                possible_team_ids_ansestor_1 = seeds_this_year.query(
                    "ReigonSeed in @possible_seeds_ansestor_1"
                ).TeamID.tolist()
                possible_team_ids_ansestor_2 = seeds_this_year.query(
                    "ReigonSeed in @possible_seeds_ansestor_2"
                ).TeamID.tolist()
                possible_team_ids_all = np.append(
                    possible_team_ids_ansestor_1, possible_team_ids_ansestor_2
                )
                teams_accounted_for.extend(possible_team_ids_all)
                possible_slot_matchups = list(
                    itertools.product(
                        possible_team_ids_ansestor_1, possible_team_ids_ansestor_2
                    )
                )
                possible_game_ids = [
                    f"{self.year}_{min(matchup_teams)}_{max(matchup_teams)}"
                    for matchup_teams in possible_slot_matchups
                ]
                for game_id in possible_game_ids:
                    team_1 = int(game_id[5:9])
                    team_2 = int(game_id[10:14])
                    prob_t1_wins_given_t2 = (
                        self.predictions.query("ID == @game_id")
                        .reset_index(drop=1)
                        .Pred[0]
                    )  # P(T1 Wins | T2 Opponent)
                    prob_t2_wins_given_t1 = 1 - prob_t1_wins_given_t2
                    if this_round == 0:
                        prob_t1_makes_it_here = 1
                        prob_t2_makes_it_here = 1
                    else:
                        prob_t1_makes_it_here = (
                            team_chances.query("TeamID == @team_1")
                            .reset_index(drop=1)
                            .iloc[0][f"Round_{this_round - 1}"]
                        )  # P(T1 Opponent)
                        prob_t2_makes_it_here = (
                            team_chances.query("TeamID == @team_2")
                            .reset_index(drop=1)
                            .iloc[0][f"Round_{this_round - 1}"]
                        )
                    prob_t1_wins_and_t2_opp = (
                        prob_t1_wins_given_t2
                        * prob_t1_makes_it_here
                        * prob_t2_makes_it_here
                    )  # P(T1 wins and T2 Opponent) = P(T1 Wins | T1 makes, T2 makes) * P(T2 Opponent) * P(T1 makes)
                    prob_t2_wins_and_t1_opp = (
                        prob_t2_wins_given_t1
                        * prob_t2_makes_it_here
                        * prob_t1_makes_it_here
                    )
                    team_round_probs[team_1] = np.append(
                        team_round_probs[team_1], np.array([prob_t1_wins_and_t2_opp])
                    )
                    team_round_probs[team_2] = np.append(
                        team_round_probs[team_2], np.array([prob_t2_wins_and_t1_opp])
                    )
            teams_not_accounted_for = list(
                set(list(teams_this_year)) - set(list(teams_accounted_for))
            )
            if this_round == 0:
                for team_id in teams_not_accounted_for:
                    team_round_probs[team_id] = np.append(
                        team_round_probs[team_id], np.array([1])
                    )  # these teams are teams in round 0 that dont play (100% chance of advancing)
            cumulative_probs = [
                np.sum(team_round_probs.get(team_id)) for team_id in teams_this_year
            ]
            team_chances[f"Round_{this_round}"] = cumulative_probs

        team_chances = team_chances.rename(
            columns={
                "Round_0": "Round of 64",
                "Round_1": "Round of 32",
                "Round_2": "Sweet 16",
                "Round_3": "Elite Eight",
                "Round_4": "Final Four",
                "Round_5": "Championship",
                "Round_6": "Champion",
            }
        ).sort_values(by=["Champion"], ascending=False)
        team_chances_numeric = team_chances.select_dtypes(include="number").multiply(
            100
        )
        team_chances = (
            pd.concat(
                [team_chances.select_dtypes(exclude="number"), team_chances_numeric],
                axis=1,
            )
            .reset_index(drop=1)
            .drop(["TeamID"], axis=1)
            .round(3)
        )
        return team_chances
    
# Define the Cauchy objective function
def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000 
    x =  preds-labels    
    grad = x / (x**2/c**2+1)
    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
    return grad, hess


# with open('cache.pickle', 'wb') as handle:
#    pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
