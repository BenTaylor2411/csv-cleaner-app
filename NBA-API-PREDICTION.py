import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import math
import itertools
import threading

from nba_api.stats.static import players as players_static
from nba_api.stats.static import teams as teams_static
from nba_api.stats.endpoints import (
    playergamelog,
    commonplayerinfo,
    commonteamroster,
    leaguestandings,
    leaguedashteamstats,
)
from nba_api.stats.library.parameters import SeasonAll

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# LOAD PLAYERS + TEAMS


all_players = players_static.get_players()
all_teams = teams_static.get_teams()

player_id_map = {}   
team_map = {}        


team_strength_cache = {}
team_reb_rating_cache = {}
team_3pt_def_rating_cache = {}
pace_data_cache = {}
team_pace_cache = {}


player_games_cache = {}  
player_info_cache = {}   


fig = None
ax = None
canvas = None


# MULTI BUILDER GLOBALS

saved_legs = []  


# A) THREADING HELPERS

calc_in_progress = False
calc_btn = None
status_lbl = None

def run_in_ui(fn):
 
    try:
        root.after(0, fn)
    except Exception:
        pass


# BASIC HELPERS


def season_int_to_str(season_int: int) -> str:
    start = season_int
    end_short = str(season_int + 1)[-2:]
    return f"{start}-{end_short}"


def _abbr_alias(abbr: str) -> str:
    alias_map = {
        "BRK": "BKN",
        "PHO": "PHX",
        "CHO": "CHA",
    }
    return alias_map.get(abbr, abbr)


def get_team_by_abbr(abbr: str):
    abbr = _abbr_alias(abbr)
    for t in all_teams:
        if t.get("abbreviation") == abbr:
            return t
    return None


def search_players(query: str):
    q = query.lower()
    return [p for p in all_players if q in p["full_name"].lower()]



# PLAYER GAME DATA


def get_player_all_games(player_id: int) -> pd.DataFrame:
    gl = playergamelog.PlayerGameLog(player_id=player_id, season=SeasonAll.all)
    df = gl.get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON"] = df["SEASON_ID"].astype(str).str[-4:].astype(int)
    df = df.sort_values("GAME_DATE")
    return df


def get_player_all_games_cached(player_id: int) -> pd.DataFrame:
    if player_id in player_games_cache:
        return player_games_cache[player_id].copy()
    df = get_player_all_games(player_id)
    player_games_cache[player_id] = df
    return df.copy()


def add_opponent_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def get_opp(row):
        matchup = row.get("MATCHUP", "")
        parts = matchup.split()
        if len(parts) != 3:
            return None
        return parts[2]

    df["OPP_ABBR"] = df.apply(get_opp, axis=1)
    return df


def get_current_season_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    current_season = df["SEASON"].max()
    return df[df["SEASON"] == current_season].copy()


def get_vs_opp_current_and_last_season(df: pd.DataFrame, opp_abbr: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    seasons_sorted = sorted(df["SEASON"].unique())
    if len(seasons_sorted) >= 2:
        wanted_seasons = seasons_sorted[-2:]
    else:
        wanted_seasons = seasons_sorted

    mask = (df["SEASON"].isin(wanted_seasons)) & (df["OPP_ABBR"] == opp_abbr)
    return df[mask].copy()



# TEAM STRENGTH (STANDINGS, 1–5)


def get_team_strength_from_standings(team_id: int, season_int: int) -> float:
    key = (team_id, season_int)
    if key in team_strength_cache:
        return team_strength_cache[key]

    season_str = season_int_to_str(season_int)
    strength = 3.0

    try:
        ls = leaguestandings.LeagueStandings(season=season_str)
        df = ls.get_data_frames()[0]

        id_col = None
        for cand in ["TeamID", "TEAM_ID"]:
            if cand in df.columns:
                id_col = cand
                break

        if id_col is None:
            team_strength_cache[key] = strength
            return strength

        df["TEAM_ID_INT"] = pd.to_numeric(df[id_col], errors="coerce")
        row = df[df["TEAM_ID_INT"] == int(team_id)]

        if row.empty:
            team_strength_cache[key] = strength
            return strength

        row = row.iloc[0]

        rank_col = None
        for cand in ["PlayoffRank", "LeagueRank", "LEAGUE_RANK", "ConfRank", "ConferenceRank"]:
            if cand in df.columns:
                rank_col = cand
                break

        if rank_col is not None:
            all_ranks = pd.to_numeric(df[rank_col], errors="coerce")
            max_rank = all_ranks.max()
            this_rank = pd.to_numeric(pd.Series([row[rank_col]]), errors="coerce").iloc[0]

            if pd.notna(this_rank) and pd.notna(max_rank) and max_rank > 1:
                strength = 1.0 + (max_rank - this_rank) * 4.0 / (max_rank - 1)
                strength = float(np.clip(strength, 1.0, 5.0))
                team_strength_cache[key] = strength
                return strength

        win_col = None
        for cand in ["WinPCT", "WIN_PCT"]:
            if cand in df.columns:
                win_col = cand
                break

        if win_col is not None:
            df["WIN_FLOAT"] = pd.to_numeric(df[win_col], errors="coerce")
            max_w = df["WIN_FLOAT"].max()
            min_w = df["WIN_FLOAT"].min()
            this_w = pd.to_numeric(pd.Series([row[win_col]]), errors="coerce").iloc[0]

            if pd.notna(this_w) and pd.notna(max_w) and pd.notna(min_w) and max_w > min_w:
                strength = 1.0 + (this_w - min_w) * 4.0 / (max_w - min_w)
                strength = float(np.clip(strength, 1.0, 5.0))

    except Exception:
        strength = 3.0

    team_strength_cache[key] = strength
    return strength



# TEAM REBOUNDING (AUTO, 1–10)


def get_team_reb_rating(team_id: int, season_int: int) -> float:
    key = (team_id, season_int)
    if key in team_reb_rating_cache:
        return team_reb_rating_cache[key]

    season_str = season_int_to_str(season_int)
    rating = 5.5

    try:
        lds = leaguedashteamstats.LeagueDashTeamStats(
            season=season_str,
            per_mode_detailed="PerGame"
        )
        df = lds.get_data_frames()[0]

        if "TEAM_ID" not in df.columns or "REB" not in df.columns:
            team_reb_rating_cache[key] = rating
            return rating

        df["TEAM_ID_INT"] = pd.to_numeric(df["TEAM_ID"], errors="coerce")
        df["REB"] = pd.to_numeric(df["REB"], errors="coerce")

        row = df[df["TEAM_ID_INT"] == int(team_id)]
        if row.empty:
            team_reb_rating_cache[key] = rating
            return rating

        row = row.iloc[0]
        this_reb = row["REB"]
        if pd.isna(this_reb):
            team_reb_rating_cache[key] = rating
            return rating

        min_reb = df["REB"].min()
        max_reb = df["REB"].max()

        if pd.notna(min_reb) and pd.notna(max_reb) and max_reb > min_reb:
            rating = 1.0 + (this_reb - min_reb) * 9.0 / (max_reb - min_reb)
            rating = float(np.clip(rating, 1.0, 10.0))

    except Exception:
        rating = 5.5

    team_reb_rating_cache[key] = rating
    return rating



# TEAM 3-POINT DEFENSE (AUTO, 1–10)


def get_team_3pt_def_rating(team_id: int, season_int: int) -> float:
    key = (team_id, season_int)
    if key in team_3pt_def_rating_cache:
        return team_3pt_def_rating_cache[key]

    season_str = season_int_to_str(season_int)
    rating = 5.5

    try:
        try:
            lds = leaguedashteamstats.LeagueDashTeamStats(
                season=season_str,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Opponent"
            )
        except TypeError:
            lds = leaguedashteamstats.LeagueDashTeamStats(
                season=season_str,
                per_mode_detailed="PerGame"
            )

        df = lds.get_data_frames()[0]

        if "TEAM_ID" not in df.columns:
            team_3pt_def_rating_cache[key] = rating
            return rating

        df["TEAM_ID_INT"] = pd.to_numeric(df["TEAM_ID"], errors="coerce")

        if "OPP_FG3_PCT" not in df.columns:
            team_3pt_def_rating_cache[key] = rating
            return rating

        df["OPP_FG3_PCT"] = pd.to_numeric(df["OPP_FG3_PCT"], errors="coerce")

        row = df[df["TEAM_ID_INT"] == int(team_id)]
        if row.empty:
            team_3pt_def_rating_cache[key] = rating
            return rating

        row = row.iloc[0]
        this_opp_3p = row["OPP_FG3_PCT"]
        if pd.isna(this_opp_3p):
            team_3pt_def_rating_cache[key] = rating
            return rating

        min_opp_3p = df["OPP_FG3_PCT"].min()
        max_opp_3p = df["OPP_FG3_PCT"].max()

        if pd.notna(min_opp_3p) and pd.notna(max_opp_3p) and max_opp_3p > min_opp_3p:
            rating = 1.0 + (max_opp_3p - this_opp_3p) * 9.0 / (max_opp_3p - min_opp_3p)
            rating = float(np.clip(rating, 1.0, 10.0))

    except Exception:
        rating = 5.5

    team_3pt_def_rating_cache[key] = rating
    return rating



# PACE (AUTO)


def get_pace_data(season_int: int):
    if season_int in pace_data_cache:
        return pace_data_cache[season_int]

    season_str = season_int_to_str(season_int)
    try:
        try:
            lds = leaguedashteamstats.LeagueDashTeamStats(
                season=season_str,
                per_mode_detailed="PerGame",
                measure_type_detailed="Advanced"
            )
        except TypeError:
            lds = leaguedashteamstats.LeagueDashTeamStats(
                season=season_str,
                per_mode_detailed="PerGame"
            )
        df = lds.get_data_frames()[0]
        if "PACE" not in df.columns or "TEAM_ID" not in df.columns:
            pace_data_cache[season_int] = (None, None)
            return pace_data_cache[season_int]

        df["PACE"] = pd.to_numeric(df["PACE"], errors="coerce")
        league_avg = df["PACE"].mean()
        pace_data_cache[season_int] = (df, league_avg)
        return pace_data_cache[season_int]
    except Exception:
        pace_data_cache[season_int] = (None, None)
        return pace_data_cache[season_int]


def get_team_pace(team_id: int, season_int: int):
    key = (team_id, season_int)
    if key in team_pace_cache:
        return team_pace_cache[key], pace_data_cache.get(season_int, (None, None))[1]

    df, league_avg = get_pace_data(season_int)
    if df is None or "TEAM_ID" not in df.columns or "PACE" not in df.columns:
        team_pace_cache[key] = None
        return None, league_avg

    df["TEAM_ID_INT"] = pd.to_numeric(df["TEAM_ID"], errors="coerce")
    row = df[df["TEAM_ID_INT"] == int(team_id)]
    if row.empty:
        team_pace_cache[key] = None
        return None, league_avg

    pace_val = float(row.iloc[0]["PACE"])
    team_pace_cache[key] = pace_val
    return pace_val, league_avg



# MARKET / METRIC HELPERS


def add_market_value_column(df: pd.DataFrame, market_code: str) -> pd.DataFrame:
    df = df.copy()

    for col in ["PTS", "REB", "AST", "FG3M"]:
        if col not in df.columns:
            df[col] = 0.0

    if market_code == "PTS":
        df["VALUE"] = df["PTS"]
    elif market_code == "REB":
        df["VALUE"] = df["REB"]
    elif market_code == "AST":
        df["VALUE"] = df["AST"]
    elif market_code == "PRA":
        df["VALUE"] = df["PTS"] + df["REB"] + df["AST"]
    elif market_code == "PR":
        df["VALUE"] = df["PTS"] + df["REB"]
    elif market_code == "PA":
        df["VALUE"] = df["PTS"] + df["AST"]
    elif market_code == "RA":
        df["VALUE"] = df["REB"] + df["AST"]
    elif market_code == "3PM":
        df["VALUE"] = df["FG3M"]
    else:
        df["VALUE"] = df["PTS"]

    return df


def get_points_weight_for_market(market_code: str) -> float:
    mapping = {
        "PTS": 1.0,
        "REB": 0.0,
        "AST": 0.6,
        "PRA": 0.7,
        "PR": 0.7,
        "PA": 0.8,
        "RA": 0.2,
        "3PM": 1.0,
    }
    return mapping.get(market_code, 0.5)


def get_rebound_weight_for_market(market_code: str) -> float:
    mapping = {
        "PTS": 0.0,
        "REB": 1.0,
        "AST": 0.0,
        "PRA": 0.6,
        "PR": 0.7,
        "PA": 0.1,
        "RA": 0.9,
        "3PM": 0.0,
    }
    return mapping.get(market_code, 0.3)


def get_three_weight_for_market(market_code: str) -> float:
    mapping = {
        "3PM": 1.0,
        "PTS": 0.5,
        "PRA": 0.4,
        "PR": 0.3,
        "PA": 0.2,
        "REB": 0.0,
        "AST": 0.1,
        "RA": 0.0,
    }
    return mapping.get(market_code, 0.2)


def market_label(market_code: str) -> str:
    labels = {
        "PTS": "Points",
        "REB": "Rebounds",
        "AST": "Assists",
        "PRA": "Points+Rebounds+Assists (PRA)",
        "PR":  "Points+Rebounds (PR)",
        "PA":  "Points+Assists (PA)",
        "RA":  "Rebounds+Assists (RA)",
        "3PM": "3-Pointers Made (3PM)",
    }
    return labels.get(market_code, market_code)



# PROBABILITY ENGINE


def hit_rate(df: pd.DataFrame, line_val: float):
    if df is None or df.empty or "VALUE" not in df.columns:
        return None
    return (df["VALUE"] >= line_val).mean()


def hit_count(df: pd.DataFrame, line_val: float):
    if df is None or df.empty or "VALUE" not in df.columns:
        return 0, 0
    total = len(df)
    hits = int((df["VALUE"] >= line_val).sum())
    return hits, total


def compute_usage_proxy(df: pd.DataFrame):
    df = df.copy()
    if "MIN" not in df.columns:
        if "MINUTES" in df.columns:
            df["MIN"] = df["MINUTES"]
        else:
            df["MIN"] = 30.0
    df["USG_PROXY"] = df["PTS"] + df["REB"] + df["AST"] + 0.25 * df["MIN"]
    return df


def combine_weighted_rates(rates_with_weights):
    valid = [(r, w, label) for (r, w, label) in rates_with_weights if r is not None]
    if not valid:
        return None, []

    total_w = sum(w for (r, w, label) in valid)
    if total_w <= 0:
        return None, []

    combined = sum(r * w for (r, w, label) in valid) / total_w
    breakdown = [(label, r, w / total_w) for (r, w, label) in valid]
    return combined, breakdown


def parse_minutes_column(df: pd.DataFrame) -> pd.Series:
    s = df["MIN"]

    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)

    def to_min(val):
        try:
            if isinstance(val, (int, float)):
                return float(val)
            text = str(val)
            if ":" in text:
                mm, ss = text.split(":")
                return float(mm) + float(ss) / 60.0
            return float(text)
        except Exception:
            return np.nan

    return s.map(to_min)


def compute_projection_probability(df_for_proj: pd.DataFrame, line_val: float):
    info = {
        "projected_minutes": None,
        "per_min_rate": None,
        "projected_mean": None,
        "std": None,
        "sample_size": 0,
    }

    if df_for_proj is None or df_for_proj.empty or "VALUE" not in df_for_proj.columns:
        return None, info

    if "MIN" not in df_for_proj.columns:
        return None, info

    mins = parse_minutes_column(df_for_proj)
    vals = df_for_proj["VALUE"].astype(float)

    mask = (~mins.isna()) & (mins > 0) & (~vals.isna())
    if mask.sum() < 5:
        info["sample_size"] = int(mask.sum())
        return None, info

    mins_valid = mins[mask]
    vals_valid = vals[mask]
    info["sample_size"] = int(mask.sum())

    idx_last10 = mins_valid.tail(10).index
    if len(idx_last10) > 0:
        projected_minutes = float(mins_valid.loc[idx_last10].mean())
    else:
        projected_minutes = float(mins_valid.mean())

    per_min_rate = float(vals_valid.sum() / mins_valid.sum())
    projected_mean = per_min_rate * projected_minutes

    std = float(vals_valid.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        std = abs(projected_mean) * 0.3 + 1e-3

    info["projected_minutes"] = projected_minutes
    info["per_min_rate"] = per_min_rate
    info["projected_mean"] = projected_mean
    info["std"] = std

    z = (line_val - projected_mean) / std
    p_proj = 0.5 * math.erfc(z / math.sqrt(2.0))
    p_proj = float(np.clip(p_proj, 0.01, 0.99))

    return p_proj, info


def calculate_probability(
    df_all,
    df_season,
    df_vs_opp,
    is_home,
    line_val,
    manual_adjust_frac,
    defender_rating,
    opp_reb_def_rating,
    opp_3pt_def_rating,
    player_team_strength,
    opp_team_strength,
    market_code,
    projection_prob=None,
    pace_adjust=0.0,
):
    season_rate  = hit_rate(df_season, line_val)
    last10_rate  = hit_rate(df_all.tail(10), line_val)
    last3_rate   = hit_rate(df_all.tail(3), line_val)
    opp_all_rate = hit_rate(df_vs_opp, line_val)

    rates = [
        (season_rate,  0.30,  "This season"),
        (last10_rate,  0.325, "Last 10 games"),
        (last3_rate,   0.225, "Last 3 games"),
        (opp_all_rate, 0.15,  "Vs opponent (current + last season)"),
    ]

    base_prob, breakdown = combine_weighted_rates(rates)
    if base_prob is None:
        return {
            "prob": None,
            "fair_odds": None,
            "breakdown": breakdown,
            "usage_info": None,
            "defense_info": None,
            "blowout_info": None,
            "hist_prob": None,
            "projection_prob": projection_prob,
            "pace_adjust": pace_adjust,
        }

    df_all_usage = compute_usage_proxy(df_all)
    overall_usage = df_all_usage["USG_PROXY"].mean()
    recent10 = df_all_usage.tail(10)
    recent_usage = recent10["USG_PROXY"].mean() if not recent10.empty else overall_usage

    if overall_usage > 0:
        usage_diff_frac = (recent_usage - overall_usage) / overall_usage
    else:
        usage_diff_frac = 0.0

    usage_adjust = np.clip(usage_diff_frac * 0.1, -0.05, 0.05)

    points_weight = get_points_weight_for_market(market_code)
    reb_weight    = get_rebound_weight_for_market(market_code)
    three_weight  = get_three_weight_for_market(market_code)

    if opp_reb_def_rating is None:
        opp_reb_def_rating = 5.5
    reb_scale = (opp_reb_def_rating - 5.5) / 4.5
    reb_def_adjust = -reb_scale * (0.05 * reb_weight)

    if opp_3pt_def_rating is None:
        opp_3pt_def_rating = 5.5
    three_scale = (opp_3pt_def_rating - 5.5) / 4.5
    three_def_adjust = -three_scale * (0.05 * three_weight)

    if defender_rating is None:
        defender_rating = 5.0
    def_scale = (defender_rating - 5.5) / 4.5
    defender_adjust = -def_scale * (0.04 * points_weight)

    home_adjust = 0.02 if is_home else -0.02

    if player_team_strength is None or opp_team_strength is None:
        blowout_gap = 0.0
        blowout_adjust = 0.0
    else:
        gap = abs(float(player_team_strength) - float(opp_team_strength))
        gap_norm = min(gap / 4.0, 1.0)
        blowout_adjust = -gap_norm * 0.06
        blowout_gap = gap

    manual_adjust = manual_adjust_frac

    hist_prob = (
        base_prob
        + usage_adjust
        + home_adjust
        + reb_def_adjust
        + three_def_adjust
        + defender_adjust
        + blowout_adjust
        + manual_adjust
    )
    hist_prob = float(np.clip(hist_prob, 0.01, 0.99))

    if projection_prob is not None:
        combined = 0.6 * hist_prob + 0.4 * projection_prob
    else:
        combined = hist_prob

    combined += pace_adjust
    prob = float(np.clip(combined, 0.01, 0.99))
    fair_odds = 1.0 / prob

    usage_info = {
        "overall_usage": overall_usage,
        "recent_usage": recent_usage,
        "usage_diff_frac": usage_diff_frac,
        "usage_adjust": usage_adjust,
        "home_adjust": home_adjust,
        "manual_adjust": manual_adjust,
    }

    defense_info = {
        "opp_reb_def_rating": opp_reb_def_rating,
        "reb_def_adjust": reb_def_adjust,
        "opp_3pt_def_rating": opp_3pt_def_rating,
        "three_def_adjust": three_def_adjust,
        "defender_rating": defender_rating,
        "defender_adjust": defender_adjust,
        "points_weight": points_weight,
        "reb_weight": reb_weight,
        "three_weight": three_weight,
    }

    blowout_info = {
        "player_team_strength": player_team_strength,
        "opp_team_strength": opp_team_strength,
        "strength_gap": blowout_gap,
        "blowout_adjust": blowout_adjust,
    }

    return {
        "prob": prob,
        "fair_odds": fair_odds,
        "breakdown": breakdown,
        "usage_info": usage_info,
        "defense_info": defense_info,
        "blowout_info": blowout_info,
        "hist_prob": hist_prob,
        "projection_prob": projection_prob,
        "pace_adjust": pace_adjust,
    }


def evaluate_vs_book(prob, book_odds):
    if prob is None or book_odds is None:
        return None

    book_implied_p = 1.0 / book_odds
    edge = prob - book_implied_p
    if edge > 0.0:
        rec = "BET (positive raw edge)"
    else:
        rec = "PASS (no edge)"

    return {
        "book_odds": book_odds,
        "book_implied_p": book_implied_p,
        "edge": edge,
        "recommendation": rec,
    }


# DEFENDER ESTIMATION


def position_group_from_string(pos_str: str):
    if not pos_str:
        return None
    s = pos_str.upper()

    if "GUARD" in s:
        return "G"
    if "FORWARD" in s:
        return "F"
    if "CENTER" in s:
        return "C"

    if any(tag in s for tag in ["PG", "SG", "G"]):
        return "G"
    if any(tag in s for tag in ["SF", "PF", "F"]):
        return "F"
    if "C" in s:
        return "C"

    for ch in s:
        if ch in ("G", "F", "C"):
            return ch
    return None


def get_offensive_player_position_group(player_id: int) -> str:
    try:
        cpi = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        df_info = cpi.get_data_frames()[0]
        pos_str = df_info.iloc[0].get("POSITION", "")
        group = position_group_from_string(pos_str)
        return group or "G"
    except Exception:
        return "G"


def estimate_defender_rating(off_player_id: int, opp_team_id: int, season_int: int):
    season_str = season_int_to_str(season_int)
    pos_group = get_offensive_player_position_group(off_player_id)

    try:
        roster = commonteamroster.CommonTeamRoster(
            team_id=opp_team_id,
            season=season_str,
            league_id_nullable='00'
        )
        df_roster = roster.get_data_frames()[0]
    except Exception:
        return {
            "rating": 5.0,
            "name": "Unknown defender",
            "stats": None,
            "note": "Failed to load team roster"
        }

    candidates = []
    for _, row in df_roster.iterrows():
        pid = row.get("PLAYER_ID")
        r_pos_str = row.get("POSITION", "")
        r_group = position_group_from_string(r_pos_str)
        if r_group == pos_group:
            candidates.append((pid, row.get("PLAYER", "Unknown"), r_pos_str))

    if not candidates:
        return {
            "rating": 5.0,
            "name": "Unknown defender",
            "stats": None,
            "note": "No matching position on opponent roster"
        }

    best_def = None
    best_min = -1.0

    for pid, pname, pos_str in candidates:
        try:
            gl = playergamelog.PlayerGameLog(player_id=pid, season=season_str)
            df = gl.get_data_frames()[0]
            if df.empty:
                continue

            mins = parse_minutes_column(df)
            avg_min = float(mins.mean())

            avg_dreb = float(df["DREB"].mean()) if "DREB" in df.columns else 0.0
            avg_stl = float(df["STL"].mean()) if "STL" in df.columns else 0.0
            avg_blk = float(df["BLK"].mean()) if "BLK" in df.columns else 0.0
            avg_pf  = float(df["PF"].mean())  if "PF"  in df.columns else 0.0

            def_score = avg_stl * 1.7 + avg_blk * 1.7 + avg_dreb * 0.35 - avg_pf * 0.4
            def_score_clamped = max(0.0, min(def_score, 6.0))
            rating = 1.0 + (def_score_clamped / 6.0) * 9.0

            if avg_min > best_min:
                best_min = avg_min
                best_def = {
                    "rating": rating,
                    "name": f"{pname} ({pos_str})",
                    "stats": {
                        "avg_min": avg_min,
                        "avg_dreb": avg_dreb,
                        "avg_stl": avg_stl,
                        "avg_blk": avg_blk,
                        "avg_pf": avg_pf,
                        "def_score": def_score,
                    },
                    "note": "Highest-minutes same-position defender on opponent"
                }
        except Exception:
            continue

    if best_def is None:
        return {
            "rating": 5.0,
            "name": "Unknown defender",
            "stats": None,
            "note": "Could not compute defender stats"
        }

    return best_def



# VISUAL TAB


def update_visual_chart(df_season_raw, line_val, market_code):
    global fig, ax, canvas

    if fig is None or ax is None or canvas is None:
        return

    ax.clear()

    if df_season_raw is None or df_season_raw.empty:
        ax.text(0.5, 0.5, "No current season data", ha="center", va="center")
        ax.set_title("Margin vs Line – No season data")
        canvas.draw()
        return

    df_vis = add_market_value_column(df_season_raw, market_code)
    if "GAME_DATE" in df_vis.columns:
        df_vis = df_vis.sort_values("GAME_DATE")

    margin = df_vis["VALUE"] - float(line_val)
    hits = margin >= 0

    colors = ["green" if h else "red" for h in hits]
    x = np.arange(len(margin))

    ax.bar(x, margin, color=colors)
    ax.axhline(0, color="black", linewidth=1)

    ax.set_title(f"Margin vs Line – {market_label(market_code)} {line_val}+ (This Season)")
    ax.set_ylabel("Margin vs line (VALUE - line)")
    ax.set_xlabel("Game (time order)")

    canvas.draw()



# MULTI PAIR SCORING


def market_weight_for_multis(market_code: str) -> float:
    weights = {
        "REB": 0.75,
        "PTS": 1.00,
        "AST": 1.00,
        "PR":  1.05,
        "PA":  1.10,
        "RA":  1.00,
        "PRA": 1.00,
        "3PM": 1.10,
    }
    return weights.get(market_code, 1.0)


def same_game(leg1: dict, leg2: dict) -> bool:
    return leg1.get("game_key") is not None and leg1.get("game_key") == leg2.get("game_key")


def sgp_rho(leg1: dict, leg2: dict) -> float:
    if not same_game(leg1, leg2):
        return 0.0

    rho = 0.08

    if leg1["player_name"] == leg2["player_name"]:
        rho += 0.25

    if leg1.get("player_team_id") is not None and leg2.get("player_team_id") is not None:
        if leg1["player_team_id"] == leg2["player_team_id"] and leg1["player_name"] != leg2["player_name"]:
            rho += 0.07

    m1, m2 = leg1.get("market_code"), leg2.get("market_code")
    if (m1 == "PTS" and m2 == "AST") or (m2 == "PTS" and m1 == "AST"):
        rho += 0.05
    if (m1 == "PTS" and m2 == "3PM") or (m2 == "PTS" and m1 == "3PM"):
        rho += 0.04
    if (m1 == "REB" and m2 == "PTS") or (m2 == "REB" and m1 == "PTS"):
        rho += 0.02

    rho = float(np.clip(rho, -0.35, 0.55))
    return rho


def joint_prob_two(p1: float, p2: float, rho: float) -> float:
    p1 = float(np.clip(p1, 0.001, 0.999))
    p2 = float(np.clip(p2, 0.001, 0.999))
    rho = float(np.clip(rho, -1.0, 1.0))

    indep = p1 * p2
    lower = max(0.0, p1 + p2 - 1.0)
    upper = min(p1, p2)

    if rho >= 0:
        jp = indep + rho * (upper - indep)
    else:
        jp = indep + rho * (indep - lower)

    return float(np.clip(jp, lower, upper))


def joint_prob_three(a: dict, b: dict, c: dict) -> float:
    p1, p2, p3 = a["prob"], b["prob"], c["prob"]
    if p1 is None or p2 is None or p3 is None:
        return None

    rho12 = sgp_rho(a, b)
    p12 = joint_prob_two(p1, p2, rho12)

    rho13 = sgp_rho(a, c)
    rho23 = sgp_rho(b, c)
    rho_comp = 0.5 * (rho13 + rho23)

    p123 = joint_prob_two(p12, p3, rho_comp)
    return float(np.clip(p123, 0.001, 0.999))


def pair_score(leg1: dict, leg2: dict) -> float:
    p1, p2 = leg1["prob"], leg2["prob"]
    o1, o2 = leg1["odds"], leg2["odds"]

    if p1 is None or p2 is None or o1 is None or o2 is None:
        return -999.0

    e1 = leg1.get("edge_proxy", None)
    e2 = leg2.get("edge_proxy", None)
    if e1 is None:
        e1 = p1 - (1.0 / o1)
    if e2 is None:
        e2 = p2 - (1.0 / o2)

    base = e1 + e2

    mw = market_weight_for_multis(leg1["market_code"]) * market_weight_for_multis(leg2["market_code"])
    base *= mw

    penalty = 0.0

    if leg1["player_name"] == leg2["player_name"]:
        penalty += 0.05

    if leg1.get("opp_team_id") == leg2.get("opp_team_id"):
        penalty += 0.02

    r1 = get_rebound_weight_for_market(leg1["market_code"])
    r2 = get_rebound_weight_for_market(leg2["market_code"])
    if r1 >= 0.7 and r2 >= 0.7:
        penalty += 0.02

    if (leg1["market_code"] == "3PM") ^ (leg2["market_code"] == "3PM"):
        base += 0.005

    if same_game(leg1, leg2):
        base += 0.003

    return base - penalty


def adjusted_parlay_odds(odds_list, discount_frac: float) -> float:
    prod = 1.0
    for o in odds_list:
        prod *= float(o)

    factor = float(np.clip(discount_frac, 0.25, 2.50))
    return 1.0 + (prod - 1.0) * factor


def compute_best_multis():
    legs = [l for l in saved_legs if l.get("odds") is not None and l.get("prob") is not None]
    if len(legs) < 2:
        return [], []

    factor = parlay_factor_var.get()

    best_pairs = []
    for a, b in itertools.combinations(legs, 2):
        ps = pair_score(a, b)

        rho = sgp_rho(a, b)
        p_multi = joint_prob_two(a["prob"], b["prob"], rho)

        o_multi = adjusted_parlay_odds([a["odds"], b["odds"]], factor)
        ev = p_multi * o_multi - 1.0

        best_pairs.append({
            "pair_score": ps,
            "ev": ev,
            "p_multi": p_multi,
            "o_multi": o_multi,
            "legs": (a, b)
        })

    best_pairs.sort(key=lambda x: x["pair_score"], reverse=True)
    best_pairs = best_pairs[:20]

    best_triples = []
    if len(legs) >= 3:
        for a, b, c in itertools.combinations(legs, 3):
            ps = pair_score(a, b) + pair_score(a, c) + pair_score(b, c)

            p_multi = joint_prob_three(a, b, c)
            if p_multi is None:
                continue

            o_multi = adjusted_parlay_odds([a["odds"], b["odds"], c["odds"]], factor)
            ev = p_multi * o_multi - 1.0

            best_triples.append({
                "pair_score": ps,
                "ev": ev,
                "p_multi": p_multi,
                "o_multi": o_multi,
                "legs": (a, b, c)
            })

        best_triples.sort(key=lambda x: x["pair_score"], reverse=True)
        best_triples = best_triples[:20]

    return best_pairs, best_triples


def refresh_multi_tab():
    for row in legs_tree.get_children():
        legs_tree.delete(row)

    for i, leg in enumerate(saved_legs):
        legs_tree.insert(
            "",
            "end",
            iid=str(i),
            values=(
                leg["player_name"],
                market_label(leg["market_code"]),
                leg["line_val"],
                f"{leg['odds']:.2f}" if leg.get("odds") else "",
                f"{leg['prob']*100:.1f}%" if leg.get("prob") else "",
                f"{leg.get('edge_proxy', 0.0)*100:.2f}%" if leg.get("edge_proxy") is not None else "",
                leg.get("tag", "")
            )
        )

    for row in pairs_tree.get_children():
        pairs_tree.delete(row)
    for row in triples_tree.get_children():
        triples_tree.delete(row)

    best_pairs, best_triples = compute_best_multis()

    for idx, item in enumerate(best_pairs, start=1):
        a, b = item["legs"]
        pairs_tree.insert(
            "",
            "end",
            values=(
                idx,
                f"{a['player_name']} ({a['market_code']} {a['line_val']})",
                f"{b['player_name']} ({b['market_code']} {b['line_val']})",
                f"{item['o_multi']:.2f}",
                f"{item['p_multi']*100:.1f}%",
                f"{item['ev']*100:.2f}%",
                f"{item['pair_score']:.5f}",
            )
        )

    for idx, item in enumerate(best_triples, start=1):
        a, b, c = item["legs"]
        triples_tree.insert(
            "",
            "end",
            values=(
                idx,
                f"{a['player_name']} ({a['market_code']} {a['line_val']})",
                f"{b['player_name']} ({b['market_code']} {b['line_val']})",
                f"{c['player_name']} ({c['market_code']} {c['line_val']})",
                f"{item['o_multi']:.2f}",
                f"{item['p_multi']*100:.1f}%",
                f"{item['ev']*100:.2f}%",
                f"{item['pair_score']:.5f}",
            )
        )


def delete_selected_leg():
    sel = legs_tree.selection()
    if not sel:
        return
    idxs = sorted([int(x) for x in sel], reverse=True)
    for i in idxs:
        if 0 <= i < len(saved_legs):
            saved_legs.pop(i)
    refresh_multi_tab()



# TKINTER CALLBACKS


def search_button_pressed():
    query = player_entry.get().strip()
    if not query:
        messagebox.showerror("Error", "Enter a player name to search.")
        return

    matches = search_players(query)
    if not matches:
        messagebox.showinfo("No Results", f"No players found matching '{query}'.")
        player_select["values"] = []
        return

    values = []
    player_id_map.clear()

    for p in matches:
        name = p["full_name"]
        active_flag = "ACTIVE" if p.get("is_active") else "RETIRED"
        text = f"{name} ({active_flag})"
        values.append(text)
        player_id_map[text] = p["id"]

    player_select["values"] = values
    player_select.set(values[0])


def get_common_player_info_cached(player_id: int):
    if player_id in player_info_cache:
        return player_info_cache[player_id]
    cpi = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    info_row = cpi.get_data_frames()[0].iloc[0]
    player_info_cache[player_id] = info_row
    return info_row



def start_calculate():
    global calc_in_progress

    if calc_in_progress:
        return

    calc_in_progress = True
    if calc_btn is not None:
        calc_btn.config(state="disabled")
    if status_lbl is not None:
        status_lbl.config(text="Calculating... (NBA API can be slow)")

    t = threading.Thread(target=_calc_worker, daemon=True)
    t.start()


def _calc_worker():
    try:
    
        get_stats_pressed_core()
        run_in_ui(lambda: status_lbl.config(text="Done ✅") if status_lbl is not None else None)
    except Exception as e:
        run_in_ui(lambda: messagebox.showerror("Error", str(e)))
        run_in_ui(lambda: status_lbl.config(text="Failed ❌") if status_lbl is not None else None)
    finally:
        def _unlock():
            global calc_in_progress
            calc_in_progress = False
            if calc_btn is not None:
                calc_btn.config(state="normal")
        run_in_ui(_unlock)



def get_stats_pressed_core():
    player_sel = player_select.get()
    team_sel = team_select.get()
    market_sel = market_select.get()

    if not player_sel:
        run_in_ui(lambda: messagebox.showerror("Error", "Select a player."))
        return
    if not team_sel:
        run_in_ui(lambda: messagebox.showerror("Error", "Select an opponent team."))
        return
    if not market_sel:
        run_in_ui(lambda: messagebox.showerror("Error", "Select a market type."))
        return

    market_map = {
        "Points": "PTS",
        "Rebounds": "REB",
        "Assists": "AST",
        "Points + Rebounds + Assists (PRA)": "PRA",
        "Points + Rebounds (PR)": "PR",
        "Points + Assists (PA)": "PA",
        "Rebounds + Assists (RA)": "RA",
        "3-Pointers Made (3PM)": "3PM",
    }
    market_code = market_map.get(market_sel, "PTS")

    try:
        line_val = float(line_entry.get().strip())
    except ValueError:
        run_in_ui(lambda: messagebox.showerror("Error", "Enter a valid line (e.g. 15 or 2.5)."))
        return

    book_odds_input = odds_entry.get().strip()
    if book_odds_input:
        try:
            book_odds = float(book_odds_input)
            if book_odds <= 1.0:
                raise ValueError
        except ValueError:
            run_in_ui(lambda: messagebox.showerror("Error", "Book odds must be a decimal > 1.0 (e.g. 1.23)."))
            return
    else:
        book_odds = None

    manual_percent = manual_scale.get()
    manual_adjust_frac = manual_percent / 100.0

    player_id = player_id_map.get(player_sel)
    team_info = team_map.get(team_sel)

    if player_id is None or team_info is None:
        run_in_ui(lambda: messagebox.showerror("Error", "Could not resolve player or team."))
        return

    opp_abbr = team_info["abbreviation"]
    opp_team_id = team_info["id"]
    is_home = home_var.get() == 1

    try:

        df_all_raw = get_player_all_games_cached(player_id)
        df_all_raw = add_opponent_column(df_all_raw)
    except Exception as e:
        run_in_ui(lambda: messagebox.showerror("API Error", f"Failed to fetch game logs:\n{e}"))
        return

    if df_all_raw.empty:
        run_in_ui(lambda: messagebox.showerror("Error", "No game logs found for this player."))
        return

    current_season_int = df_all_raw["SEASON"].max()
    df_season_raw = get_current_season_games(df_all_raw)
    df_vs_opp_raw = get_vs_opp_current_and_last_season(df_all_raw, opp_abbr)

    player_team_id = None
    player_team_abbr = None
    try:

        info_row = get_common_player_info_cached(player_id)
        if "TEAM_ID" in info_row:
            player_team_id = int(info_row["TEAM_ID"])
        if "TEAM_ABBREVIATION" in info_row:
            player_team_abbr = info_row["TEAM_ABBREVIATION"]
    except Exception:
        player_team_id = None
        player_team_abbr = None

    if player_team_id is not None:
        player_team_strength = get_team_strength_from_standings(player_team_id, current_season_int)
    else:
        player_team_strength = None

    opp_team_strength = get_team_strength_from_standings(opp_team_id, current_season_int)

    opp_reb_def_rating = get_team_reb_rating(opp_team_id, current_season_int)
    opp_3pt_def_rating = get_team_3pt_def_rating(opp_team_id, current_season_int)

    pace_adjust = 0.0
    pace_info = {
        "player_team_pace": None,
        "opp_team_pace": None,
        "league_pace_avg": None,
        "matchup_pace": None,
    }

    player_team_pace = None
    league_avg_pace = None
    if player_team_id is not None:
        player_team_pace, league_avg_pace = get_team_pace(player_team_id, current_season_int)
        pace_info["player_team_pace"] = player_team_pace
        pace_info["league_pace_avg"] = league_avg_pace

    opp_team_pace, _tmp = get_team_pace(opp_team_id, current_season_int)
    pace_info["opp_team_pace"] = opp_team_pace

    if league_avg_pace is None:
        _df, league_avg_pace2 = get_pace_data(current_season_int)
        if league_avg_pace is None:
            league_avg_pace = league_avg_pace2
            pace_info["league_pace_avg"] = league_avg_pace

    if league_avg_pace is not None and league_avg_pace > 0:
        if player_team_pace is not None and opp_team_pace is not None:
            matchup_pace = (player_team_pace + opp_team_pace) / 2.0
        elif opp_team_pace is not None:
            matchup_pace = opp_team_pace
        elif player_team_pace is not None:
            matchup_pace = player_team_pace
        else:
            matchup_pace = None

        pace_info["matchup_pace"] = matchup_pace

        if matchup_pace is not None:
            pace_factor = (matchup_pace - league_avg_pace) / league_avg_pace
            pace_adjust = float(np.clip(pace_factor * 0.08, -0.05, 0.05))
    else:
        pace_adjust = 0.0

    defender_info = estimate_defender_rating(
        off_player_id=player_id,
        opp_team_id=opp_team_id,
        season_int=current_season_int
    )
    defender_rating = defender_info["rating"]

    df_all = add_market_value_column(df_all_raw, market_code)
    df_season = add_market_value_column(df_season_raw, market_code)
    df_vs_opp = add_market_value_column(df_vs_opp_raw, market_code)

    df_for_proj = df_season if not df_season.empty else df_all
    projection_prob, proj_info = compute_projection_probability(df_for_proj, line_val)

    prob_result = calculate_probability(
        df_all=df_all,
        df_season=df_season,
        df_vs_opp=df_vs_opp,
        is_home=is_home,
        line_val=line_val,
        manual_adjust_frac=manual_adjust_frac,
        defender_rating=defender_rating,
        opp_reb_def_rating=opp_reb_def_rating,
        opp_3pt_def_rating=opp_3pt_def_rating,
        player_team_strength=player_team_strength,
        opp_team_strength=opp_team_strength,
        market_code=market_code,
        projection_prob=projection_prob,
        pace_adjust=pace_adjust,
    )

    prob = prob_result["prob"]
    fair_odds = prob_result["fair_odds"]
    breakdown = prob_result["breakdown"]
    usage_info = prob_result["usage_info"]
    defense_info = prob_result["defense_info"]
    blowout_info = prob_result["blowout_info"]
    hist_prob = prob_result["hist_prob"]
    projection_prob_final = prob_result["projection_prob"]
    pace_adjust_final = prob_result["pace_adjust"]

    if prob is None:
        run_in_ui(lambda: messagebox.showerror("Error", "Not enough data to estimate probability."))
        return

    book_eval = evaluate_vs_book(prob, book_odds) if book_odds is not None else None

    hits_season, total_season = hit_count(df_season, line_val)
    hits_last10, total_last10 = hit_count(df_all.tail(10), line_val)
    hits_last3, total_last3 = hit_count(df_all.tail(3), line_val)
    hits_vs_opp, total_vs_opp = hit_count(df_vs_opp, line_val)


    run_in_ui(lambda: update_visual_chart(df_season_raw, line_val, market_code))

 
    # SAVE LEG INTO MULTI BUILDER
  
    if add_to_multi_var.get() == 1:
        try:
            odds_for_leg = float(odds_entry.get().strip())
        except Exception:
            odds_for_leg = None

        if odds_for_leg is None:
            run_in_ui(lambda: messagebox.showinfo(
                "Multi Builder",
                "You ticked 'Save this leg' but no odds were entered.\n"
                "Enter odds next time so the Multi Builder can score it."
            ))
        else:
            title_player = player_sel.split("(")[0].strip()
            tag = tag_entry.get().strip()

            edge_proxy = None
            if book_eval is not None:
                edge_proxy = book_eval["edge"]
            else:
                edge_proxy = prob - (1.0 / odds_for_leg)

            game_key = None
            if player_team_id is not None and opp_team_id is not None:
                game_key = (int(player_team_id), int(opp_team_id), bool(is_home))

            saved_legs.append({
                "player_name": title_player,
                "market_code": market_code,
                "line_val": line_val,
                "odds": odds_for_leg,
                "prob": prob,
                "edge_proxy": edge_proxy,
                "opp_team_id": opp_team_id,
                "player_team_id": player_team_id,
                "is_home": is_home,
                "game_key": game_key,
                "tag": tag,
            })
            run_in_ui(refresh_multi_tab)


    # POPUP DETAIL WINDOW (UI thread)

    def _open_popup():
        top = tk.Toplevel(root)
        title_player = player_sel.split("(")[0].strip()
        label_market = market_label(market_code)
        title_suffix = f" | {label_market} line: {line_val}+"
        top.title(f"{title_player} vs {team_info['full_name']}{title_suffix}")
        top.geometry("900x820")

        top_frame = tk.Frame(top, padx=10, pady=10)
        top_frame.pack(fill=tk.X)

        rec_frame = tk.Frame(top, padx=10, pady=5)
        rec_frame.pack(fill=tk.X)

        text_frame = tk.Frame(top, padx=10, pady=5)
        text_frame.pack(fill=tk.BOTH, expand=True)

        header_label = tk.Label(
            top_frame,
            text=f"{title_player} vs {team_info['full_name']} ({opp_abbr})  |  {label_market} {line_val}+",
            font=("Segoe UI", 12, "bold")
        )
        header_label.pack(anchor="w")

        sub_label = tk.Label(
            top_frame,
            text=f"Player team: {player_team_abbr}   |   Location: {'HOME' if is_home else 'AWAY'}",
            font=("Segoe UI", 10)
        )
        sub_label.pack(anchor="w", pady=(4, 0))

        if book_eval is not None:
            rec_text = book_eval["recommendation"]
            if "BET" in rec_text:
                bg_color = "#1c7c2f"
            else:
                bg_color = "#b22222"

            rec_label = tk.Label(
                rec_frame,
                text=f"{rec_text}  |  Edge: {book_eval['edge']*100:.1f} percentage points",
                font=("Segoe UI", 13, "bold"),
                fg="white",
                bg=bg_color,
                padx=12,
                pady=6
            )
        else:
            rec_label = tk.Label(
                rec_frame,
                text="No odds entered – no BET/PASS recommendation",
                font=("Segoe UI", 12, "bold"),
                fg="white",
                bg="#555555",
                padx=12,
                pady=6
            )
        rec_label.pack(fill=tk.X)

        text = tk.Text(
            text_frame,
            width=120,
            height=40,
            font=("Consolas", 9)
        )
        text.pack(fill=tk.BOTH, expand=True)

        text.insert(tk.END, "=== PROBABILITY SUMMARY ===\n")
        if hist_prob is not None:
            text.insert(tk.END, f"History-based probability: {hist_prob*100:.1f}%\n")
        if projection_prob_final is not None:
            text.insert(tk.END, f"Projection-based probability (minutes + per-min): {projection_prob_final*100:.1f}%\n")
        text.insert(tk.END, f"Pace adjustment: {pace_adjust_final*100:.1f} percentage points\n")
        text.insert(tk.END, f"FINAL combined probability: {prob*100:.1f}%\n")
        text.insert(tk.END, f"Model fair odds: {fair_odds:.3f}\n\n")

        text.insert(tk.END, "Breakdown of weighted hit rates (normalized weights):\n")
        for label, r, w_norm in breakdown:
            text.insert(tk.END, f"  - {label}: hit rate = {r*100:.1f}% | weight = {w_norm*100:.1f}%\n")
        text.insert(tk.END, "\n")

        text.insert(tk.END, "=== MINUTES / PROJECTION DETAILS ===\n")
        if proj_info["sample_size"] > 0:
            text.insert(tk.END, f"Sample size for projection: {proj_info['sample_size']} games\n")
            if proj_info["projected_minutes"] is not None:
                text.insert(tk.END, f"Projected minutes (avg recent): {proj_info['projected_minutes']:.1f}\n")
            if proj_info["per_min_rate"] is not None:
                text.insert(tk.END, f"Per-minute VALUE rate: {proj_info['per_min_rate']:.3f}\n")
            if proj_info["projected_mean"] is not None:
                text.insert(tk.END, f"Projected VALUE at these minutes: {proj_info['projected_mean']:.2f}\n")
            if projection_prob_final is not None:
                text.insert(tk.END, f"Projection-based P(VALUE >= line): {projection_prob_final*100:.1f}%\n")
        else:
            text.insert(tk.END, "Not enough data for projection.\n")
        text.insert(tk.END, "\n")

        text.insert(tk.END, "=== RAW HIT COUNTS (sanity check) ===\n")
        if total_season > 0:
            text.insert(tk.END, f"  This season: {hits_season}/{total_season}  ({hits_season/total_season*100:.1f}%)\n")
        else:
            text.insert(tk.END, "  This season: no data\n")
        if total_last10 > 0:
            text.insert(tk.END, f"  Last 10 games: {hits_last10}/{total_last10}  ({hits_last10/total_last10*100:.1f}%)\n")
        else:
            text.insert(tk.END, "  Last 10 games: no data\n")
        if total_last3 > 0:
            text.insert(tk.END, f"  Last 3 games: {hits_last3}/{total_last3}  ({hits_last3/total_last3*100:.1f}%)\n")
        else:
            text.insert(tk.END, "  Last 3 games: no data\n")
        if total_vs_opp > 0:
            text.insert(tk.END, f"  Vs opponent (current + last season): {hits_vs_opp}/{total_vs_opp}  ({hits_vs_opp/total_vs_opp*100:.1f}%)\n")
        else:
            text.insert(tk.END, "  Vs opponent (current + last season): no data\n")
        text.insert(tk.END, "\n")

        if book_eval is not None:
            text.insert(tk.END, "=== VS BOOK ODDS ===\n")
            text.insert(tk.END, f"Book odds: {book_eval['book_odds']:.3f}\n")
            text.insert(tk.END, f"Book implied probability: {book_eval['book_implied_p']*100:.1f}%\n")
            text.insert(tk.END, f"Your edge: {(book_eval['edge']*100):.1f} percentage points\n")
            text.insert(tk.END, f"Recommendation: {book_eval['recommendation']}\n\n")
        else:
            text.insert(tk.END, "No book odds entered, so no EV comparison.\n\n")

    run_in_ui(_open_popup)



# BUILD GUI (WITH TABS)


root = tk.Tk()
root.title("NBA Player Prop Helper")
root.geometry("1040x620")

notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

main_frame = tk.Frame(notebook, padx=10, pady=10)
notebook.add(main_frame, text="Main")

visual_frame = tk.Frame(notebook, padx=10, pady=10)
notebook.add(visual_frame, text="Visual")

multi_frame = tk.Frame(notebook, padx=10, pady=10)
notebook.add(multi_frame, text="Multi Builder")

tk.Label(main_frame, text="Search Player Name:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
player_entry = tk.Entry(main_frame, width=30)
player_entry.grid(row=0, column=1, padx=5)
tk.Button(main_frame, text="Search", command=search_button_pressed).grid(row=0, column=2, padx=5)

tk.Label(main_frame, text="Select Player:", font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=(8,0))
player_select = ttk.Combobox(main_frame, width=45)
player_select.grid(row=1, column=1, columnspan=2, sticky="w", pady=(8,0))

tk.Label(main_frame, text="Select Opponent Team:", font=("Segoe UI", 10)).grid(row=2, column=0, sticky="w", pady=(8,0))
team_select = ttk.Combobox(main_frame, width=45)
team_select.grid(row=2, column=1, columnspan=2, sticky="w", pady=(8,0))

team_values = []
for t in sorted(all_teams, key=lambda x: x["full_name"]):
    text_val = f"{t['full_name']} ({t['abbreviation']})"
    team_values.append(text_val)
    team_map[text_val] = t

team_select["values"] = team_values
if team_values:
    team_select.set(team_values[0])

tk.Label(main_frame, text="Market Type:", font=("Segoe UI", 10)).grid(row=3, column=0, sticky="w", pady=(8,0))
market_select = ttk.Combobox(main_frame, width=45)
market_select["values"] = [
    "Points",
    "Rebounds",
    "Assists",
    "Points + Rebounds + Assists (PRA)",
    "Points + Rebounds (PR)",
    "Points + Assists (PA)",
    "Rebounds + Assists (RA)",
    "3-Pointers Made (3PM)",
]
market_select.set("Points")
market_select.grid(row=3, column=1, columnspan=2, sticky="w", pady=(8,0))

tk.Label(main_frame, text="Line (e.g. 15 or 2.5):", font=("Segoe UI", 10)).grid(row=4, column=0, sticky="w", pady=(8,0))
line_entry = tk.Entry(main_frame, width=10)
line_entry.insert(0, "15")
line_entry.grid(row=4, column=1, sticky="w", pady=(8,0))

tk.Label(main_frame, text="Book odds (required to save legs):", font=("Segoe UI", 10)).grid(row=5, column=0, sticky="w", pady=(8,0))
odds_entry = tk.Entry(main_frame, width=10)
odds_entry.grid(row=5, column=1, sticky="w", pady=(8,0))

home_var = tk.IntVar(value=1)
home_check = tk.Checkbutton(main_frame, text="Game at HOME", variable=home_var)
home_check.grid(row=6, column=0, sticky="w", pady=(8,0))

tk.Label(main_frame, text="Manual adjust (gut feel): -10% to +10%", font=("Segoe UI", 10)).grid(row=7, column=0, sticky="w", pady=(8,0))
manual_scale = tk.Scale(main_frame, from_=-10, to=10, orient=tk.HORIZONTAL, length=260)
manual_scale.set(0)
manual_scale.grid(row=7, column=1, columnspan=2, sticky="w", pady=(8,0))

add_to_multi_var = tk.IntVar(value=0)
tk.Checkbutton(main_frame, text="Save this leg to Multi Builder", variable=add_to_multi_var).grid(row=8, column=0, sticky="w", pady=(10,0))

tk.Label(main_frame, text="Tag (optional):", font=("Segoe UI", 10)).grid(row=8, column=1, sticky="e", pady=(10,0))
tag_entry = tk.Entry(main_frame, width=18)
tag_entry.grid(row=8, column=2, sticky="w", pady=(10,0))


calc_btn = tk.Button(main_frame, text="Calculate Probability", command=start_calculate)
calc_btn.grid(row=9, column=0, columnspan=3, pady=(15,0))


status_lbl = tk.Label(main_frame, text="", font=("Segoe UI", 9))
status_lbl.grid(row=10, column=0, columnspan=3, sticky="w", pady=(6,0))

fig = Figure(figsize=(7, 3.8), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("Margin vs Line – chart will update after calculation")
ax.set_xlabel("Game")
ax.set_ylabel("Margin vs line")

canvas = FigureCanvasTkAgg(fig, master=visual_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)

top_controls = tk.Frame(multi_frame)
top_controls.pack(fill=tk.X)

tk.Label(top_controls, text="Parlay factor (tune to match Dabble multi pricing):").pack(side=tk.LEFT)
parlay_factor_var = tk.DoubleVar(value=1.00)
parlay_scale = tk.Scale(top_controls, from_=0.50, to=2.00, resolution=0.02,
                        orient=tk.HORIZONTAL, length=260, variable=parlay_factor_var,
                        command=lambda _=None: refresh_multi_tab())
parlay_scale.pack(side=tk.LEFT, padx=8)

tk.Button(top_controls, text="Delete selected leg(s)", command=delete_selected_leg).pack(side=tk.RIGHT)

tk.Label(multi_frame, text="Saved legs:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 3))
legs_tree = ttk.Treeview(multi_frame, columns=("player","market","line","odds","prob","edge","tag"), show="headings", height=7)
for c, w in [("player",180),("market",200),("line",60),("odds",60),("prob",70),("edge",70),("tag",120)]:
    legs_tree.heading(c, text=c.upper())
    legs_tree.column(c, width=w, anchor="w")
legs_tree.pack(fill=tk.X)

tk.Label(multi_frame, text="Best 2-leg multis (by pair score):", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 3))
pairs_tree = ttk.Treeview(multi_frame, columns=("rank","leg1","leg2","odds","p","ev","score"), show="headings", height=8)
for c, w in [("rank",50),("leg1",320),("leg2",320),("odds",70),("p",70),("ev",70),("score",80)]:
    pairs_tree.heading(c, text=c.upper())
    pairs_tree.column(c, width=w, anchor="w")
pairs_tree.pack(fill=tk.X)

tk.Label(multi_frame, text="Best 3-leg multis (by pair score sum):", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 3))
triples_tree = ttk.Treeview(multi_frame, columns=("rank","leg1","leg2","leg3","odds","p","ev","score"), show="headings", height=8)
for c, w in [("rank",50),("leg1",240),("leg2",240),("leg3",240),("odds",70),("p",70),("ev",70),("score",80)]:
    triples_tree.heading(c, text=c.upper())
    triples_tree.column(c, width=w, anchor="w")
triples_tree.pack(fill=tk.X)

root.mainloop()
