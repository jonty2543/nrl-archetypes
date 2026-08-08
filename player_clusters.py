import pandas as pd
import numpy as np
import functions as f
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from supabase import create_client, Client
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# --- Configuration ---

supabase: Client = create_client(f.SUPABASE_URL, f.SUPABASE_KEY)

YEARS_TO_PROCESS = [2023, 2024, 2025, 2026]
ARCHETYPE_TABLE = "player_archetypes"
UPSERT_BATCH_SIZE = 500
PLAYER_NAME_ALIASES = {
    "Nicholas Hynes": "Nicho Hynes",
}
EDGE_STRIKE_MIN_ATTACKING_THREAT = 1.0
EDGE_SUPPORT_MAX_DEFENSIVE_WORKRATE = -0.75
MIDDLE_IMPACT_MIN_BALL_RUNNING = 1.0

class PositionConfig:
    def __init__(self, name, features1, features2, features3, pc_names, n_clusters, labels, descriptions, min_games=5, profiles=None, assign_to_profiles=False):
        self.name = name
        self.features1 = features1
        self.features2 = features2
        self.features3 = features3
        self.pc_names = pc_names # [pc1_name, pc2_name, pc3_name]
        self.n_clusters = n_clusters
        self.labels = labels
        self.descriptions = descriptions
        self.min_games = min_games
        self.profiles = profiles or {}
        self.assign_to_profiles = assign_to_profiles

# Define Profiles (Ideal Centroids)
PROFILES_FULLBACK = {
    'Playmaker Fullback': [1.5, 0.0, -0.5],
    'Ball Running Fullback': [-0.5, 1.5, 0.0],
    'Workhorse Fullback': [-0.5, 0.0, 1.5],
    'Balanced Fullback': [0.5, 0.5, 0.5],
    'Support Fullback': [-1.0, -1.0, -1.0]
}

PROFILES_WINGER = {
    'Finisher Winger': [0.0, 1.5, -0.5],
    'Workhorse Winger': [1.0, 0.0, 1.5],
    'Support Winger': [-1.0, -1.0, -1.0]
}

PROFILES_CENTRE = {
    'Link Centre': [1.5, 0.0, -0.75],
    'Workhorse Centre': [0.0, 0.0, 1.5],
    'Support Centre': [-1.0, -1.0, -1.0],
    'Strike Centre': [0.0, 1.5, 0.0],
}

PROFILES_HALF = {
    'Dominant Half': [0.0, 1.0, 1.5],
    'Running Half': [1.5, 0.0, 0.0],
    'Organising Half': [-0.5, 0.0, 1.0]
}

PROFILES_HOOKER = {
    'Balanced Hooker': [0.0, 0.0, 0.0],
    'Running Hooker': [1.5, 0.0, -1.0], 
    'Link Hooker': [-0.5, 0.0, 1.5],
    'Crafty Hooker': [0.0, 1.5, 0.0]
}

PROFILES_EDGE = {
    'Strike Attacking Edge': [0.0, 1.5, -0.5],
    'Strong Attacking Edge': [1.5, 0.0, -0.5],
    'Defensive Enforcer Edge': [-0.5, -0.5, 1.5],
    'Support Edge': [-1.0, -1.0, -1.0]
}

PROFILES_MIDDLE = {
    'Ball Playing Middle': [1.5, 0.0, 0.0],
    'Impact Middle': [0.0, 1.5, 0.0],
    'Standard Middle': [-0.5, -0.5, 1.0]
}

def build_middle_label_map(cluster_centroids):
    label_map = {i: 'Standard Middle' for i in range(len(cluster_centroids))}
    ball_playing_cluster = int(np.argmax(cluster_centroids[:, 0]))
    impact_candidates = [i for i in range(len(cluster_centroids)) if i != ball_playing_cluster]
    impact_clusters = sorted(
        impact_candidates,
        key=lambda i: cluster_centroids[i, 1],
        reverse=True,
    )[:2]

    label_map[ball_playing_cluster] = 'Ball Playing Middle'
    for cluster_id in impact_clusters:
        label_map[cluster_id] = 'Impact Middle'
    return label_map


def apply_archetype_assignment_rules(df, config):
    if config.name == '2nd Row':
        low_threat_strike = (
            (df['cluster_name'] == 'Strike Attacking Edge')
            & (df['pc2'] < EDGE_STRIKE_MIN_ATTACKING_THREAT)
            & (df['pc3'] < EDGE_SUPPORT_MAX_DEFENSIVE_WORKRATE)
        )
        df.loc[low_threat_strike, 'cluster_name'] = 'Support Edge'

    if config.name == 'Middle':
        low_running_impact = (
            (df['cluster_name'] == 'Impact Middle')
            & (df['pc2'] < MIDDLE_IMPACT_MIN_BALL_RUNNING)
        )
        df.loc[low_running_impact, 'cluster_name'] = 'Standard Middle'

    return df


POSITION_CONFIGS = [
    PositionConfig(
        name='Fullback',
        features1=['line_break_assists_per_80', 'try_assists_per_80', 'passes_per_80'],
        features2=['line_breaks_per_80', 'tries_per_80', 'tackle_breaks_per_80'],
        features3=['all_run_metres_per_80', 'post_contact_metres_per_80', 'all_runs_per_80'],
        pc_names=['Playmaking', 'Evasiveness', 'Workrate'],
        n_clusters=5,
        labels=['Ball Running Fullback', 'Balanced Fullback', 'Workhorse Fullback', 'Playmaker Fullback', 'Support Fullback'],
        descriptions=[
            "Fullbacks who are quick and able to break the defensive line, and opt for game breaking runs over tough carries.",
            "These well rounded fullbacks balance workrate, playmaking and elusiveness making them the complete package.",
            "High-effort players who are always around the ball. They rack up high run metres and support plays.",
            "These playmakers save their energy for the big moments, with reduced workrates but high involvement in tries and try assists.",
            "Players who are less involved in attack, but may specialise in defense or defusing kicks."
        ],
        min_games=5,
        profiles=PROFILES_FULLBACK,
        assign_to_profiles=True,
    ),
    PositionConfig(
        name='Winger',
        features1=['tackle_breaks_per_80', 'offloads_per_80', 'post_contact_metres_per_80'],
        features2=['tries_per_80','line_breaks_per_80'],
        features3=['all_run_metres_per_80', 'all_runs_per_80'],
        pc_names=['Strength In Contact', 'Try Scoring', 'Workrate'],
        n_clusters=3,
        labels=['Support Winger', 'Finisher Winger', 'Workhorse Winger'],
        descriptions=[
            "These wingers tend to be less involved in the game, perhaps due to lack of skill or opportunity.",
            "Wingers who are specialist try scorers, often with great positional awareness and speed.",
            "High involvement wingers who are strong in contact, often taking carries out of their own end."
        ],
        min_games=5,
        profiles=PROFILES_WINGER,
        assign_to_profiles=True,
    ),
    PositionConfig(
        name='Centre',
        features1=['passes_per_80', 'pass_run_ratio', 'line_break_assists_per_80', 'try_assists_per_80'],
        features2=['tries_per_80','line_breaks_per_80'],
        features3=['all_run_metres_per_80', 'tackle_breaks_per_80', 'all_runs_per_80'],
        pc_names=['Passing', 'Try Scoring', 'Workrate'],
        n_clusters=4,
        labels=['Link Centre', 'Workhorse Centre', 'Support Centre', 'Strike Centre'],
        descriptions=[
            "These centres play more of a Five-Eighth role with a high pass to run ratio, often looking to set up their winger.",
            "Attacking weapons who are heavily involved in gaining metres aswell as breaking the line and scoring tries.",
            "These players are less involved with ball in hand and may play other roles for the team.",
            "Centres who are heavily involved in try scoring, and may look to set up those around them rather than taking tough carries."
        ],
        min_games=5,
        profiles=PROFILES_CENTRE,
        assign_to_profiles=True,
    ),
    PositionConfig(
        name='Half',
        features1=['tries_per_80', 'all_run_metres_per_80', 'line_breaks_per_80', 'tackle_breaks_per_80'],
        features2=['line_break_assists_per_80', 'try_assists_per_80', 'forced_drop_outs_per_80', 'forty_twenty_per_80'],
        features3=['kicks_per_80', 'kicking_metres_per_80', 'one_point_field_goals_per_80'],
        pc_names=['Running', 'Creativity', 'Kicking'],
        n_clusters=3,
        labels=['Dominant Half', 'Running Half', 'Organising Half'],
        descriptions=[
            "These players control the attack, and are usually relied upon to set up tries and do most of the kicking.",
            "Halves with strong running games who look to break the line, usually Five-Eighths.",
            "Less dominant halves who may rely on their halves partner to control the attack, focusing on organising their edge."
        ],
        min_games=5,
        profiles=PROFILES_HALF
    ),
    PositionConfig(
        name='Hooker',
        features1=['all_run_metres', 'tackle_breaks', 'line_breaks'],
        features2=['try_assists', 'line_break_assists', 'forty_twenty', 'forced_drop_outs'],
        features3=['passes_to_run_ratio'],
        pc_names=['Ball Running', 'Creativity', 'Pass - Run Ratio'],
        n_clusters=4,
        labels=['Balanced Hooker', 'Running Hooker', 'Link Hooker', 'Crafty Hooker'],
        descriptions=[
            "Hookers who balance dummy half runs and creativity.",
            "Strong ball running hookers who often look to run from dummy half.",
            "Hookers that look to pass rather than run, usually having strong ball playing.",
            "Creative types who specialise in finding the right pass for their forwards."
        ],
        min_games=5,
        profiles=PROFILES_HOOKER,
        assign_to_profiles=True,
    ),
    PositionConfig(
        name='2nd Row', # Mapped to 'Edge' in output
        features1=['all_run_metres', 'tackle_breaks', 'offloads', 'hit_ups'],
        features2=['line_breaks', 'tries'],
        features3=['tackles_made', 'tackle_efficiency'],
        pc_names=['Attacking Workrate', 'Attacking Threat', 'Defensive Workrate'],
        n_clusters=4,
        labels=['Defensive Enforcer Edge', 'Support Edge', 'Strong Attacking Edge', 'Strike Attacking Edge'],
        descriptions=[
            "Defensive specialists who are key in protecting their edge. Less involved in attacking situations.",
            "These edges are less involved in attack and defense, and may specialise in other areas.",
            "These players are strong in contact and are relied upon to make metres for their team, often involved in tries as a result.",
            "Great line runners, often breaking the line and scoring tries, playing like a centre in attack."
        ],
        min_games=5,
        profiles=PROFILES_EDGE
    ),
    PositionConfig(
        name='Middle',
        features1=['passes_to_run_ratio', 'passes', 'line_break_assists', 'try_assists'],
        features2=['all_run_metres', 'tackle_breaks', 'post_contact_metres', 'offloads'],
        features3=['tackles_made', 'tackle_efficiency'],
        pc_names=['Ball Playing', 'Ball Running', 'Defense'],
        n_clusters=5,
        labels=['Ball Playing Middle', 'Impact Middle', 'Standard Middle'],
        descriptions=[
            "These middles often play in the lock position with strong ball playing skills, directing players in the middle of the park.",
            "The most effective ball runners, these middles are characterised by strong carries, tackle breaks and post-contact metres.",
            "Making up the rest of the middle, these players share the hit up and tackling duties."
        ],
        min_games=5,
        profiles=PROFILES_MIDDLE
    )
]

# --- Data Loading & Preprocessing ---

BASE_PLAYER_COLUMNS = {
    'player',
    'team',
    'match_date',
    'mins_played',
    'number',
    'position',
    'all_runs',
    'passes_to_run_ratio',
    'tackle_efficiency',
}

RATIO_FEATURES = {'pass_run_ratio', 'passes_to_run_ratio', 'tackle_efficiency'}


def _base_stat_name(feature):
    if feature.endswith('_per_80'):
        return feature[:-7]
    if feature.endswith('_team_share'):
        return feature[:-11]
    if feature == 'pass_run_ratio':
        return 'passes_to_run_ratio'
    return feature


def _team_share_feature_name(feature):
    if feature in RATIO_FEATURES:
        return feature
    return f"{_base_stat_name(feature)}_team_share"


def build_team_share_configs(configs):
    share_configs = []
    for config in configs:
        share_configs.append(PositionConfig(
            name=config.name,
            features1=[_team_share_feature_name(feature) for feature in config.features1],
            features2=[_team_share_feature_name(feature) for feature in config.features2],
            features3=[_team_share_feature_name(feature) for feature in config.features3],
            pc_names=config.pc_names,
            n_clusters=config.n_clusters,
            labels=config.labels,
            descriptions=config.descriptions,
            min_games=config.min_games,
            profiles=config.profiles,
            assign_to_profiles=config.assign_to_profiles,
        ))
    return share_configs


def _player_stat_columns(configs):
    columns = set(BASE_PLAYER_COLUMNS)
    for config in configs:
        for feature in config.features1 + config.features2 + config.features3:
            columns.add(_base_stat_name(feature))
    return sorted(columns)


def fetch_player_stats_for_years(years, configs, batch=500):
    columns = _player_stat_columns(configs)
    cache_dir = os.getenv("ARCHETYPE_PLAYER_STATS_CACHE_DIR")
    if cache_dir:
        frames = []
        for cache_file in sorted(Path(cache_dir).glob("player_stats_*.json")):
            with cache_file.open() as f:
                data = json.load(f)
            if data:
                frames.append(pd.DataFrame(data))
        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)

    select_cols = ','.join(columns)
    frames = []

    for year in sorted(set(years)):
        start_date = f"{year}-01-01"
        end_date = f"{year + 1}-01-01"
        offset = 0

        while True:
            response = (
                supabase
                .schema("nrl")
                .table("player_stats")
                .select(select_cols)
                .gte("match_date", start_date)
                .lt("match_date", end_date)
                .gte("mins_played", 40)
                .order("match_date")
                .range(offset, offset + batch - 1)
                .execute()
            )
            data = response.data or []
            if not data:
                break

            frames.append(pd.DataFrame(data))
            offset += batch

    if not frames:
        return pd.DataFrame(columns=columns)

    return pd.concat(frames, ignore_index=True)


def load_and_process_data(configs, player_data, stat_mode='production', recent_games=None):
    print(f"Processing {stat_mode.replace('_', ' ')} player stats...")
    
    # Filter for relevant years
    start_date = f"{min(YEARS_TO_PROCESS)}-01-01"
    end_date = f"{max(YEARS_TO_PROCESS)+1}-01-01"
    
    player_df = player_data[
        (player_data['match_date'] >= start_date) & 
        (player_data['match_date'] < end_date) & 
        (player_data['mins_played'] >= 40)
    ].copy()
    
    # Extract year
    player_df['year'] = pd.to_datetime(player_df['match_date']).dt.year
    player_df['player'] = player_df['player'].replace(PLAYER_NAME_ALIASES)
    player_df = player_df.drop_duplicates().copy()

    # Prefer the recorded position; retain number as a fallback for legacy rows.
    player_df['position'] = player_df.apply(
        lambda row: f.map_position(row.get('position'), row.get('number')),
        axis=1,
    )

    if stat_mode == 'team_share':
        required_stats = sorted({
            _base_stat_name(feature)
            for config in POSITION_CONFIGS
            for feature in config.features1 + config.features2 + config.features3
            if _team_share_feature_name(feature) != feature
        })
        group_keys = ['match_date', 'team']
        team_totals = player_df.groupby(group_keys, dropna=False)[required_stats].transform('sum')
        for stat in required_stats:
            denominator = team_totals[stat].replace(0, np.nan)
            player_df[f'{stat}_team_share'] = (player_df[stat] / denominator * 100).fillna(0)

    if recent_games:
        player_df = (
            player_df
            .sort_values('match_date')
            .groupby(['player', 'year', 'position'], group_keys=False)
            .tail(recent_games)
            .copy()
        )
    
    # Calculate per_80 stats
    num_cols = player_df.select_dtypes('number').columns
    for col in num_cols:
        if col not in ['year', 'mins_played']: # Avoid overwriting or dividing year
            player_df[f'{col}_per_80'] = player_df[col] * (80 / player_df['mins_played'])
            
    per_80_cols = [c for c in player_df.columns if c.endswith('_per_80')]
    
    # Other features
    player_df['metres_per_run'] = player_df['all_run_metres'] / player_df['all_runs']
    
    # Aggregate per player, year and position so each archetype only reflects
    # games actually played at that mapped position.
    # We need to support both 'per_80' (mean of per_80) and 'raw' (mean of per match)
    # The original code used 'player_agg' (per_80) and 'player_agg_unadjusted' (raw + per_80)
    # We will create one super-aggregated dataframe with ALL columns
    
    agg_dict = {
        'games': ('match_date', 'nunique'),
        'total_minutes': ('mins_played', 'sum'),
        'pass_run_ratio': ('passes_to_run_ratio', 'mean'),
        'tackle_efficiency': ('tackle_efficiency', 'mean'),
    }
    
    # Add all numeric columns (mean)
    for col in num_cols:
        if col not in ['year']:
             agg_dict[col] = (col, 'mean')
             
    # Add all per_80 columns (mean)
    for col in per_80_cols:
        agg_dict[col] = (col, 'mean')
        
    training_agg = (
        player_df
        .groupby(['player', 'year', 'position'], as_index=False)
        .agg(**agg_dict)
    )
    
    return training_agg

# --- Model Training ---

def train_models(training_agg, configs, game_window=None):
    models = {}
    
    print("\n====== TRAINING MODELS (GLOBAL) ======")
    
    for config in configs:
        print(f"\nTraining {config.name}...")
        
        # Filter data
        df = training_agg[training_agg['position'] == config.name].copy()
        minimum_games = min(config.min_games, game_window) if game_window else config.min_games
        df = df[df['games'] >= minimum_games]
        
        # Special exclusion for 2nd Row
        if config.name == '2nd Row':
            df = df[df['player'] != 'Chris Randall']
            
        if df.empty:
            print(f"  No data for {config.name}")
            continue
            
        # Combine features
        all_features = list(set(config.features1 + config.features2 + config.features3))
        X = df[all_features].fillna(0)
        
        # 1. Global Scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 2. Global KMeans
        kmeans = KMeans(n_clusters=config.n_clusters, random_state=42)
        kmeans.fit(X_scaled)
        
        # 3. Global PCAs
        pcas = {}
        for i, features in enumerate([config.features1, config.features2, config.features3]):
            X_sub = df[features].fillna(0)
            scaler_sub = StandardScaler()
            X_sub_scaled = scaler_sub.fit_transform(X_sub)
            
            pca = PCA(n_components=1)
            pca.fit(X_sub_scaled)
            
            pcas[f'pc{i+1}'] = {
                'model': pca,
                'scaler': scaler_sub,
                'features': features
            }
            
        # 4. Calculate the training rows in the visible PC space.
        for pc_key, pc_data in pcas.items():
            X_sub = df[pc_data['features']].fillna(0)
            X_sub_scaled = pc_data['scaler'].transform(X_sub)
            df[pc_key] = pc_data['model'].transform(X_sub_scaled)

        profile_scaler = None
        profile_matrix = None
        profile_labels = None
        if config.assign_to_profiles:
            active_profiles = {k: v for k, v in config.profiles.items() if k in config.labels}
            profile_labels = list(active_profiles.keys())
            profile_matrix = np.array(list(active_profiles.values()))
            if len(profile_labels) != config.n_clusters:
                raise ValueError(f"{config.name} requires one profile per cluster")

            profile_scaler = StandardScaler()
            pc_values = df[['pc1', 'pc2', 'pc3']].to_numpy()
            pc_values_scaled = profile_scaler.fit_transform(pc_values)
            df['cluster'] = cdist(pc_values_scaled, profile_matrix, metric='euclidean').argmin(axis=1)
        else:
            df['cluster'] = kmeans.predict(X_scaled)

        # Calculate centroids in PC space for reporting and label matching.
        cluster_centroids = []
        for i in range(config.n_clusters):
            cluster_data = df[df['cluster'] == i]
            if not cluster_data.empty:
                c_mean = [cluster_data['pc1'].mean(), cluster_data['pc2'].mean(), cluster_data['pc3'].mean()]
            else:
                c_mean = [0, 0, 0] # Should not happen if n_clusters is appropriate
            cluster_centroids.append(c_mean)
            
        cluster_centroids = np.array(cluster_centroids)
        
        # Match with profiles
        label_map = {} # Cluster ID -> Label Name
        
        if config.assign_to_profiles:
            label_map = {i: label for i, label in enumerate(profile_labels)}
            print("  Label Mapping:")
            for cid, label in label_map.items():
                print(f"    Profile {cid} -> {label}")
        elif config.name == 'Middle':
            label_map = build_middle_label_map(cluster_centroids)
            print("  Label Mapping:")
            for cid, label in label_map.items():
                print(f"    Cluster {cid} -> {label}")
        elif config.profiles:
            # Filter profiles to only those in labels list
            active_profiles = {k: v for k, v in config.profiles.items() if k in config.labels}
            profile_labels = list(active_profiles.keys())
            profile_matrix = np.array(list(active_profiles.values()))
            
            # Standardize centroids for comparison
            scaler_centroids = StandardScaler()
            centroids_scaled = scaler_centroids.fit_transform(cluster_centroids)
            
            # Distance matrix
            cost_matrix = cdist(centroids_scaled, profile_matrix, metric='euclidean')
            
            # Hungarian Algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for row, col in zip(row_ind, col_ind):
                label_map[row] = profile_labels[col]
                
            print("  Label Mapping:")
            for cid, label in label_map.items():
                print(f"    Cluster {cid} -> {label}")
        else:
            # Fallback: Map by index
            for i in range(config.n_clusters):
                label_map[i] = config.labels[i] if i < len(config.labels) else f"Cluster {i}"
                
        models[config.name] = {
            'scaler': scaler,
            'kmeans': kmeans,
            'pcas': pcas,
            'label_map': label_map,
            'centroids': cluster_centroids,
            'assignment': 'profiles' if config.assign_to_profiles else 'kmeans',
            'profile_scaler': profile_scaler,
            'profile_matrix': profile_matrix,
        }
        
    return models

# --- Generation ---

def export_position_name(config):
    return 'Edge' if config.name == '2nd Row' else config.name


def json_number(value, digits=None):
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not np.isfinite(value):
            return None
        return round(value, digits) if digits is not None else value

    if isinstance(value, (int,)):
        return int(value)

    return value


def build_stat_map(row, features, suffix="", digits=3):
    out = {}
    for feature in features:
        key = f"{feature}{suffix}"
        value = json_number(row.get(key), digits)
        if value is not None:
            out[feature] = value
    return out


def build_player_archetype_record(row, config, features):
    return {
        "player": str(row["player"]),
        "year": int(row["year"]),
        "position": export_position_name(config),
        "source_position": config.name,
        "archetype": str(row["cluster_name"]),
        "cluster_id": int(row["cluster"]),
        "games": int(row["games"]),
        "minutes": json_number(row.get("mins_played"), 2),
        "total_minutes": json_number(row.get("total_minutes"), 2),
        "pc1": json_number(row.get("pc1"), 4),
        "pc2": json_number(row.get("pc2"), 4),
        "pc3": json_number(row.get("pc3"), 4),
        "pc1_name": config.pc_names[0],
        "pc2_name": config.pc_names[1],
        "pc3_name": config.pc_names[2],
        "centroid_distance": json_number(row.get("centroid_distance"), 4),
        "second_centroid_distance": json_number(row.get("second_centroid_distance"), 4),
        "confidence": json_number(row.get("confidence"), 4),
        "key_stats": build_stat_map(row, features),
        "key_stat_percentiles": build_stat_map(row, features, suffix="_percentile", digits=2),
    }


def upsert_player_archetypes(records):
    if not records:
        print("\nNo player archetype rows to upsert.")
        return

    print(f"\nUpserting {len(records)} rows to nrl.{ARCHETYPE_TABLE}...")
    for start in range(0, len(records), UPSERT_BATCH_SIZE):
        batch = records[start:start + UPSERT_BATCH_SIZE]
        (
            supabase
            .schema("nrl")
            .table(ARCHETYPE_TABLE)
            .upsert(batch, on_conflict="player,year,position")
            .execute()
        )

    removed_below_minimum = 0
    for config in POSITION_CONFIGS:
        response = (
            supabase
            .schema("nrl")
            .table(ARCHETYPE_TABLE)
            .delete()
            .eq("position", export_position_name(config))
            .in_("year", YEARS_TO_PROCESS)
            .lt("games", config.min_games)
            .execute()
        )
        removed_below_minimum += len(response.data or [])
    if removed_below_minimum:
        print(f"Removed {removed_below_minimum} archetype rows below their position minimum.")

    canonical_players = {record["player"] for record in records}
    stale_aliases = [
        alias
        for alias, canonical in PLAYER_NAME_ALIASES.items()
        if canonical in canonical_players
    ]
    if stale_aliases:
        (
            supabase
            .schema("nrl")
            .table(ARCHETYPE_TABLE)
            .delete()
            .in_("player", stale_aliases)
            .execute()
        )
        print(f"Removed stale player aliases: {', '.join(stale_aliases)}.")
    print(f"Upserted {len(records)} rows to nrl.{ARCHETYPE_TABLE}.")

def generate_outputs(training_agg, models, configs, plot_suffix="", stat_mode="production", game_window=None):
    full_cluster_data_export = {}
    player_archetype_records = []
    
    # Only process "All" as requested by user
    process_years = ["All"]
    
    for year in process_years:
        print(f"\n====== GENERATING OUTPUTS FOR {year} ======")
        cluster_data_export = {}
        
        if year == "All":
            year_data = training_agg.copy()
            # Create hover label with year
            year_data['hover_label'] = year_data['player'] + " (" + year_data['year'].astype(str) + ")"
        else:
            year_data = training_agg[training_agg['year'] == year].copy()
            year_data['hover_label'] = year_data['player']
        
        if year_data.empty:
            print(f"No data for {year}")
            continue
            
        for config in configs:
            if config.name not in models:
                continue
                
            model_data = models[config.name]
            
            # Filter data
            df = year_data[year_data['position'] == config.name].copy()
            minimum_games = min(config.min_games, game_window) if game_window else config.min_games
            df = df[df['games'] >= minimum_games]
            
            if config.name == '2nd Row':
                df = df[df['player'] != 'Chris Randall']
                
            if df.empty:
                print(f"  No players for {config.name} in {year}")
                continue
                
            # Transform features and calculate the visible PC coordinates first.
            all_features = list(set(config.features1 + config.features2 + config.features3))
            X = df[all_features].fillna(0)
            X_scaled = model_data['scaler'].transform(X)

            for pc_key, pc_info in model_data['pcas'].items():
                X_sub = df[pc_info['features']].fillna(0)
                X_sub_scaled = pc_info['scaler'].transform(X_sub)
                df[pc_key] = pc_info['model'].transform(X_sub_scaled)

            if model_data['assignment'] == 'profiles':
                pc_values = df[['pc1', 'pc2', 'pc3']].to_numpy()
                pc_values_scaled = model_data['profile_scaler'].transform(pc_values)
                distances = cdist(pc_values_scaled, model_data['profile_matrix'], metric='euclidean')
                df['cluster'] = distances.argmin(axis=1)
            else:
                df['cluster'] = model_data['kmeans'].predict(X_scaled)
                distances = model_data['kmeans'].transform(X_scaled)

            assigned_clusters = df['cluster'].to_numpy(dtype=int)
            nearest_distances = distances[np.arange(len(df)), assigned_clusters]
            second_distances = np.partition(distances, 1, axis=1)[:, 1] if config.n_clusters > 1 else np.full(len(df), np.nan)
            confidence = np.where(
                second_distances > 0,
                np.clip((second_distances - nearest_distances) / second_distances, 0, 1),
                1,
            )

            df['centroid_distance'] = nearest_distances
            df['second_centroid_distance'] = second_distances
            df['confidence'] = confidence

            # Map Labels
            df['cluster_name'] = df['cluster'].map(model_data['label_map'])
            df = apply_archetype_assignment_rules(df, config)

            percentile_features = sorted(all_features)
            percentile_df = df.groupby('year')[percentile_features].rank(pct=True, method='average') * 100
            for feature in percentile_features:
                df[f'{feature}_percentile'] = percentile_df[feature]

            if year == "All":
                player_archetype_records.extend(
                    build_player_archetype_record(row, config, percentile_features)
                    for _, row in df.iterrows()
                )
                
            # Prepare Export Data
            # Use 'Edge' instead of '2nd Row' for export key if needed, but config name is used
            export_name = export_position_name(config)
            
            position_data = {
                "stat_mode": stat_mode,
                "archetypes": [],
                "pc_axes": {
                    "pc1": {"name": config.pc_names[0], "features": config.features1},
                    "pc2": {"name": config.pc_names[1], "features": config.features2},
                    "pc3": {"name": config.pc_names[2], "features": config.features3}
                }
            }
            
            archetype_map = {}
            # We iterate through the DEFINED labels to ensure order/completeness
            # But we only count players present in this year
            
            # Get counts
            counts = df['cluster_name'].value_counts()
            
            for i, label in enumerate(config.labels):
                count = int(counts.get(label, 0))
                description = config.descriptions[i] if i < len(config.descriptions) else ""
                
                archetype_map[label] = {
                    "id": i,
                    "name": label,
                    "count": count,
                    "description": description
                }
                
            position_data["archetypes"] = list(archetype_map.values())
            cluster_data_export[export_name] = position_data
            
            # Generate Plot
            if year == "All":
                # For the "All" view, we create separate traces for each (Year, Archetype)
                # to make filtering by year much more robust.
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly
                archetypes = config.labels
                
                for i, arch in enumerate(archetypes):
                    color = colors[i % len(colors)]
                    legend_shown = False
                    for y in YEARS_TO_PROCESS:
                        mask = (df['cluster_name'] == arch) & (df['year'] == y)
                        sub_df = df[mask]
                        if sub_df.empty:
                            continue
                            
                        fig.add_trace(go.Scatter3d(
                            x=sub_df['pc1'],
                            y=sub_df['pc2'],
                            z=sub_df['pc3'],
                            mode='markers',
                            name=arch,
                            marker=dict(size=5, color=color, opacity=0.8),
                            hovertext=sub_df['hover_label'],
                            hoverinfo='text',
                            legendgroup=arch,
                            showlegend=not legend_shown, # Show each archetype once in legend (first year it has data)
                            customdata=np.full(len(sub_df), y) # Store year for filtering
                        ))
                        legend_shown = True

                # Interactive Year Filter Buttons
                buttons = []
                # "All Years" button
                buttons.append(dict(
                    label="All Years",
                    method="update",
                    args=[{"marker.opacity": 0.8, "hoverinfo": "text"}]
                ))
                
                for target_y in YEARS_TO_PROCESS:
                    opacities = []
                    hoverinfos = []
                    for trace in fig.data:
                        # Each trace has a single year in its customdata
                        trace_tag = trace.customdata[0]
                        if int(trace_tag) == target_y:
                            opacities.append(0.8)
                            hoverinfos.append("text")
                        else:
                            opacities.append(0.15)
                            hoverinfos.append("none")
                    
                    buttons.append(dict(
                        label=str(target_y),
                        method="update",
                        args=[{"marker.opacity": opacities, "hoverinfo": hoverinfos}]
                    ))
                
                fig.update_layout(
                    updatemenus=[dict(
                        type="buttons",
                        direction="right",
                        x=0.02,
                        y=0.98,
                        xanchor="left",
                        yanchor="top",
                        buttons=buttons,
                        showactive=True,
                        active=0, # Set "All Years" as active by default
                        bgcolor="white", # Set default background to white
                        font=dict(color="#0A1128", size=11), # Default navy text
                        bordercolor="#2A3B6E", # Navy border for non-active
                        borderwidth=1
                    )],
                    scene=dict(
                        xaxis=dict(title=config.pc_names[0], showspikes=False),
                        yaxis=dict(title=config.pc_names[1], showspikes=False),
                        zaxis=dict(title=config.pc_names[2], showspikes=False),
                        dragmode='turntable'
                    )
                )
            else:
                # Standard px plot for individual years
                fig = px.scatter_3d(
                    df,
                    x='pc1',
                    y='pc2',
                    z='pc3',
                    color='cluster_name',
                    hover_name='hover_label',
                    opacity=0.8,
                    labels={
                        'pc1': config.pc_names[0],
                        'pc2': config.pc_names[1],
                        'pc3': config.pc_names[2],
                        'cluster_name': 'Archetype'
                    }
                )
                fig.update_traces(marker=dict(size=5))
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(showspikes=False),
                        yaxis=dict(showspikes=False),
                        zaxis=dict(showspikes=False),
                        dragmode='turntable'
                    )
                )

            fig.update_layout(
                legend_title_text='Archetype',
                margin=dict(l=0, r=0, b=0, t=30),
                paper_bgcolor='#f0f0f0',
                plot_bgcolor='#f0f0f0',
                font=dict(color="#0A1128")
            )
            
            suffix = f"_{plot_suffix}" if plot_suffix else ""
            filename = f"nrl_cluster_plot_{export_name.lower().replace(' ', '_')}{suffix}_{str(year).lower()}.html"
            
            # Inject custom CSS and JS to fix Plotly button styling, add mobile responsiveness,
            # and allow projecting the 3D archetype space onto any 2D plane.
            custom_head = """
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <style>
                body {
                    margin: 0;
                }
                #plotly-wrapper {
                    position: relative;
                }
                #dimension-toggle {
                    position: absolute;
                    top: 44px;
                    left: 10px;
                    z-index: 20;
                    display: flex;
                    gap: 4px;
                    flex-wrap: nowrap;
                    max-width: calc(100% - 90px);
                    padding: 3px;
                    background: rgba(10, 17, 40, 0.74);
                    border: 1px solid rgba(248, 250, 252, 0.24);
                    border-radius: 6px;
                    box-shadow: 0 2px 8px rgba(10, 17, 40, 0.22);
                }
                .dimension-toggle-btn {
                    appearance: none;
                    border: 1px solid rgba(248, 250, 252, 0.34);
                    border-radius: 4px;
                    background: rgba(15, 23, 42, 0.92);
                    color: #f8fafc;
                    cursor: pointer;
                    font: 700 9px/1.1 "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                    padding: 3px 6px;
                    white-space: nowrap;
                }
                .dimension-toggle-btn:hover,
                .dimension-toggle-btn:focus-visible {
                    border-color: #C9FF00;
                    outline: none;
                }
                .dimension-toggle-btn.is-dropped {
                    background: rgba(201, 255, 0, 0.14);
                    border-color: #C9FF00;
                    color: #ffffff;
                }
                #plotly-wrapper .updatemenu-button rect.updatemenu-item-bg {
                    rx: 8px !important;
                    ry: 8px !important;
                    fill: white !important;
                }
                #plotly-wrapper .updatemenu-item-text {
                    fill: #0A1128 !important;
                }
                
                /* Mobile specific adjustments */
                @media (max-width: 768px) {
                    #plotly-wrapper .updatemenu-button rect.updatemenu-item-bg {
                        height: 35px !important; /* Larger touch target */
                    }
                    #plotly-wrapper .updatemenu-item-text {
                        font-size: 14px !important;
                    }
                    /* Prevent flicker by hiding buttons until positioned */
                    #plotly-wrapper .updatemenu-container {
                        visibility: hidden;
                    }
                    #plotly-wrapper.ready .updatemenu-container {
                        visibility: visible;
                    }
                    #dimension-toggle {
                        top: 48px;
                        left: 6px;
                        max-width: calc(100% - 64px);
                    }
                    .dimension-toggle-btn {
                        flex: 1 1 auto;
                        min-width: 0;
                        padding: 5px 6px;
                    }
                }
            </style>
            <script>
            const projectionDimensions = [
                { key: 'pc1', axis: 'x', label: __PC1_NAME__ },
                { key: 'pc2', axis: 'y', label: __PC2_NAME__ },
                { key: 'pc3', axis: 'z', label: __PC3_NAME__ }
            ];
            let droppedProjectionDimension = null;
            let originalProjectionData = null;

            function applyButtonStyles() {
                const rects = document.querySelectorAll('.updatemenu-item-bg');
                const isMobile = window.innerWidth < 768;
                
                rects.forEach(rect => {
                    rect.setAttribute('rx', '8');
                    rect.setAttribute('ry', '8');
                    
                    if (isMobile) {
                        rect.setAttribute('height', '35');
                    }
                    
                    const parentGroup = rect.closest('.updatemenu-button');
                    const text = parentGroup ? parentGroup.querySelector('.updatemenu-item-text') : null;

                    if (parentGroup && parentGroup.classList.contains('active')) {
                        rect.style.fill = 'white';
                        rect.style.stroke = '#C9FF00';
                        rect.style.strokeWidth = '2px';
                        if (text) {
                            text.style.fill = 'black';
                            text.setAttribute('fill', 'black');
                        }
                    } else {
                        rect.style.fill = 'white';
                        rect.style.stroke = '#2A3B6E';
                        rect.style.strokeWidth = '1px';
                        if (text) {
                            text.style.fill = '#0A1128';
                            text.setAttribute('fill', '#0A1128');
                        }
                    }
                });
            }

            function readTraceValue(trace, axis) {
                const value = trace[axis] || [];
                if (ArrayBuffer.isView(value)) {
                    return Array.from(value);
                }
                return Array.isArray(value) ? [...value] : value;
            }

            function ensureProjectionData(gd) {
                if (originalProjectionData) return true;
                if (!gd || !gd.data || !gd.data.length) return false;

                const firstTrace = gd.data[0];
                if (!firstTrace || !firstTrace.x || !firstTrace.y || !firstTrace.z) return false;

                originalProjectionData = gd.data.map(trace => ({
                    x: readTraceValue(trace, 'x'),
                    y: readTraceValue(trace, 'y'),
                    z: readTraceValue(trace, 'z')
                }));

                return true;
            }

            function getProjectionTrace(trace, index, dimensionsToKeep) {
                const base = { ...trace };
                const source = originalProjectionData[index];
                const firstAxis = dimensionsToKeep[0].axis;
                const secondAxis = dimensionsToKeep[1].axis;

                delete base.scene;
                delete base.z;
                base.type = 'scatter';
                base.mode = trace.mode || 'markers';
                base.x = source[firstAxis];
                base.y = source[secondAxis];
                base.marker = { ...(trace.marker || {}) };

                return base;
            }

            function getRestoredTrace(trace, index) {
                const base = { ...trace };
                const source = originalProjectionData[index];

                delete base.xaxis;
                delete base.yaxis;
                base.type = 'scatter3d';
                base.mode = trace.mode || 'markers';
                base.x = source.x;
                base.y = source.y;
                base.z = source.z;
                base.marker = { ...(trace.marker || {}) };

                return base;
            }

            function getProjectedAxis(label) {
                return {
                    title: { text: label, font: { color: "#f8fafc", size: 15 } },
                    tickfont: { color: "#f8fafc", size: 12 },
                    showline: false,
                    mirror: false,
                    zeroline: true,
                    zerolinecolor: "rgba(229, 231, 235, 0.82)",
                    zerolinewidth: 2,
                    showgrid: true,
                    gridcolor: "rgba(229, 231, 235, 0.28)",
                    gridwidth: 1,
                    ticks: ""
                };
            }

            function applyProjection() {
                const gd = document.querySelector('.plotly-graph-div');
                if (!gd || !window.Plotly) return;

                if (!ensureProjectionData(gd)) return;

                const dimensionsToKeep = projectionDimensions.filter(dimension => dimension.key !== droppedProjectionDimension);
                const projectedData = gd.data.map((trace, index) => (
                    droppedProjectionDimension
                        ? getProjectionTrace(trace, index, dimensionsToKeep)
                        : getRestoredTrace(trace, index)
                ));
                const baseMargin = gd.layout.margin || {};
                const layout = {
                    ...gd.layout,
                    margin: droppedProjectionDimension
                        ? { ...baseMargin, t: Math.max(baseMargin.t || 0, 82), r: Math.max(baseMargin.r || 0, 64) }
                        : { ...baseMargin },
                    legend: { ...gd.layout.legend },
                    scene: {
                        ...(gd.layout.scene || {}),
                        xaxis: { ...((gd.layout.scene || {}).xaxis || {}), title: { text: projectionDimensions[0].label }, showspikes: false },
                        yaxis: { ...((gd.layout.scene || {}).yaxis || {}), title: { text: projectionDimensions[1].label }, showspikes: false },
                        zaxis: { ...((gd.layout.scene || {}).zaxis || {}), title: { text: projectionDimensions[2].label }, showspikes: false },
                        dragmode: 'turntable'
                    },
                    xaxis: droppedProjectionDimension
                        ? getProjectedAxis(dimensionsToKeep[0].label)
                        : gd.layout.xaxis,
                    yaxis: droppedProjectionDimension
                        ? { ...getProjectedAxis(dimensionsToKeep[1].label), scaleanchor: 'x', scaleratio: 1 }
                        : gd.layout.yaxis,
                    dragmode: droppedProjectionDimension ? 'pan' : gd.layout.dragmode
                };

                Plotly.react(gd, projectedData, layout, {
                    responsive: true,
                    scrollZoom: true,
                    displaylogo: false
                }).then(() => {
                    updateProjectionAttributes();
                    applyButtonStyles();
                    adjustPlotlyForMobile();
                });
            }

            function updateProjectionAttributes() {
                const wrapper = document.getElementById('plotly-wrapper');
                if (!wrapper) return;

                wrapper.dataset.projectionMode = droppedProjectionDimension ? '2d' : '3d';
                wrapper.dataset.droppedDimension = droppedProjectionDimension || '';
                wrapper.dataset.projectionReady = originalProjectionData ? 'true' : 'false';
            }

            function renderDimensionToggle() {
                const wrapper = document.getElementById('plotly-wrapper');
                const gd = document.querySelector('.plotly-graph-div');
                if (!wrapper || !gd || wrapper.querySelector('#dimension-toggle')) return;

                if (!ensureProjectionData(gd)) return;
                updateProjectionAttributes();

                const controls = document.createElement('div');
                controls.id = 'dimension-toggle';
                projectionDimensions.forEach(dimension => {
                    const button = document.createElement('button');
                    button.type = 'button';
                    button.className = 'dimension-toggle-btn';
                    button.textContent = dimension.label;
                    button.title = `Toggle ${dimension.label} projection`;
                    button.addEventListener('click', () => {
                        droppedProjectionDimension = droppedProjectionDimension === dimension.key ? null : dimension.key;
                        controls.querySelectorAll('.dimension-toggle-btn').forEach(btn => {
                            btn.classList.toggle('is-dropped', btn === button && droppedProjectionDimension === dimension.key);
                        });
                        applyProjection();
                    });
                    controls.appendChild(button);
                });
                wrapper.appendChild(controls);
            }

            function adjustPlotlyForMobile() {
                const gd = document.querySelector('.plotly-graph-div');
                if (!gd) return;
                
                if (window.innerWidth < 768) {
                    const update = {
                        legend: {
                            orientation: 'h',
                            y: -0.15,
                            x: 0.5,
                            xanchor: 'center',
                            yanchor: 'top',
                            font: { size: 10 }
                        },
                        margin: { l: 5, r: 5, b: 100, t: 50 },
                        updatemenus: [{
                            ...gd.layout.updatemenus[0],
                            x: 0.5,
                            y: 1.12, /* Move higher to avoid modebar */
                            xanchor: 'center',
                            yanchor: 'bottom',
                            direction: 'right',
                            font: { size: 10 }
                        }],
                        modebar: {
                            orientation: 'v',
                            x: 1,
                            y: 0.5,
                            xanchor: 'right',
                            yanchor: 'middle'
                        }
                    };
                    Plotly.relayout(gd, update).then(() => {
                        Plotly.relayout(gd, { 'scene.dragmode': 'turntable' });
                        Plotly.restyle(gd, { 'marker.size': 6 });
                        document.getElementById('plotly-wrapper').classList.add('ready');
                    });
                } else {
                    document.getElementById('plotly-wrapper').classList.add('ready');
                }
            }

            // Watch for changes to the plot
            const observer = new MutationObserver((mutations) => {
                applyButtonStyles();
            });

            document.addEventListener('DOMContentLoaded', () => {
                const target = document.body;
                observer.observe(target, { childList: true, subtree: true });
                applyButtonStyles();
                renderDimensionToggle();
                
                // Initial mobile adjustment - faster timeout
                setTimeout(() => {
                    renderDimensionToggle();
                    adjustPlotlyForMobile();
                }, 100);
            });
            
            window.addEventListener('resize', () => {
                applyButtonStyles();
                adjustPlotlyForMobile();
            });
            
            // Also run on a timer as a fallback
            setInterval(() => {
                applyButtonStyles();
                renderDimensionToggle();
            }, 1000);
            </script>
            """
            custom_head = (custom_head
                .replace('__PC1_NAME__', json.dumps(config.pc_names[0]))
                .replace('__PC2_NAME__', json.dumps(config.pc_names[1]))
                .replace('__PC3_NAME__', json.dumps(config.pc_names[2]))
            )
            
            html_content = fig.to_html(
                include_plotlyjs='cdn', 
                full_html=True,
                config={'responsive': True, 'scrollZoom': True, 'displaylogo': False}
            )
            
            # Inject into head
            head_end = html_content.find('</head>')
            if head_end != -1:
                html_content = html_content[:head_end] + custom_head + html_content[head_end:]
            
            # Wrap body content in our wrapper
            body_start = html_content.find('<body>') + 6
            body_end = html_content.find('</body>')
            if body_start != 5 and body_end != -1:
                html_content = (html_content[:body_start] + 
                               '<div id="plotly-wrapper" style="height:100%; width:100%;">' + 
                               html_content[body_start:body_end] + 
                               '</div>' + 
                               html_content[body_end:])
            
            with open(filename, 'w') as output_file:
                output_file.write(html_content)
            print(f"  Saved plot {filename}")
            
        full_cluster_data_export[str(year)] = cluster_data_export
        
    return full_cluster_data_export, player_archetype_records


def save_cluster_exports(data, file_stem, variable_name):
    with open(f'{file_stem}.json', 'w') as output_file:
        json.dump(data, output_file, indent=4)
    with open(f'{file_stem}.js', 'w') as output_file:
        output_file.write(f"const {variable_name} = {json.dumps(data, indent=4)};")
    print(f"Exported {file_stem}.json and {file_stem}.js")

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Load Data
    print("Fetching player stats...")
    player_data = fetch_player_stats_for_years(YEARS_TO_PROCESS, POSITION_CONFIGS)
    training_agg = load_and_process_data(POSITION_CONFIGS, player_data, stat_mode='production')
    
    # 2. Train Models (Global)
    models = train_models(training_agg, POSITION_CONFIGS)
    
    # 3. Generate Outputs (Per Year)
    full_data, player_archetype_records = generate_outputs(training_agg, models, POSITION_CONFIGS)
    
    # 4. Save JSON and browser data
    save_cluster_exports(full_data, 'nrl_cluster_data', 'clusterData')

    # 5. Upsert player-level archetype outputs
    if os.getenv("SKIP_ARCHETYPE_UPSERT") == "1":
        print("Skipped player archetype upsert.")
    else:
        upsert_player_archetypes(player_archetype_records)

    # 6. Generate alternate team-share archetype view
    team_share_configs = build_team_share_configs(POSITION_CONFIGS)
    team_share_training_agg = load_and_process_data(team_share_configs, player_data, stat_mode='team_share')
    team_share_models = train_models(team_share_training_agg, team_share_configs)
    team_share_data, _ = generate_outputs(
        team_share_training_agg,
        team_share_models,
        team_share_configs,
        plot_suffix="team_share",
        stat_mode="team_share",
    )
    save_cluster_exports(team_share_data, 'nrl_cluster_data_team_share', 'clusterDataTeamShare')

    # 7. Generate recent qualifying-game views.
    for game_window in (3, 5, 10):
        window_label = f'l{game_window}'
        variable_suffix = f'L{game_window}'

        window_training_agg = load_and_process_data(
            POSITION_CONFIGS,
            player_data,
            stat_mode='production',
            recent_games=game_window,
        )
        # Keep the All player-year feature space and cluster centroids fixed.
        window_data, _ = generate_outputs(
            window_training_agg,
            models,
            POSITION_CONFIGS,
            plot_suffix=window_label,
            game_window=game_window,
        )
        save_cluster_exports(window_data, f'nrl_cluster_data_{window_label}', f'clusterData{variable_suffix}')

        window_team_share_agg = load_and_process_data(
            team_share_configs,
            player_data,
            stat_mode='team_share',
            recent_games=game_window,
        )
        # Team-share windows likewise reuse the All team-share model.
        window_team_share_data, _ = generate_outputs(
            window_team_share_agg,
            team_share_models,
            team_share_configs,
            plot_suffix=f'team_share_{window_label}',
            stat_mode='team_share',
            game_window=game_window,
        )
        save_cluster_exports(
            window_team_share_data,
            f'nrl_cluster_data_team_share_{window_label}',
            f'clusterDataTeamShare{variable_suffix}',
        )
