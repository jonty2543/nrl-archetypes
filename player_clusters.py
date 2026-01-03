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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# --- Configuration ---

SUPABASE_URL = "https://glrzwxpxkckxaogpkwmn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdscnp3eHB4a2NreGFvZ3Brd21uIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NjA3OTU3NiwiZXhwIjoyMDcxNjU1NTc2fQ.YOF9ryJbhBoKKHT0n4eZDMGrR9dczR8INHVs_By4vRU"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

YEARS_TO_PROCESS = [2022, 2023, 2024, 2025]

class PositionConfig:
    def __init__(self, name, features1, features2, features3, pc_names, n_clusters, labels, descriptions, min_games=5, profiles=None):
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
    'Link Centre': [1.5, 0.0, 0.0],
    'Strike Centre': [0.0, 1.5, 0.0],
    'Workhorse Centre': [0.0, 0.0, 1.5],
    'Support Centre': [-1.0, -1.0, -1.0]
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

POSITION_CONFIGS = [
    PositionConfig(
        name='Fullback',
        features1=['line_break_assists_per_80', 'try_assists_per_80', 'passes_per_80'],
        features2=['line_breaks_per_80', 'tries_per_80', 'tackle_breaks_per_80'],
        features3=['all_run_metres_per_80', 'hit_ups_per_80', 'post_contact_metres_per_80'],
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
        profiles=PROFILES_FULLBACK
    ),
    PositionConfig(
        name='Winger',
        features1=['tackle_breaks_per_80', 'offloads_per_80'],
        features2=['tries_per_80','line_breaks_per_80'],
        features3=['all_run_metres_per_80'],
        pc_names=['Strength In Contact', 'Try Scoring', 'Workrate'],
        n_clusters=3,
        labels=['Support Winger', 'Finisher Winger', 'Workhorse Winger'],
        descriptions=[
            "These wingers tend to be less involved in the game, perhaps due to lack of skill or opportunity.",
            "Wingers who are specialist try scorers, often with great positional awareness and speed.",
            "High involvement wingers who are strong in contact, often taking carries out of their own end."
        ],
        min_games=5,
        profiles=PROFILES_WINGER
    ),
    PositionConfig(
        name='Centre',
        features1=['passes_per_80', 'pass_run_ratio'],
        features2=['tries_per_80','line_breaks_per_80'],
        features3=['all_run_metres_per_80', 'tackle_breaks_per_80', 'hit_ups_per_80'],
        pc_names=['Passing', 'Try Scoring', 'Workrate'],
        n_clusters=4,
        labels=['Link Centre', 'Workhorse Centre', 'Support Centre', 'Strike Centre'],
        descriptions=[
            "These centres play more of a Five-Eighth role with a high pass to run ratio, often looking to set up their winger.",
            "HAttacking weapons who are heavily involved in gaining metres aswell as breaking the line and scoring tries.",
            "These players are less involved with ball in hand and may play other roles for the team.",
            "Centres who are heavily involved in try scoring, and may look to set up those around them rather than taking tough carries."
        ],
        min_games=5,
        profiles=PROFILES_CENTRE
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
        features2=['try_assists', 'line_break_assists'],
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
        min_games=7,
        profiles=PROFILES_HOOKER
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
        min_games=7,
        profiles=PROFILES_EDGE
    ),
    PositionConfig(
        name='Middle',
        features1=['passes_to_run_ratio', 'passes', 'line_break_assists', 'try_assists'],
        features2=['all_run_metres', 'tackle_breaks', 'post_contact_metres', 'offloads'],
        features3=['tackles_made', 'tackle_efficiency'],
        pc_names=['Ball Playing', 'Ball Running', 'Defense'],
        n_clusters=3,
        labels=['Ball Playing Middle', 'Impact Middle', 'Standard Middle'],
        descriptions=[
            "These middles often play in the lock position with strong ball playing skills, directing players in the middle of the park.",
            "The most effective hit up takers, these middles are characterised by their strength and big engines.",
            "Making up the rest of the middle, these players share the hit up and tackling duties."
        ],
        min_games=7,
        profiles=PROFILES_MIDDLE
    )
]

# --- Data Loading & Preprocessing ---

def load_and_process_data():
    print("Fetching player stats...")
    player_data = f.fetch_all("player_stats")
    
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
    
    # Calculate per_80 stats
    num_cols = player_df.select_dtypes('number').columns
    for col in num_cols:
        if col not in ['year', 'mins_played']: # Avoid overwriting or dividing year
            player_df[f'{col}_per_80'] = player_df[col] * (80 / player_df['mins_played'])
            
    per_80_cols = [c for c in player_df.columns if c.endswith('_per_80')]
    
    # Map positions
    player_df['position'] = player_df['number'].apply(f.map_position)
    
    # Other features
    player_df['metres_per_run'] = player_df['all_run_metres'] / player_df['all_runs']
    
    # Aggregate per player per year
    # We need to support both 'per_80' (mean of per_80) and 'raw' (mean of per match)
    # The original code used 'player_agg' (per_80) and 'player_agg_unadjusted' (raw + per_80)
    # We will create one super-aggregated dataframe with ALL columns
    
    agg_dict = {
        'games': ('match_date', 'nunique'),
        'position': ('position', lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
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
        .groupby(['player', 'year'], as_index=False) # Group by Player AND Year
        .agg(**agg_dict)
    )
    
    # Fix position if it was lost or weird (it's aggregated by mode)
    # Also we need to filter by position later, so ensure it's there.
    
    return training_agg

# --- Model Training ---

def train_models(training_agg, configs):
    models = {}
    
    print("\n====== TRAINING MODELS (GLOBAL) ======")
    
    for config in configs:
        print(f"\nTraining {config.name}...")
        
        # Filter data
        df = training_agg[training_agg['position'] == config.name].copy()
        df = df[df['games'] >= config.min_games]
        
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
            
        # 4. Determine Label Mapping (Dynamic Assignment)
        # We need to predict clusters for the training data to get centroids
        df['cluster'] = kmeans.predict(X_scaled)
        
        # Calculate centroids in PC space (using the global PCAs)
        for pc_key, pc_data in pcas.items():
            X_sub = df[pc_data['features']].fillna(0)
            X_sub_scaled = pc_data['scaler'].transform(X_sub)
            df[pc_key] = pc_data['model'].transform(X_sub_scaled)
            
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
        
        if config.profiles:
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
            'centroids': cluster_centroids
        }
        
    return models

# --- Generation ---

def generate_outputs(training_agg, models, configs):
    full_cluster_data_export = {}
    
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
            df = df[df['games'] >= config.min_games]
            
            if config.name == '2nd Row':
                df = df[df['player'] != 'Chris Randall']
                
            if df.empty:
                print(f"  No players for {config.name} in {year}")
                continue
                
            # Transform Features
            all_features = list(set(config.features1 + config.features2 + config.features3))
            X = df[all_features].fillna(0)
            X_scaled = model_data['scaler'].transform(X)
            
            # Predict Clusters
            df['cluster'] = model_data['kmeans'].predict(X_scaled)
            
            # Map Labels
            df['cluster_name'] = df['cluster'].map(model_data['label_map'])
            
            # Calculate PCs
            for pc_key, pc_info in model_data['pcas'].items():
                X_sub = df[pc_info['features']].fillna(0)
                X_sub_scaled = pc_info['scaler'].transform(X_sub)
                df[pc_key] = pc_info['model'].transform(X_sub_scaled)
                
            # Prepare Export Data
            # Use 'Edge' instead of '2nd Row' for export key if needed, but config name is used
            export_name = 'Edge' if config.name == '2nd Row' else config.name
            
            position_data = {
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

                # Add Centroids
                centroids = model_data['centroids']
                for i, arch in enumerate(archetypes):
                    color = colors[i % len(colors)]
                    # The centroid index matches the cluster ID, but we need to find which cluster ID maps to this label
                    # Find cluster ID for this label
                    cid = next((k for k, v in model_data['label_map'].items() if v == arch), None)
                    if cid is not None:
                        c_pos = centroids[cid]
                        fig.add_trace(go.Scatter3d(
                            x=[c_pos[0]],
                            y=[c_pos[1]],
                            z=[c_pos[2]],
                            mode='markers',
                            name=f"{arch} Centroid",
                            marker=dict(
                                size=8, 
                                color=color, 
                                symbol='diamond',
                                line=dict(color='white', width=1)
                            ),
                            hovertext=f"Centroid: {arch}",
                            hoverinfo='text',
                            legendgroup=arch,
                            showlegend=False, # Already shown by player traces
                            customdata=np.array(["Centroid"]) # Special tag for filtering
                        ))
                
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
                        # Each trace has a single year or "Centroid" in its customdata
                        trace_tag = trace.customdata[0]
                        if trace_tag == "Centroid" or int(trace_tag) == target_y:
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
                        zaxis=dict(title=config.pc_names[2], showspikes=False)
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
                        zaxis=dict(showspikes=False)
                    )
                )

            fig.update_layout(
                legend_title_text='Archetype',
                margin=dict(l=0, r=0, b=0, t=30),
                paper_bgcolor='#f0f0f0',
                plot_bgcolor='#f0f0f0',
                font=dict(color="#0A1128")
            )
            
            filename = f"nrl_cluster_plot_{export_name.lower().replace(' ', '_')}_{str(year).lower()}.html"
            
            # Inject custom CSS and JS to fix Plotly button styling
            custom_head = """
            <style>
                #plotly-wrapper .updatemenu-button rect.updatemenu-item-bg {
                    rx: 8px !important;
                    ry: 8px !important;
                    fill: white !important;
                }
                #plotly-wrapper .updatemenu-item-text {
                    fill: #0A1128 !important;
                }
            </style>
            <script>
            function applyButtonStyles() {
                const rects = document.querySelectorAll('.updatemenu-item-bg');
                rects.forEach(rect => {
                    rect.setAttribute('rx', '8');
                    rect.setAttribute('ry', '8');
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

            // Watch for changes to the plot
            const observer = new MutationObserver((mutations) => {
                applyButtonStyles();
            });

            document.addEventListener('DOMContentLoaded', () => {
                const target = document.body;
                observer.observe(target, { childList: true, subtree: true });
                applyButtonStyles();
            });
            
            // Also run on a timer as a fallback
            setInterval(applyButtonStyles, 1000);
            </script>
            """
            
            html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
            
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
            
            with open(filename, 'w') as f:
                f.write(html_content)
            print(f"  Saved plot {filename}")
            
        full_cluster_data_export[str(year)] = cluster_data_export
        
    return full_cluster_data_export

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Load Data
    training_agg = load_and_process_data()
    
    # 2. Train Models (Global)
    models = train_models(training_agg, POSITION_CONFIGS)
    
    # 3. Generate Outputs (Per Year)
    full_data = generate_outputs(training_agg, models, POSITION_CONFIGS)
    
    # 4. Save JSON
    with open('nrl_cluster_data.json', 'w') as f:
        json.dump(full_data, f, indent=4)
    print("\nExported cluster data to nrl_cluster_data.json")

    with open('nrl_cluster_data.js', 'w') as f:
        f.write(f"const clusterData = {json.dumps(full_data, indent=4)};")
    print("Exported cluster data to nrl_cluster_data.js")
