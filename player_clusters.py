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
import json
import os

cluster_data_export = {}


SUPABASE_URL = "https://glrzwxpxkckxaogpkwmn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdscnp3eHB4a2NreGFvZ3Brd21uIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NjA3OTU3NiwiZXhwIjoyMDcxNjU1NTc2fQ.YOF9ryJbhBoKKHT0n4eZDMGrR9dczR8INHVs_By4vRU"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

player_data = f.fetch_all("player_stats")

player_df = player_data[
    (player_data['match_date'] >= '2025-01-01') & 
    (player_data['mins_played'] >= 40)
].copy()

joe_chan = player_df[player_df['player'] == 'Joe Chan']

num_cols = player_df.select_dtypes('number').columns
    
for col in num_cols:
    player_df[f'{col}_per_80'] = player_df[col]*(80/player_df['mins_played'])

per_80_cols = [f'{c}_per_80' for c in num_cols]

player_df['position'] = player_df['number'].apply(f.map_position)

player_df['metres_per_run'] = player_df['all_run_metres']/player_df['all_runs']


player_agg = (
    player_df
    .groupby(['player', 'position'], as_index=False)
    .agg(
        games=('match_date', 'nunique'),
        pass_run_ratio=('passes_to_run_ratio', 'mean'),
        tackle_efficiency=('tackle_efficiency', 'mean'),
        position=('position', lambda s: s.mode().iloc[0]),
        **{c: (c, 'mean') for c in per_80_cols}
    )
)

player_agg_unadjusted = (
    player_df
    .groupby(['player', 'position'], as_index=False)
    .agg(
        games=('match_date', 'nunique'),
        position=('position', lambda s: s.mode().iloc[0]),
        **{c: (c, 'mean') for c in per_80_cols},
        **{c: (c, 'mean') for c in num_cols}
    )
)


player_agg = player_agg[player_agg['games'] >= 5]
player_agg_unadjusted = player_agg_unadjusted[player_agg_unadjusted['games'] >= 5]

def clustering_analysis(player_agg, position, features1, features2, features3, clusters, plot, 
                        pc1_name=None, pc2_name=None, pc3_name=None, cluster_labels=None, cluster_descriptions=None):
    """
    Perform clustering analysis with 3 separate feature subsets.
    Runs PC1 analysis on each subset individually to create 3 axes.
    
    Args:
        player_agg: DataFrame with player statistics
        position: Position name for plot title
        features1: List of features for PC1 axis
        features2: List of features for PC2 axis
        features3: List of features for PC3 axis
        clusters: Number of clusters for KMeans
        plot: Whether to generate 3D plot
        pc1_name, pc2_name, pc3_name: Custom axis labels
        cluster_labels: Optional list of cluster names (length must match clusters)
    """
    # Combine all features for clustering
    all_features = list(set(features1 + features2 + features3))
    X_all = player_agg[all_features].fillna(0)
    X_all_scaled = StandardScaler().fit_transform(X_all)
    
    # Perform clustering on all features combined
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    labels = kmeans.fit_predict(X_all_scaled)
    player_agg['cluster'] = labels
    
    # Run PC1 analysis on each feature subset
    print(f"\n=== PC1 Analysis on Feature Subsets ===")
    
    # PC1 from features1
    X1 = player_agg[features1].fillna(0)
    X1_scaled = StandardScaler().fit_transform(X1)
    pca1 = PCA(n_components=1)
    pc1_values = pca1.fit_transform(X1_scaled)
    player_agg['pc1'] = pc1_values
    
    print(f"\nPC1 ({pc1_name or 'Axis 1'}):")
    print(f"  Features: {features1}")
    print(f"  Variance explained: {pca1.explained_variance_ratio_[0]:.3f}")
    loadings1 = pd.Series(pca1.components_[0], index=features1).sort_values(key=lambda x: x.abs(), ascending=False)
    print(f"  Top loadings:\n{loadings1.head(5)}")
    
    # PC2 from features2
    X2 = player_agg[features2].fillna(0)
    X2_scaled = StandardScaler().fit_transform(X2)
    pca2 = PCA(n_components=1)
    pc2_values = pca2.fit_transform(X2_scaled)
    player_agg['pc2'] = pc2_values
    
    print(f"\nPC2 ({pc2_name or 'Axis 2'}):")
    print(f"  Features: {features2}")
    print(f"  Variance explained: {pca2.explained_variance_ratio_[0]:.3f}")
    loadings2 = pd.Series(pca2.components_[0], index=features2).sort_values(key=lambda x: x.abs(), ascending=False)
    print(f"  Top loadings:\n{loadings2.head(5)}")
    
    # PC3 from features3
    X3 = player_agg[features3].fillna(0)
    X3_scaled = StandardScaler().fit_transform(X3)
    pca3 = PCA(n_components=1)
    pc3_values = pca3.fit_transform(X3_scaled)
    player_agg['pc3'] = pc3_values
    
    print(f"\nPC3 ({pc3_name or 'Axis 3'}):")
    print(f"  Features: {features3}")
    print(f"  Variance explained: {pca3.explained_variance_ratio_[0]:.3f}")
    loadings3 = pd.Series(pca3.components_[0], index=features3).sort_values(key=lambda x: x.abs(), ascending=False)
    print(f"  Top loadings:\n{loadings3.head(5)}")
    
    # Generate cluster labels if not provided
    if cluster_labels is None:
        cluster_labels = []
        # Calculate cluster centroids in PC space
        for i in range(clusters):
            cluster_data = player_agg[player_agg['cluster'] == i]
            pc1_mean = cluster_data['pc1'].mean()
            pc2_mean = cluster_data['pc2'].mean()
            pc3_mean = cluster_data['pc3'].mean()
            
            # Determine if each axis is high/low relative to overall mean
            pc1_level = "High" if pc1_mean > player_agg['pc1'].mean() else "Low"
            pc2_level = "High" if pc2_mean > player_agg['pc2'].mean() else "Low"
            pc3_level = "High" if pc3_mean > player_agg['pc3'].mean() else "Low"
            
            # Create descriptive label
            label = f"{pc1_level} {pc1_name or 'PC1'}, {pc2_level} {pc2_name or 'PC2'}, {pc3_level} {pc3_name or 'PC3'}"
            cluster_labels.append(label)
    
    # Map cluster numbers to labels
    player_agg['cluster_name'] = player_agg['cluster'].map(lambda x: cluster_labels[x])
    
    # Prepare data for export
    position_data = {
        "archetypes": [],
        "pc_axes": {
            "pc1": {"name": pc1_name or "PC1", "features": features1},
            "pc2": {"name": pc2_name or "PC2", "features": features2},
            "pc3": {"name": pc3_name or "PC3", "features": features3}
        }
    }
    
    # Print cluster summary
    print(f"\n=== Cluster Summary ===")
    
    # Aggregate clusters by name for export
    archetype_map = {}
    
    for i in range(clusters):
        cluster_size = int((player_agg['cluster'] == i).sum())
        name = cluster_labels[i]
        description = cluster_descriptions[i] if cluster_descriptions and i < len(cluster_descriptions) else None
        
        print(f"Cluster {i}: {name} ({cluster_size} players)")
        
        if name not in archetype_map:
            archetype_map[name] = {
                "id": len(archetype_map), # Assign new sequential ID
                "name": name,
                "count": 0,
                "description": description
            }
        
        archetype_map[name]["count"] += cluster_size
        # Keep the first description found for this name (or update if needed, but assuming they are same)
        
    position_data["archetypes"] = list(archetype_map.values())
        
    cluster_data_export[position] = position_data
    
    # Generate 3D plot
    if plot:
        fig = px.scatter_3d(
            player_agg,
            x='pc1',
            y='pc2',
            z='pc3',
            color='cluster_name',
            hover_name='player',
            opacity=0.8,
            labels={
                'pc1': pc1_name or 'PC1',
                'pc2': pc2_name or 'PC2',
                'pc3': pc3_name or 'PC3',
                'cluster_name': 'Archetype'
            }
        )
        
        fig.update_traces(marker=dict(size=5))
        
        fig.update_layout(
            legend_title_text='Archetype',
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Save individual plot for dashboard
        filename = f"nrl_cluster_plot_{position.lower().replace(' ', '_')}.html"
        fig.write_html(filename, auto_open=False, include_plotlyjs='cdn', full_html=False)
        print(f"Saved plot to {filename}")

'''
#1. All Players
# Ball Running features
features1 = ['tries_per_80', 'all_run_metres_per_80', 'kick_return_metres_per_80', 
             'post_contact_metres_per_80', 'line_breaks_per_80', 'tackle_breaks_per_80', 
             'hit_ups_per_80']
# Defensive features
features2 = ['tackles_made_per_80']
# Usage/Playmaking features
features3 = ['line_break_assists_per_80', 'pass_run_ratio', 'passes_per_80', 
             'receipts_per_80', 'kicking_metres_per_80']

clustering_analysis(player_agg, 'All Positions', features1, features2, features3, 5, True, 'Ball Running', 'Defensive', 'Usage')
'''

#2. Fullbacks

# Playmaking features
features1 = ['line_break_assists_per_80', 'try_assists_per_80']
# Evasiveness features
features2 = ['tries_per_80','line_breaks_per_80']
# Workrate features
features3 = ['all_run_metres_per_80']

fullbacks=player_agg[player_agg['position'] == 'Fullback']

clustering_analysis(fullbacks, 'Fullback', features1, features2, features3, 5, True, 
                    'Playmaking', 'Evasiveness', 'Workrate',
                    cluster_labels=['Ball Running Fullback', 'Balanced Fullback', 'Workhorse Fullback', 'Playmaker Fullback', 'Support Fullback'],
                    cluster_descriptions=[
                        "Fullbacks who are quick and are able to break the defensive line. Tend to struggle with ball playing however.",
                        "These well rounded fullbacks balance workrate, playmaking and elusiveness making them the complete package.",
                        "High-effort players who are always around the ball. They rack up high run metres and support plays.",
                        "These playmakers save their energy for the big moments, with reduced workrates but high involvement in tries and try assists.",
                        "Players who are less involved in attack, but may specialise in defense or defusing kicks."
                    ])


#2. Wingers

# Strength In Contact features
features1 = ['tackle_breaks_per_80', 'offloads_per_80']
# Try Scoring features
features2 = ['tries_per_80','line_breaks_per_80']
# Workrate features
features3 = ['all_run_metres_per_80']

wingers=player_agg[player_agg['position'] == 'Winger']

clustering_analysis(wingers, 'Winger', features1, features2, features3, 3, True, 'Strength In Contact', 'Try Scoring', 'Workrate',
                    cluster_labels=['Support Winger', 'Finisher Winger', 'Workhorse Winger'],
                    cluster_descriptions=[
                        "These wingers tend to be less involved in the game, perhaps due to lack of skill or opportunity.",
                        "Wingers who are specialist try scorers, often with great positional awareness and speed.",
                        "High involvement wingers who are strong in contact, often taking carries out of their own end."
                    ])


#3. Centres
# Playmaking features
features1 = ['passes_per_80', 'pass_run_ratio']
# Try Scoring features
features2 = ['tries_per_80','line_breaks_per_80']
# Workrate features
features3 = ['all_run_metres_per_80', 'tackle_breaks_per_80', 'hit_ups_per_80']


centres=player_agg[player_agg['position'] == 'Centre']

clustering_analysis(centres, 'Centre', features1, features2, features3, 4, True, 'Passing', 'Try Scoring', 'Workrate',
                    cluster_labels=['Link Centre', 'Workhorse Centre', 'Support Centre', 'Strike Centre'],
                    cluster_descriptions=[
                        "These centres play more of a Five-Eighth role with a high pass to run ratio, often looking to set up their winger.",
                        "Highly involved with the ball, these centres have big engines and focus on gaining metres for their team.",
                        "These players balance scoring tries and setting up those around them, often with lower workrate with the ball.",
                        "Attacking weapons who are heavily involved in gaining metres aswell as breaking the line and scoring tries."
                    ])


#4. Halves
# Running features
features1 = ['tries_per_80', 'all_run_metres_per_80', 
             'line_breaks_per_80', 'tackle_breaks_per_80']
# Creativity features
features2 = ['line_break_assists_per_80', 'try_assists_per_80', 'forced_drop_outs_per_80']
# Kicking features
features3 = ['kicks_per_80', 'kicking_metres_per_80', 'one_point_field_goals_per_80']

halves=player_agg[player_agg['position'] == 'Half']

clustering_analysis(halves, 'Half', features1, features2, features3, 3, True, 
                    'Running', 'Creativity', 'Kicking',
                    cluster_labels=['Dominant Half', 'Running Half', 'Organising Half'],
                    cluster_descriptions=[
                        "These players control the attack, and are usually relied upon to set up tries and do most of the kicking.",
                        "Halves with strong running games who look to break the line, usually Five-Eighths.",
                        "Less dominant halves who may rely on their halves partner to control the attack, focusing on organising their edge."
                    ])

player_agg_unadjusted = player_agg_unadjusted[player_agg_unadjusted['games'] >= 7]

#5. Hookers
# Ball Running features
features1 = ['all_run_metres', 'tackle_breaks', 'line_breaks']
# Creativity features
features2 = ['try_assists', 'line_break_assists']
# Pass to run ratio features
features3 = ['passes_to_run_ratio']

hookers=player_agg_unadjusted[player_agg_unadjusted['position'] == 'Hooker']

clustering_analysis(hookers, 'Hooker', features1, features2, features3,4, True, 'Ball Running', 'Creativity', 'Pass - Run Ratio',
                    cluster_labels=['Support Hooker', 'Running Hooker', 'Link Hooker', 'Crafty Hooker'],
                    cluster_descriptions=[
                        "Hookers who are less involved in the attack and may focus on defense instead.",
                        "Strong ball running hookers who often look to run from dummy half.",
                        "Hookers that look to pass rather than run, usually having strong ball playing.",
                        "Creative types who specialise in finding the right pass for their forwards."
                    ])

#6. 2nd Row
# Workrate features
features1 = ['all_run_metres', 'tackle_breaks', 'offloads', 'hit_ups']
# Attacking Threat features
features2 = ['line_breaks', 'tries']
# Defense features
features3 = ['tackles_made', 'tackle_efficiency']

second_row=player_agg_unadjusted[(player_agg_unadjusted['position'] == '2nd Row') & ~(player_agg_unadjusted['player'] == 'Chris Randall')]

clustering_analysis(second_row, 'Edge', features1, features2, features3, 4, True, 
                    'Attacking Workrate', 'Attacking Threat', 'Defensive Workrate',
                    cluster_labels=['Defensive Enforcer Edge', 'Support Edge', 'Strong Attacking Edge', 'Strike Attacking Edge'],
                    cluster_descriptions=[
                        "Defensive specialists who are key in protecting their edge. Less involved in attacking situations.",
                        "These edges are less involved in attack and defense, and may specialise in other areas.",
                        "These players are strong in contact and are relied upon to make metres for their team, often involved in tries as a result.",
                        "Great line runners, often breaking the line and scoring tries, playing like a centre in attack."
                    ])


#7. Middles
# Ball Playing features
features1 = ['passes_to_run_ratio', 'passes', 'line_break_assists', 'try_assists']
# AWorkrate features
features2 = ['all_run_metres', 'tackle_breaks', 'post_contact_metres', 'offloads']
# Defensive Workrate features
features3 = ['tackles_made', 'tackle_efficiency']

middle=player_agg_unadjusted[player_agg_unadjusted['position'] == 'Middle']

clustering_analysis(middle, 'Middle', features1, features2, features3, 3, True, 'Ball Playing', 'Ball Running', 'Defense',
                    cluster_labels=['Ball Playing Middle', 'Impact Middle', 'Standard Middle'],
                    cluster_descriptions=[
                        "These middles often play in the lock position with strong ball playing skills, directing players in the middle of the park.",
                        "The most effective hit up takers, these middles are characterised by their strength and big engines.",
                        "Making up the rest of the middle, these players share the hit up and tackling duties."
                    ])

# Export cluster data to JSON
with open('nrl_cluster_data.json', 'w') as f:
    json.dump(cluster_data_export, f, indent=4)
print("\nExported cluster data to nrl_cluster_data.json")

# Export cluster data to JS (for dynamic loading without CORS issues)
with open('nrl_cluster_data.js', 'w') as f:
    f.write(f"const clusterData = {json.dumps(cluster_data_export, indent=4)};")
print("Exported cluster data to nrl_cluster_data.js")


