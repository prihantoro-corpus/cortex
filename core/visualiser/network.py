import os
import tempfile
import pandas as pd
from core.visualiser.styles import POS_COLOR_MAP

def create_pyvis_graph(target_word, coll_df):
    try:
        from pyvis.network import Network
    except ImportError:
        return ""

    net = Network(height="400px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='local')
    if coll_df.empty: return ""
    max_ll = coll_df['LL'].max()
    min_ll = coll_df['LL'].min()
    ll_range = max_ll - min_ll
    
    net.set_options("""
    var options = {
      "nodes": {"borderWidth": 2, "size": 15, "font": {"size": 30}},
      "edges": {"width": 5, "smooth": {"type": "dynamic"}},
      "physics": {"barnesHut": {"gravitationalConstant": -10000, "centralGravity": 0.3, "springLength": 95, "springConstant": 0.04, "damping": 0.9, "avoidOverlap": 0.5}, "minVelocity": 0.75}
    }
    """)
    
    net.add_node(target_word, label=target_word, size=40, color='#FFFF00', title=f"Target: {target_word}", x=0, y=0, fixed=True, font={'color': 'black'})
    
    LEFT_BIAS = -500; RIGHT_BIAS = 500
    all_directions = coll_df['Direction'].unique()
    if 'R' not in all_directions and 'L' in all_directions: RIGHT_BIAS = -500
    elif 'L' not in all_directions and 'R' in all_directions: LEFT_BIAS = 500

    for index, row in coll_df.iterrows():
        collocate = row['Collocate']
        ll_score = row['LL']
        observed = row['Observed']
        pos_tag = row['POS']
        direction = row.get('Direction', 'R') 
        obs_l = row.get('Obs_L', 0)
        obs_r = row.get('Obs_R', 0)
        x_position = LEFT_BIAS if direction in ('L', 'B') else RIGHT_BIAS

        pos_code = pos_tag[0].upper() if pos_tag and len(pos_tag) > 0 else 'O'
        if pos_tag.startswith('##'): pos_code = '#'
        elif pos_code not in ['N', 'V', 'J', 'R']: pos_code = 'O'
        
        color = POS_COLOR_MAP.get(pos_code, POS_COLOR_MAP['O'])
        
        node_size = 25
        if ll_range > 0:
            normalized_ll = (ll_score - min_ll) / ll_range
            node_size = 15 + normalized_ll * 25 
            
        tooltip_title = (
            f"POS: {row['POS']}\n"
            f"Obs: {observed} (Left: {obs_l}, Right: {obs_r})\n"
            f"LL: {ll_score:.2f}\n"
            f"Dominant Direction: {direction}"
        )

        net.add_node(collocate, label=collocate, size=node_size, color=color, title=tooltip_title, x=x_position)
        net.add_edge(target_word, collocate, value=ll_score, width=5, title=f"LL: {ll_score:.2f}")

    html_content = ""; temp_path = None
    try:
        temp_filename = "pyvis_graph.html"
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, temp_filename)
        net.write_html(temp_path, notebook=False)
        with open(temp_path, 'r', encoding='utf-8') as f: html_content = f.read()
    finally:
        if temp_path and os.path.exists(temp_path): os.remove(temp_path)

    return html_content
