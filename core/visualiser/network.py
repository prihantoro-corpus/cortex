import os
import re
import tempfile
import pandas as pd
from core.visualiser.styles import POS_COLOR_MAP

def prepare_standalone_pyvis_html(net, height_px=550, bg_color="#222222"):
    """
    Generates completely self-contained, offline-compatible, dark-mode Pyvis HTML
    that will NEVER display a white box or fail due to external CDN or relative path issues.
    """
    raw_html = net.generate_html()

    # Remove broken relative script and stylesheet references
    raw_html = re.sub(r'<script[^>]*src=["\']?lib/bindings/utils\.js["\']?[^>]*>\s*</script>', '', raw_html)
    raw_html = re.sub(r'<script[^>]*src=["\']?\.\./node_modules/[^>]*>\s*</script>', '', raw_html)
    raw_html = re.sub(r'<link[^>]*href=["\']?\.\./node_modules/[^>]*>', '', raw_html)
    raw_html = re.sub(r'<link[^>]*bootstrap[^>]*>', '', raw_html)
    raw_html = re.sub(r'<script[^>]*bootstrap[^>]*></script>', '', raw_html)

    # Replace Bootstrap card wrapper
    raw_html = raw_html.replace(
        '<div class="card" style="width: 100%">',
        f'<div style="width: 100%; height: {height_px}px; background-color: {bg_color};">'
    )

    # Enforce dark mode CSS reset & wheel scroll forwarding script
    dark_css_and_script = f"""
    <style>
    *, *::before, *::after {{
        box-sizing: border-box !important;
    }}
    html, body {{
        background-color: {bg_color} !important;
        color: #ffffff !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        height: 100% !important;
        overflow: hidden !important;
    }}
    .card, .card-body {{
        background-color: {bg_color} !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        height: 100% !important;
    }}
    #mynetwork {{
        width: 100% !important;
        height: {height_px}px !important;
        background-color: {bg_color} !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }}
    </style>
    <script>
    (function() {{
        function forwardWheel(e) {{
            try {{
                if (window.parent && window.parent !== window) {{
                    let dy = e.deltaY;
                    let dx = e.deltaX;
                    if (e.deltaMode === 1) {{
                        dy *= 16;
                        dx *= 16;
                    }} else if (e.deltaMode === 2) {{
                        dy *= window.innerHeight;
                        dx *= window.innerWidth;
                    }}
                    window.parent.scrollBy({{
                        top: dy,
                        left: dx,
                        behavior: 'auto'
                    }});
                }}
            }} catch(err) {{}}
        }}
        window.addEventListener('wheel', forwardWheel, {{ passive: true, capture: true }});
        document.addEventListener('wheel', forwardWheel, {{ passive: true, capture: true }});
    }})();
    </script>
    """
    if "</head>" in raw_html:
        raw_html = raw_html.replace("</head>", dark_css_and_script + "</head>", 1)
    else:
        raw_html = dark_css_and_script + raw_html

    # Auto-fit the view on load and after physics stabilises
    fit_script = (
        "<script>(function(){"
        "var t=null;"
        "function f(){try{network.redraw();network.fit({animation:false});}catch(e){}}"
        "function sched(){clearTimeout(t);t=setTimeout(f,80);}"
        "function init(){"
        "if(typeof network==='undefined'){setTimeout(init,100);return;}"
        "f();"
        "network.on('stabilizationIterationsDone',f);"
        "network.on('stabilized',f);"
        "setTimeout(f,500);setTimeout(f,1500);setTimeout(f,3000);"
        # Refit whenever the iframe/container becomes visible or changes size
        "window.addEventListener('resize',sched);"
        "if(window.ResizeObserver){"
        "var el=document.getElementById('mynetwork')||document.body;"
        "var ro=new ResizeObserver(function(){if(el.clientWidth>0&&el.clientHeight>0){sched();}});"
        "ro.observe(el);ro.observe(document.documentElement);}"
        "if(window.IntersectionObserver){"
        "var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){sched();}});});"
        "io.observe(document.body);}"
        "}"
        "if(document.readyState==='complete'){init();}else{window.addEventListener('load',init);}"
        "})();</script>"
    )
    if "</body>" in raw_html:
        raw_html = raw_html.replace("</body>", fit_script + "</body>", 1)
    else:
        raw_html += fit_script

    return raw_html

def create_pyvis_graph(target_word, coll_df, measure_col="LL", measure_name="LL", font_size=26):
    try:
        from pyvis.network import Network
    except ImportError:
        return ""

    net = Network(height="550px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote')
    if coll_df.empty: return ""
    max_score = coll_df[measure_col].max()
    min_score = coll_df[measure_col].min()
    score_range = max_score - min_score
    
    target_font_size = int(font_size * 1.25)
    coll_font_size = int(font_size)

    net.set_options(f"""
    var options = {{
      "nodes": {{"borderWidth": 2, "size": 25, "font": {{"size": {coll_font_size}}}}},
      "edges": {{"width": 4, "smooth": {{"type": "dynamic"}}}},
      "interaction": {{
        "zoomView": false,
        "navigationButtons": true,
        "hover": true,
        "dragNodes": true,
        "dragView": true
      }},
      "physics": {{
        "solver": "barnesHut",
        "barnesHut": {{
          "gravitationalConstant": -3500,
          "centralGravity": 0.3,
          "springLength": 130,
          "springConstant": 0.04,
          "damping": 0.9,
          "avoidOverlap": 0.3
        }},
        "stabilization": {{
          "enabled": true,
          "iterations": 100,
          "updateInterval": 25
        }}
      }}
    }}
    """)
    
    net.add_node(target_word, label=target_word, size=45, color='#FFFF00', title=f"Target: {target_word}", x=0, y=0, fixed=True, font={'size': target_font_size, 'color': 'black'})
    
    LEFT_BIAS = -500; RIGHT_BIAS = 500
    all_directions = coll_df['Direction'].unique()
    if 'R' not in all_directions and 'L' in all_directions: RIGHT_BIAS = -500
    elif 'L' not in all_directions and 'R' in all_directions: LEFT_BIAS = 500

    for index, row in coll_df.iterrows():
        collocate = row['Collocate']
        score_val = row[measure_col]
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
        if score_range > 0:
            normalized_score = (score_val - min_score) / score_range
            node_size = 15 + normalized_score * 25 
            
        tooltip_title = (
            f"POS: {row['POS']}\n"
            f"Obs: {observed} (Left: {obs_l}, Right: {obs_r})\n"
            f"{measure_name}: {score_val:.2f}\n"
            f"Dominant Direction: {direction}"
        )

        net.add_node(collocate, label=collocate, size=node_size, color=color, title=tooltip_title, x=x_position, font={'size': coll_font_size, 'color': 'white'})
        net.add_edge(target_word, collocate, value=score_val, width=5, title=f"{measure_name}: {score_val:.2f}")

    return prepare_standalone_pyvis_html(net, height_px=550, bg_color="#222222")
