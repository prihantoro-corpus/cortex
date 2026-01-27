from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POS_COLOR_MAP = {
    'N': '#1f77b4', # Noun - Blue
    'V': '#ff7f0e', # Verb - Orange
    'J': '#2ca02c', # Adj - Green
    'R': '#d62728', # Adv - Red
    'O': '#7f7f7f'  # Other - Gray
}

def create_word_cloud(data, use_pos=False):
    """
    Unified entry point for word cloud generation.
    Handles Both DataFrames and dictionaries without Truth Value ambiguity.
    """
    if data is None:
        return None
        
    try:
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return None
            
            # Determine frequency dictionary
            # Expects columns like 'token' and 'frequency'
            word_freq = dict(zip(data['token'], data['frequency']))
            
            if use_pos and 'pos' in data.columns:
                word_to_pos = dict(zip(data['token'], data['pos']))
                return generate_tagged_wordcloud(word_freq, word_to_pos)
            else:
                return generate_wordcloud(word_freq)
                
        elif isinstance(data, dict):
            if not data:
                return None
            return generate_wordcloud(data)
    except Exception:
        return None
        
    return None

def generate_tagged_wordcloud(word_freq_dict, word_to_pos):
    """
    Generates a POS-colored word cloud.
    """
    if word_freq_dict is None or len(word_freq_dict) == 0:
        return None

    wc = WordCloud(
        width=800,
        height=400,
        background_color='black',
        min_font_size=10
    )
    
    try:
        wordcloud = wc.generate_from_frequencies(word_freq_dict)
    except Exception:
        return None 

    def final_color_func(word, *args, **kwargs):
        pos_tag = word_to_pos.get(word, 'O')
        pos_code = pos_tag[0].upper() if pos_tag and len(pos_tag) > 0 else 'O'
        if pos_code not in POS_COLOR_MAP:
            pos_code = 'O'
        return POS_COLOR_MAP.get(pos_code, POS_COLOR_MAP['O'])

    wordcloud = wordcloud.recolor(color_func=final_color_func)
        
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    fig.tight_layout(pad=0)
    
    return fig

def generate_wordcloud(freq_dict, title="", color_scheme='viridis', width=400, height=200):
    """
    Generates a word cloud from a dictionary or DataFrame.
    """
    if freq_dict is None:
        return None
        
    # Standardize to dict
    if isinstance(freq_dict, pd.DataFrame):
        if freq_dict.empty: return None
        freq_dict = dict(zip(freq_dict['token'], freq_dict['frequency']))

    # Final check before passing to WordCloud
    if not isinstance(freq_dict, dict) or len(freq_dict) == 0:
        return None
        
    wc = WordCloud(
        width=width * 2,
        height=height * 2,
        background_color='black',
        colormap=color_scheme,
        min_font_size=10
    )
    
    try:
        wordcloud = wc.generate_from_frequencies(freq_dict)
    except Exception:
        return None
        
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    if title:
        ax.set_title(title, color='white')
    fig.tight_layout(pad=0)
    
    return fig
