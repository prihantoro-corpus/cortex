import pandas as pd
from core.visualiser.styles import POS_COLOR_MAP

def create_word_cloud(freq_data, is_tagged_mode):
    """Generates a word cloud from frequency data with conditional POS coloring."""
    
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None
        
    # Filter out multi-word units for visualization stability
    single_word_freq_data = freq_data[~freq_data['token'].str.contains(' ')].copy()
    if single_word_freq_data.empty:
        return None 

    word_freq_dict = single_word_freq_data.set_index('token')['frequency'].to_dict()
    word_to_pos = single_word_freq_data.set_index('token').get('pos', pd.Series('O')).to_dict()
    
    stopwords = set(["the", "of", "to", "and", "in", "that", "is", "a", "for", "on", "it", "with", "as", "by", "this", "be", "are", "have", "not", "will", "i", "we", "you"])
    
    wc = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap='viridis', 
        stopwords=stopwords,
        min_font_size=10
    )
    
    try:
        wordcloud = wc.generate_from_frequencies(word_freq_dict)
    except ValueError:
        return None 

    if is_tagged_mode:
        def final_color_func(word, *args, **kwargs):
            pos_tag = word_to_pos.get(word, 'O')
            pos_code = pos_tag[0].upper() if pos_tag and len(pos_tag) > 0 else 'O'
            if pos_code not in POS_COLOR_MAP:
                pos_code = 'O'
            return POS_COLOR_MAP.get(pos_code, POS_COLOR_MAP['O'])

        wordcloud = wordcloud.recolor(color_func=final_color_func)
        
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    return fig

def generate_wordcloud(freq_dict, title="", color_scheme='viridis'):
    """
    Generates a word cloud from a dictionary of word:score.
    
    Args:
        freq_dict (dict): Dictionary mapping words to scores/frequencies.
        title (str): Title for the plot.
        color_scheme (str): Matplotlib colormap name.
        
    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    if not freq_dict:
        return None
        
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except ImportError:
        return None
        
    wc = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap=color_scheme,
        min_font_size=10
    )
    
    try:
        wordcloud = wc.generate_from_frequencies(freq_dict)
    except Exception as e:
        return None
        
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    if title:
        ax.set_title(title, color='white')
    plt.tight_layout(pad=0)
    
    return fig
