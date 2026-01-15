import nltk
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import duckdb

# Optional BERTopic imports (will be checked at runtime)
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# --- PRE-DEFINED TOPIC KEYWORDS ---
# Lightweight Knowledge Base for Classification without Massive Models
TOPIC_KEYWORDS = {
    "Sport": [
        "match", "game", "score", "player", "team", "tournament", "league", "cup", 
        "football", "basketball", "tennis", "golf", "win", "lose", "champion", 
        "stadium", "goal", "coach", "athlete", "medal", "olympic"
    ],
    "Religion": [
        "god", "church", "pray", "faith", "belief", "holy", "bible", "quran", 
        "mosque", "temple", "divine", "soul", "spirit", "worship", "sacred", 
        "priest", "monk", "prophet", "religion", "religious", "spiritual"
    ],
    "Science": [
        "study", "research", "data", "experiment", "theory", "method", "analysis", 
        "result", "scientist", "laboratory", "physics", "chemistry", "biology", 
        "observation", "evidence", "hypothesis", "technology", "scientific", "journal"
    ],
    "Politics": [
        "government", "policy", "law", "election", "vote", "party", "minister", 
        "president", "parliament", "senate", "campaign", "candidate", "democracy", 
        "political", "state", "republic", "tax", "regulation", "diplomacy"
    ],
    "Economy": [
        "market", "price", "money", "business", "company", "trade", "finance", 
        "cost", "profit", "loss", "bank", "stock", "investment", "growth", 
        "economic", "industry", "supply", "demand", "inflation", "currency", "revenue"
    ],
    "Technology": [
        "computer", "software", "internet", "device", "system", "app", "mobile", 
        "digital", "code", "algorithm", "network", "server", "user", "interface", 
        "hardware", "processor", "ai", "robot", "tech", "online"
    ]
}

def ensure_nltk_resources():
    """Checks and downloads necessary NLTK data (vader_lexicon)."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

def classify_sentiment_vader(texts):
    """
    Classifies a list of texts into Positive, Negative, Neutral using VADER.
    Returns a list of strings.
    """
    ensure_nltk_resources()
    sia = SentimentIntensityAnalyzer()
    
    results = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            results.append("Neutral")
            continue
            
        # VADER scoring
        scores = sia.polarity_scores(text)
        compound = scores['compound']
        
        # Thresholds: >= 0.05 Pos, <= -0.05 Neg, else Neu
        if compound >= 0.05:
            results.append("Positive")
        elif compound <= -0.05:
            results.append("Negative")
        else:
            results.append("Neutral")
            
    return results

def classify_topics_keyword_weighted(texts):
    """
    Classifies texts into pre-defined topics using TF-IDF weighted keyword matching.
    Returns: (topic_assignments, topic_info_dict)
    
    topic_info_dict format:
    {
        'Sport': {'keywords': [...], 'count': N, 'label': 'Sport'},
        'Science': {...},
        ...
    }
    """
    if not texts: return [], {}
    
    # 1. Prepare Seed Keywords as "Documents"
    topic_labels = list(TOPIC_KEYWORDS.keys())
    seed_docs = [" ".join(TOPIC_KEYWORDS[label]) for label in topic_labels]
    
    # 2. Fit Vectorizer on Seed Keywords AND Target Texts (to build shared vocab)
    # Using a subset of target texts to avoid memory issues if corpus is huge
    sample_size = min(len(texts), 5000)
    sample_texts = texts[:sample_size] if isinstance(texts, list) else texts.head(sample_size).tolist()
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    all_corpus = seed_docs + [t for t in sample_texts if isinstance(t, str)]
    vectorizer.fit(all_corpus)
    
    # 3. Transform Seeds -> Topic Vectors
    seed_vectors = vectorizer.transform(seed_docs) # Shape: (n_topics, n_features)
    
    # 4. Transform Target Texts
    # Process in batches if needed, but for now simple transform
    # Handle non-string inputs gracefully
    clean_texts = [t if isinstance(t, str) else "" for t in texts]
    text_vectors = vectorizer.transform(clean_texts) # Shape: (n_texts, n_features)
    
    # 5. Compute Similarity (Dot Product)
    # Result: (n_texts, n_topics) matrix of scores
    scores = text_vectors.dot(seed_vectors.T)
    
    # 6. Assign Topics
    # Convert sparse matrix to dense array for argmax
    # If using scipy sparse, access directly
    dense_scores = scores.toarray()
    
    topic_results = []
    for row_scores in dense_scores:
        if row_scores.max() == 0:
            topic_results.append("General") # No keywords matched
        else:
            best_idx = row_scores.argmax()
            topic_results.append(topic_labels[best_idx])
    
    # 7. Build topic_info for consistency with BERTopic
    topic_info = {}
    for label in topic_labels + ["General"]:
        count = topic_results.count(label)
        if count > 0:
            topic_info[label] = {
                'keywords': TOPIC_KEYWORDS.get(label, []),
                'count': count,
                'label': label
            }
            
    return topic_results, topic_info

def classify_topics_bertopic(texts, n_topics='auto', min_topic_size=10):
    """
    Uses BERTopic to discover topics and classify texts.
    Returns: (topic_assignments, topic_info_dict)
    
    Args:
        texts: List of text strings to classify
        n_topics: Number of topics ('auto' or integer)
        min_topic_size: Minimum documents per topic
    
    Returns:
        topic_assignments: List of topic labels for each text
        topic_info: Dict with topic metadata {topic_id: {'keywords': [...], 'count': N, 'label': 'Topic_X'}}
    """
    if not BERTOPIC_AVAILABLE:
        raise ImportError(
            "BERTopic is not installed. Install with: pip install bertopic sentence-transformers"
        )
    
    if not texts:
        return [], {}
    
    # Filter out empty texts
    clean_texts = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    
    # Use lightweight sentence transformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=n_topics if n_topics != 'auto' else None,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,  # Faster
        verbose=False
    )
    
    # Fit and transform
    topics, _ = topic_model.fit_transform(clean_texts)
    
    # Extract topic info with keywords
    topic_info = {}
    topic_assignments = []
    
    for topic_id in set(topics):
        if topic_id != -1:  # -1 is outlier topic
            keywords = topic_model.get_topic(topic_id)
            keyword_list = [word for word, score in keywords[:10]]
            
            # Auto-generate descriptive label from top 3 keywords
            if len(keyword_list) >= 3:
                # Capitalize first letter of each word and join with "+"
                auto_label = " + ".join([kw.capitalize() for kw in keyword_list[:3]])
            elif len(keyword_list) >= 1:
                auto_label = " + ".join([kw.capitalize() for kw in keyword_list])
            else:
                auto_label = f'Topic_{topic_id}'
            
            topic_info[topic_id] = {
                'keywords': keyword_list,
                'count': topics.count(topic_id),
                'label': auto_label
            }
    
    # Convert topic IDs to labels
    for topic_id in topics:
        if topic_id == -1:
            topic_assignments.append('Outlier')
        else:
            topic_assignments.append(topic_info[topic_id]['label'])
    
    # Add outlier info if present
    if -1 in topics:
        topic_info['Outlier'] = {
            'keywords': [],
            'count': topics.count(-1),
            'label': 'Outlier'
        }
    
    return topic_assignments, topic_info

def apply_classification_to_db(db_path, id_list, topics, sentiments):
    """
    Updates the DuckDB corpus table with new 'topic' and 'sentiment' columns.
    """
    con = duckdb.connect(db_path)
    
    # 1. Create columns if they don't exist
    try:
        con.execute("ALTER TABLE corpus ADD COLUMN topic VARCHAR")
    except: pass # Already exists
    
    try:
        con.execute("ALTER TABLE corpus ADD COLUMN sentiment VARCHAR")
    except: pass
    
    # 2. Update rows
    # Bulk update is tricky in SQL without temp table.
    # Approach: Create a temporary DataFrame -> Table, then Update Join.
    
    update_df = pd.DataFrame({
        'id': id_list,
        'new_topic': topics,
        'new_sentiment': sentiments
    })
    
    con.register('update_df', update_df)
    
    con.execute("""
        UPDATE corpus
        SET topic = update_df.new_topic,
            sentiment = update_df.new_sentiment
        FROM update_df
        WHERE corpus.id = update_df.id
    """)
    
    con.unregister('update_df')
    con.close()
    return True

def apply_classification_by_sentence(db_path, filenames, sent_ids, topics=None, sentiments=None):
    """
    Updates the DuckDB corpus table with new 'topic' and 'sentiment' columns
    assigned at the sentence level. Supports updating one or both.
    """
    con = duckdb.connect(db_path)
    
    # 1. Create columns if they don't exist
    if topics is not None:
        try: con.execute("ALTER TABLE corpus ADD COLUMN topic VARCHAR")
        except: pass
    
    if sentiments is not None:
        try: con.execute("ALTER TABLE corpus ADD COLUMN sentiment VARCHAR")
        except: pass
    
    # 2. Update rows
    data = {'filename': filenames, 'sent_id': sent_ids}
    set_clauses = []
    
    if topics is not None:
        data['new_topic'] = topics
        set_clauses.append("topic = update_df.new_topic")
    
    if sentiments is not None:
        data['new_sentiment'] = sentiments
        set_clauses.append("sentiment = update_df.new_sentiment")
        
    if not set_clauses:
        con.close()
        return True

    update_df = pd.DataFrame(data)
    con.register('update_df', update_df)
    
    con.execute(f"""
        UPDATE corpus
        SET {', '.join(set_clauses)}
        FROM update_df
        WHERE corpus.filename = update_df.filename 
          AND corpus.sent_id = update_df.sent_id
    """)
    
    con.unregister('update_df')

    # 3. Add Indices for Performance
    if topics is not None:
        try: con.execute("CREATE INDEX IF NOT EXISTS idx_topic ON corpus(topic)")
        except: pass
        
    if sentiments is not None:
        try: con.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON corpus(sentiment)")
        except: pass

    con.close()
    return True
