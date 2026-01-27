import os

# Local corpora directory
CORPORA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'corpora')
TAGSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tagset')

# Metadata mapping for known corpora (Display Name -> Relative Path from CORPORA_DIR)
KNOWN_CORPORA_MAP = {
    "ID-BPPT (XML Tagged)": "indonesian/ID-BPPT-tagged.xml",
    "EN-BPPT (XML Tagged)": "english/EN-BPPT-tagged.xml",
    "Brown 50% Only (XML EN TAGGED)": "english/BrownCorpus.xml",
    "KOSLAT-ID (XML Tagged)": "indonesian/KOSLAT-full.xml",
}

# Alias for backward compatibility
BUILT_IN_CORPORA = KNOWN_CORPORA_MAP

def get_available_corpora():
    """
    Returns a dictionary of Available Corpus Name -> Filename.
    Scans CORPORA_DIR and maps to known display names where possible.
    """
    available = {}
    if not os.path.exists(CORPORA_DIR):
        return {}
        
    # Reverse map for easy lookup: filename -> nice name
    filename_to_name = {v: k for k, v in KNOWN_CORPORA_MAP.items()}
    
    # Recursive walk
    for root, dirs, files in os.walk(CORPORA_DIR):
        for f in files:
            if f.lower().endswith(('.xml', '.txt', '.csv', '.xlsx')):
                full_path = os.path.join(root, f)
                
                # Get relative path from CORPORA_DIR
                rel_path = os.path.relpath(full_path, CORPORA_DIR)
                
                # If key known by filename, map it
                # We check simply if the filename (basename) matches the known map
                basename = os.path.basename(f)
                
                if basename in filename_to_name:
                    display_name = filename_to_name[basename]
                    available[display_name] = rel_path
                else:
                    # Use relative path as display name for nested files, or just filename if root
                    if root == CORPORA_DIR:
                         available[f] = f
                    else:
                         # e.g. "indonesian/sample.txt"
                         # Normalize separators for display
                         display_rel = rel_path.replace(os.path.sep, '/')
                         available[display_rel] = rel_path

    return available



BUILT_IN_CORPUS_DETAILS = {
    "ID-BPPT (XML Tagged)": 
        """
        The **ID-BPPT Corpus** is a tagged Indonesian corpus (POS/Lemma). 
        <br><br>
        **Source:** BPPT (Badan Pengkajian dan Penerapan Teknologi).
        """,
    "EN-BPPT (XML Tagged)":
        """
        The **EN-BPPT Corpus** is a tagged English corpus (POS/Lemma) used for parallel experiments or monolingual analysis.
        <br><br>
        **Source:** BPPT.
        """,

    "Brown 50% Only (XML EN TAGGED)":
        """
        A 50% subsample of the Brown Corpus, the first million-word electronic corpus of English. This sample is provided in a **TreeTagger-style XML format** containing token, POS, and lemma.
        <br><br>
        **Source/Citation:** Francis, W. N., & Kučera, H. (1979). **Brown Corpus Manual: Standard Corpus of Present-Day Edited American English for Use with Digital Computers.** Brown University.
        """,
    "KOSLAT-ID (XML Tagged)":
        """
        KOSLAT-ID v.1.0 is the first narrative-annotated corpus of reviews of healthcare facilities in Indonesia. It is provided in a **tagged XML format** (token, POS, lemma).
        <br><br>
        **Source/Citation:** Prihantoro., Yuliawati, S., Ekawati, D., & Rachmat, A. (2026-in press). **KOSLAT-ID v.1.0: The first narrative-annotated corpus of reviews of healthcare facilities in Indonesia.** [Corpora, 21(1), xx–xx.](https://www.prihantoro.com)
        """,

}

# Language codes for Stanza integration
STANZA_LANG_MAP = {
    "English": "en",
    "Indonesian": "id",
    "Japanese": "ja",
    "Chinese": "zh",
    "Arabic": "ar",
    "Korean": "ko"
}
