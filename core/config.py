import os

# Local corpora directory
CORPORA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'corpora')
TAGSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tagset')

# Metadata mapping for known corpora (Display Name -> Filename)
KNOWN_CORPORA_MAP = {
    "ID-BPPT (XML Tagged)": "ID-BPPT-tagged.xml",
    "EN-BPPT (XML Tagged)": "EN-BPPT-tagged.xml",
    "Europarl 1M Only": "europarl_en-1M-only.txt",
    "Brown 50% Only (XML EN TAGGED)" : "BrownCorpus.xml",
    "KOSLAT-ID (XML Tagged)": "KOSLAT-full.xml",
    "DICO-JALF V1 (XML Tagged)": "DICO-JALF v1-raw.xml",
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
        
    try:
        files = [f for f in os.listdir(CORPORA_DIR) if os.path.isfile(os.path.join(CORPORA_DIR, f)) and not f.startswith('.')]
    except Exception:
        return {}
        
    # Reverse map for easy lookup: filename -> nice name
    filename_to_name = {v: k for k, v in KNOWN_CORPORA_MAP.items()}
    
    for f in files:
        if f.lower().endswith(('.xml', '.txt', '.csv', '.xlsx')):
            if f in filename_to_name:
                display_name = filename_to_name[f]
                available[display_name] = f
            else:
                available[f] = f
                
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
    "Europarl 1M Only": 
        """
        The Europarl Corpus is a large collection of European Parliament proceedings. This sample contains approximately 1 million tokens of English text. 
        It is provided as a **verticalised T/P/L file** for demonstration.
        <br><br>
        **Source/Citation:** Koehn, Philipp. (2005). Europarl: A Parallel Corpus for Statistical Machine Translation. In: **Proceedings of the Tenth Machine Translation Summit (MT Summit X)**, Phuket, Thailand.
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
    "DICO-JALF V1 (XML Tagged)":
        """
        This is DICO-JALF.
        <br><br>
        **Source/Citation:** Prihantoro, P., Ishikawa, S., Liu, T., Fadli, Z. A., Rini, E. I. H. A. N., & Kepirianto, C. (2025). DICO-JALF v.1.0: Diponegoro Corpus of Japanese Learners as a Foreign Language in Indonesia with AI Error Annotation and Human Supervision. Jurnal Arbitrer, 12(3), 274–288. https://doi.org/10.25077/ar.12.3.274-288.2025 
        """
}
