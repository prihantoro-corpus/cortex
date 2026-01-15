import re

# ============================================================
# Default exception lexicon for 'e' (can be overridden by user)
# ============================================================
DEFAULT_LEX_E = {
    "emas": "əmas",
    "merah": "mɛrah",
    "berak": "bɛrak",
    "enak": "enak",
    "empat": "əmpat",
    "enam": "ənam",
}

VOWELS = "aiueoəɛɔ"

# ============================================================
# Phonological handlers
# ============================================================
def handle_o(text):
    # o before consonant -> ɔ
    text = re.sub(r"o(?=[bcdfghjklmnpqrstvwxyz])", "ɔ", text)
    # o before vowel -> o
    text = re.sub(r"o(?=[aiueo])", "o", text)
    # o at end of word -> o
    text = re.sub(r"o\b", "o", text)
    return text


def handle_e_word(word, lex_e=None):
    if lex_e is None:
        lex_e = DEFAULT_LEX_E
        
    # Exception lexicon first
    if word in lex_e:
        return lex_e[word]

    w = word

    # Common prefixes -> schwa
    w = re.sub(r"^(be|me|pe|ke|se|te)(?=[bcdfghjklmnpqrstvwxyz])",
               lambda m: m.group(1)[0] + "ə", w)

    # Heavy/expressive tendency -> ɛ
    w = re.sub(r"e(?=[rktp])", "ɛ", w)

    # Word-initial e + consonant -> e
    w = re.sub(r"^e(?=[bcdfghjklmnpqrstvwxyz])", "e", w)

    # Remaining e -> schwa
    w = re.sub(r"e", "ə", w)

    return w


# ============================================================
# Core G2P rules (applied after e/o handling)
# ============================================================
RULES = [
    # Diphthongs
    (r"ai", "ai̯"),
    (r"au", "au̯"),
    (r"oi", "oi̯"),

    # Digraphs
    (r"ng", "ŋ"),
    (r"ny", "ɲ"),
    (r"sy", "ʃ"),
    (r"kh", "x"),
    (r"dz", "dz"),

    # Letters
    (r"c", "tʃ"),
    (r"j", "dʒ"),
    (r"y", "j"),
    (r"x", "ks"),
    (r"q", "k"),

    # Consonants (mostly identical)
    (r"b", "b"),
    (r"d", "d"),
    (r"f", "f"),
    (r"g", "g"),
    (r"h", "h"),
    (r"k", "k"),
    (r"l", "l"),
    (r"m", "m"),
    (r"n", "n"),
    (r"p", "p"),
    (r"r", "r"),
    (r"s", "s"),
    (r"t", "t"),
    (r"v", "v"),
    (r"w", "w"),
    (r"z", "z"),
]


# ============================================================
# Final /k/ -> ʔ
# ============================================================
def handle_final_k(ipa):
    return re.sub(r"k\b", "ʔ", ipa)


# ============================================================
# Onset-maximising, coda-aware syllabifier
# ============================================================
def syllabify_onset_max(word):
    """
    Onset-maximising heuristic for Indonesian:
    - V.V -> split
    - VCV -> split before C (C goes to onset)
    - VCCV -> split C.C
    """
    chars = list(word)
    syllables = []
    current = ""
    i = 0

    while i < len(chars):
        c = chars[i]
        current += c

        if c in VOWELS:
            if i + 1 < len(chars):
                nxt = chars[i + 1]

                # V V -> boundary
                if nxt in VOWELS:
                    syllables.append(current)
                    current = ""

                # V C V -> boundary before C
                elif nxt not in VOWELS and i + 2 < len(chars) and chars[i + 2] in VOWELS:
                    syllables.append(current)
                    current = ""

                # V C C V -> split C.C
                elif (
                    nxt not in VOWELS
                    and i + 2 < len(chars)
                    and chars[i + 2] not in VOWELS
                    and i + 3 < len(chars)
                    and chars[i + 3] in VOWELS
                ):
                    current += nxt
                    syllables.append(current)
                    current = ""
                    i += 1  # consume extra consonant

        i += 1

    if current:
        syllables.append(current)

    return ".".join(syllables)

def get_ipa_transcription(token, lex_e=None, syllabify=True):
    """
    Main function to get IPA transcription for an Indonesian token.
    """
    w = token.lower().strip()
    
    # Simple xml skip check
    if w.startswith("<") or re.search(r"[<>{}=\\/]", w):
        return token

    w = handle_e_word(w, lex_e)
    w = handle_o(w)

    ipa = w
    for pattern, repl in RULES:
        ipa = re.sub(pattern, repl, ipa)

    ipa = handle_final_k(ipa)

    if syllabify:
        ipa = syllabify_onset_max(ipa)

    return ipa
