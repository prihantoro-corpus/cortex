import streamlit as st
from ui_streamlit.state_manager import get_state, set_state

GUIDELINES = {
    "Overview": {
        "XML Structure": """
            ### 📖 XML Structure Guide
            * See the structure of the corpus from XML annotations (if any). Useful for searching.
            * (Note: XML structure will update if you choose to annotate).
        """,
        "Sub-corpus Stats": """
            ### 📖 Sub-Corpus Statistics Guide
            * See the corpus structure in a chart and table. Useful for searching.
            * (Note: Sub-corpus stats will update if you choose to annotate).
        """,
        "Frequency List": """
            ### 📖 Frequency List Guide
            * See the frequency list and download it.
        """,
        "Unique POS Tags": """
            ### 📖 Unique POS Tags Guide
            * See all POS tags used in the corpus and their definitions (if any).
        """,
        "Word Cloud": """
            ### 📖 Word Cloud Guide
            * See a word cloud generated from the corpus.
        """,
        "Metadata Annotation": """
            ### 📖 Metadata Annotation Guide
            * Annotate metadata (e.g., gender, sex, education, etc.).
            * **File Level:** Annotate each text. Add attributes (e.g., text_type, author_sex, etc.). In the table, write the values for each file (e.g., male or female for author_sex), then apply metadata annotation.
            * **Segmental Level:** Annotate at a level lower than the text (parts of the text). Select a file, then choose to annotate word by word or by sentence.
               * If by word, drag from the word grid to select.
               * If by sentence, use the checkmarks.
               * Then, in "Annotate Selection", write the attribute and values.
        """,
        "🏷️ Sentiment & Topic Analysis": """
            ### 📖 Sentiment & Topic Analysis Guide
            * **Topic Analysis:** (Under construction)
            * **Sentiment Analysis:** (Note: Only works for English corpus)
               * Choose TF/IDF or BERTopic > configure Sentiment > run labelling.
               * **Note:** Check the Sub-corpus Stats to see the XML changes after running!
        """,
        "🏷️ Named Entity Recognition (NER)": """
            ### 📖 Named Entity Recognition (NER) Guide
            *(Note: Only works for English corpus)*
            * **Entity Classification:** Automatically extracts and labels proper nouns into categories (e.g., Person, Organization, Location, Date).
            * **NER extraction method:** Choose between Dependency-based or Regex (you need to write the regex manually).
            * **Spacy Pipeline model:** Choose your preferred model.
            * **RUN:** Annotates the corpus with XML tags. (**Note:** Check the Sub-corpus Stats to see the XML changes after running!)
        """,
        "🔱 Dependency Parsing": """
            ### 📖 Dependency Parsing Guide
            *(Note: Only works for English corpus)*
            * Annotate with dependency parsing (e.g., nsubj for noun subject, agent, relcl for relative clause).
            * **Spacy pipeline model:** Choose your preferred model > Run dependency parsing.
        """,
        "📖 Reading Ease": """
            ### 📖 Reading Ease Guide
            * Evaluates texts using standard indexes (Flesch Reading Ease, Flesch-Kincaid Grade Level, LIX, RIX).
            * **Chunk Readability:** Useful if 1 text has multiple difficulties segments.
               * **By segment:** e.g., 10, 100, 1000.
               * **By subcorpora:** Choose attributes from the radio button.
            * **Readability annotation:** Annotate by sentence (if chosen, chunk will automatically be disabled) or by chunk > annotate.
        """,
        "📖 Lexical Complexity": """
            ### 📖 Lexical Complexity Guide
            *(Generic works for all languages; Lu's LCA only for English corpus)*
            * **Generic:** Automatically computed.
            * **Lu's LCA:** Choose wordlist (for lexical sophistication e.g., NGSL, AWL etc.) > Choose analysis level.
               * Default is by file.
               * If by sub-corpus, choose sub-corpus then choose attribute (e.g., choosing sentiment will display 3 segments: positive, negative, and neutral complexity in the table).
        """
    },
    "Concordance": """
        ### 📖 Concordance (KWIC) User Guide
        
        A concordance displays occurrences of a query word (the **node word**) in its immediate context. Here is how to use the Concordance module:
        
        #### 1. Node Word Input
        You can search for direct words, wildcards, lemmas, or specific grammatical patterns:
        * **Direct Token**: Type a literal word (e.g., `run` or `beautiful`).
        * **Wildcard (`*`)**: Use `*` to match prefix, suffix, or parts of words.
          * Example: `run*` matches *run*, *runs*, *running*, *runner*.
          * Example: `*ing` matches *sing*, *playing*, *jumping*.
        * **Union Search (`|`)**: Use the pipe symbol `|` to search for multiple strings simultaneously (OR logic).
          * Example: `can|could|will|would` matches any of those literal words.
          * Example: `[be|have|do]` matches all inflected forms of those base lemmas.
          * Example: `can|could_MD` matches those specific words, restricted to the `MD` part-of-speech tag.
        * **Lemma Unit**: Wrap the base form in square brackets to match all inflected forms.
          * Example: `[run]` matches *run*, *ran*, *running*, *runs*.
        * **POS Restricted Tokens**: Search for words with specific Part-of-Speech tags.
          * Example: `light_V*` matches *light* when used as a verb.
          * Example: `_NN` matches any singular noun.
        * **XML Tags**: Match structural XML tags and token-level metadata (like dependency tags) directly.
          * Example: `<PN type="human">` matches specific XML elements.
          * Example: `<dep_rel="nsubj">` matches token dependency relations (e.g. nominal subjects).
          
        #### 2. Context Window & Max Lines
        * **Context Window**: Adjusts how many words are retrieved to the left and right of the node word (e.g., 5 words on each side).
        * **Max Lines**: Limits the maximum number of matching rows to return (e.g., 100 lines), ensuring fast responsiveness.
        
        #### 3. Filtering by Collocate (Regex)
        * Apply a Regular Expression filter to narrow down results to only those where a specific pattern exists in the surrounding context.
        * Example: `\\b(very|extremely)\\b` filters for lines containing either "very" or "extremely" near the node word.
        * Example: `not` filters for lines containing the word "not".
        
        #### 4. Display Checkboxes (POS, Lemma, Metadata)
        * **Show POS**: Appends Part-of-Speech tags to all words in the context windows.
        * **Show Lemma**: Displays the base dictionary form for all words in the context windows.
        * **Show Metadata**: Displays document/corpus metadata (such as Text ID, source, or subcorpus classification) in a column on the left.
        
        #### 5. Query Triggering
        * Once you are satisfied with these controls, click the **Generate Concordance Lines** (or **Generate Concordance**) button to run the query and generate the concordance lines.
        
        #### 6. Clustering by Sub-corpora (Advanced)
        * To cluster the results of concordance lines:
          1. In the **Advanced** tab, first restrict searches using the **XML Restriction Filters** based on sub-corpora present in the corpus.
          2. Generate your initial concordance lines.
          3. Select your categorical metadata filters in the restricted search.
          4. Click **🧩 Cluster Mode** to group and sample results based on the selected sub-corpora.
          * **Example**: You can compare and analyze the presence of concordance lines in different combinations, such as *negative sentiment* vs. *positive sentiment*, or *negative sentiments in the economy section* vs. *negative sentiments in the sports section*.

        #### 7. Interactive Annotation Mode
        * **Annotate Lines**: Toggle **✍️ Annotation Mode** to start annotating the concordance lines. You can tag occurrences with your own invented custom attributes and values.
        * **Save & Resume**: You do not have to complete your annotations all at once. Click **💾 Save Annotation Progress** to save your annotations to a JSON file on your machine.
        * **Upload & Restore**: When you are ready to resume, click **📁 Continue Annotation** and upload your saved JSON file to instantly restore your annotation progress.
        * **Index & Retrieve**: Once you are happy with the labels, click **🏛️ Apply to Session** to index and save these annotations directly into the corpus database. They are then fully retrieved and searchable across other views (like Overview and Restricted Search).

        #### 8. Annotation Tags Guide
        * In the **Search Controls** (or Advanced Settings), you can click the **❓ Annotation Tags Guide** popover to explore the active tags in your corpus.
        * Use the dropdown menu to view definitions and random corpus examples for **Part-of-Speech (POS)** tags, as well as other annotations you might have added from the Overview module (such as **Named Entity Recognition (NER)**, **Dependency Parsing**, or **Sentiment Analysis**).
    """,
    "N-Gram": """
        ### 📖 N-Gram Analysis User Guide
        
        An N-Gram is a contiguous sequence of *N* items (words, lemmas, or POS tags) from a given text. This module helps you find and analyze frequent word patterns in your corpus:
        
        #### 1. N-Gram Size & General Settings
        * **N-Gram Size (N)**: Use the slider to select the length of the sequences (e.g., 2 for Bigrams, 3 for Trigrams, etc.).
        * **Skip Punctuation**: Excludes punctuation marks and special symbols from the sequences.
        * **Output Basis**: Select the default base representation of the n-gram elements:
          * **Token**: Matches raw, literal words (e.g., *went*, *running*).
          * **Lemma**: Matches base dictionary forms (e.g., *go*, *run*).
          * **POS Tag**: Matches part-of-speech categories (e.g., *VBD*, *VBG*).
          * **NER / Dependency Parsing**: If your corpus is annotated, you can select these layers as a basis to retrieve N-Grams of named entities or syntactic dependencies.
        
        #### 2. Advanced Positional Filters
        In **Advanced** mode, you can customize the **Basis** and specify **Filters** for *each position* in the N-gram sequence separately:
        * **Wildcards**: Use `*`, `%`, or `_` to match partial words (e.g. `inter*` matches *international*, *internet*).
        * **POS Tag Suffixes**: Match specific parts of speech by appending `_TAG` (e.g., `_NN` for nouns).
        * **Lemma Override**: Search by lemma in a token-based query by wrapping the lemma in brackets (e.g., `[be]` matches *am*, *is*, *are*, *was*, *were*).
        * **Negation**: Prefix a word with a minus sign (`-`) to exclude it from that position (e.g. `-the` matches any word except *the*).
        
        #### 3. XML Restriction Filters
        * Filter the source data before generating n-grams based on document-level metadata (sub-corpora attributes such as *author*, *publication year*, or *sentiment*).
        
        #### 4. Query Triggering
        * After choosing settings or positioning filters, click **Generate N-Grams** (or **Generate Comparison N-Grams** in Comparison Mode) to run the analysis.
        
        #### 5. Results & Interpretation
        * **Metrics**: Results display the Absolute Frequency, Relative Frequency (per Million Words / PMW), Zipf Scores, and Zipf Law Frequency Bands.
        * **Excel Export**: Download the full list of n-gram patterns as an Excel spreadsheet.
        * **Interpret with AI**: Submit the top patterns to the LLM to get a detailed semantic analysis.
    """,
    "Collocation": """
        ### 📖 Collocation Analysis User Guide
        
        Collocations are pairs or groups of words that co-occur more frequently than would be expected by chance.
        
        #### 1. Search Settings
        * **Node Word**: Input the target word around which to find collocates.
        * **Association Measure**: Select the formula used to calculate collocation strength:
          * **Log-Likelihood (LL)**: Best for high-frequency patterns.
          * **Log-Dice**: Reflects the exclusive co-occurrence (independent of corpus size).
          * **Mutual Information (MI)**: Emphasizes strongly bound, rare terms.
          * **Dice Coefficient**: Evaluates overlap ratio.
        * **Context Window**: Determine the span of words surrounding the node word to search (e.g., 5 words to the left/right).
        * **Show all collocates in concordance**: Check this box to retrieve and display **all** matching concordance lines for each collocate instead of a single sample instance.
        
        #### 2. Advanced Filters & XML Restrictions
        * Apply positional filters (Tokens, Lemmas, and POS) and XML restrictions to analyze collocations in specific sub-corpora.
        * **Case-Insensitive Positive & Negative Filters**:
          * Prefix a term with a minus sign (`-`) to exclude it. For example:
            * Token Filter: `-the` excludes "the".
            * POS Filter: `-JJ` excludes adjectives.
            * Lemma Filter: `-buy` excludes the lemma "buy".
          * Comparisons are completely case-insensitive (e.g., `-jj` works identically to `-JJ`).
        * **Union Syntax & Negation**:
          * Wrap choices in parentheses and separate them with a pipe character, e.g. `(JJ|NN)` matches adjectives or nouns. This applies to token, lemma, and POS filters.
          * Supports negation inside parentheses, e.g. `(-JJ|-NN|-RB)` to exclude adjectives, nouns, and adverbs.
          * Supports negation outside parentheses, e.g. `-(JJ|NN|RB)` or `-(buy|sell)` which dynamically distributes the exclusion to all elements inside.
        
        #### 3. Collocation Patterns (Optional)
        Cluster collocates dynamically using grammar patterns defined line-by-line (`label : pattern`).
        * **Syntax Symbols**:
          * `#` : Represents the **node word**.
          * `<...>` : Represents the **collocate**.
          * `*` : Optional token (0 or 1 words).
          * `+` : Required token (exactly 1 word).
          * `_TAG` : POS tag constraint (e.g. `_VB`).
          * `[lemma]` : Lemma constraint (e.g. `[be]`).
        
        #### 4. Multi-Node Collocation Comparison
        Compare the collocates of up to 5 node words inside the active corpus to identify shared and distinct vocabulary environments.
        
        * **Shared Collocates**: Shows collocates appearing with more than one node word, sorted by Overlap Degree.
        * **Unique Collocates**: Separate lists containing collocates exclusive to only one specific node word.
        * **Combined Strength Formulas**: Choose how the overall association strength is calculated across multiple nodes:
          * **Simple Aggregate**: The raw sum of scores across node words:
            $$\\text{Combined Score} = \\sum_{i=1}^{N} \\text{Score}_i$$
          * **Harmonic Mean**: A strict consensus metric that penalizes uneven association strengths (requires high scores across all nodes to rank high):
            $$\\text{Combined Score} = \\frac{N_{\\text{active}}}{\\sum \\frac{1}{\\text{Score}_i}}$$
          * **Min-Max Normalization**: Scales individual node collocate scores to a $[0, 1]$ range before combining, eliminating scale differences:
            $$\\text{Normalized Score} = \\frac{\\text{Score} - \\text{Score}_{\\text{min}}}{\\text{Score}_{\\text{max}} - \\text{Score}_{\\text{min}}}$$
          * **Z-Score Standardization**: Scores collocates by standard deviations ($\\sigma$) from each node's mean score ($\\mu$), highlighting relative importance:
            $$\\text{Z-Score} = \\frac{\\text{Score} - \\mu}{\\sigma}$$
        
        #### 5. Interactive Visualizations
        * **Bubble Matrix (Clean Grid)**: Plotted on a clean grid (Node Words vs. Collocates) where circle size represents **Frequency** and circle color represents **Association Score**.
        * **Stacked Bar Chart**: Displays combined strength as a single horizontal bar where segments are color-coded by node word.
        * **Overlap Size Overview**: Custom HTML horizontal bars mapping shared combinations. Collocate words are displayed directly inside each bar, with their font sizes dynamically scaled based on their combined association strength.
    """,
    "Dictionary": """
        ### 📖 Dictionary User Guide
        
        The Dictionary module provides built-in dictionary lookups, definition tracking, and thesaurus synonyms:
        
        #### 1. Definition Lookup
        * Type any word to fetch standard semantic definitions, phonetic transcriptions, grammatical categories, and usage examples.
        * Uses your confirmed corpus language settings to route requests to the correct language database.
        
        #### 2. Thesaurus & Synonyms
        * Find synonyms, antonyms, and related words to explore lexical variations and vocabulary alternatives.
    """,
    "Word Profiler": """
        ### 📖 Word Profiler User Guide
        
        The Word Profiler analyzes your corpus coverage using pre-defined wordlists (e.g. vocab bands, academic vocabulary lists).
        
        #### 1. Supported Wordlist Formats
        Wordlists can be uploaded or loaded from the following file formats:
        * **Plain Text (`.txt`)**: Tab-separated or single-word per line.
        * **CSV (`.csv`)**: Commas or tabs separated, containing 1 or 2 columns.
        * **Excel (`.xlsx` / `.xls`)**: Spreadsheet sheets containing 1 or 2 columns.
        
        ##### Layout Formats:
        * **Plain Wordlist (Single Column)**:
          * A list where each row contains a single word.
          * All matched words are aggregated under a default category labeled **"Coverage"**.
          * *Example:*
            ```text
            coffee
            tea
            milk
            ```
        * **Categorized Wordlist (Two Columns)**:
          * Row structures where Column 1 contains the word and Column 2 contains the category label.
          * Tab-separated for `.txt`, comma-separated for `.csv`, or cells in columns A and B for Excel.
          * Useful for evaluating coverage across custom categories (e.g. GSL/AWL lists, vocabulary bands, or semantic domains).
          * *Example:*
            ```text
            apple,Fruits
            coffee,Beverages
            milk,Beverages
            ```
        * **Three-Column Wordlist (Word, Label, Lemma)**:
          * Row structures where Column 1 contains the literal word, Column 2 contains the category label, and Column 3 contains the base lemma.
          * Tab-separated for `.txt`, comma-separated for `.csv`, or columns A, B, and C for Excel.
          * Both the literal word (Col 1) and its lemma (Col 3) are mapped to the category. This ensures that corpus occurrences match correctly whether they appear as the inflected word or the base lemma (e.g. matching "ran" via its lemma "run").
          * *Example:*
            ```text
            apples,Fruits,apple
            drinking,Beverages,drink
            ```
        * *Note*: If the first row contains labels like *word*, *token*, *lemma*, *collocate*, or *category*, it will be automatically identified and treated as a header row (skipped during analysis). Any words in your corpus not matching the loaded wordlist will automatically be categorized under **OOV** (Out of Vocabulary).
        
        #### 2. Configuration Settings
        * **Analysis Basis**: Evaluate vocabulary coverage across:
          * **Whole Corpus**: Calculate total corpus coverage metrics.
          * **By Filename**: Compare coverage levels between different files.
          * **By Metadata**: Segment and compare coverage based on document attributes (e.g. publication year, genre).
        * **Filtering**: Restrict analysis to specific document cohorts using XML filters.

        #### 3. AI Interpretation
        * Click the **🤖 Interpret Results with AI** button under any wordlist results box to generate a scholarly markdown interpretation of vocabulary coverage and distribution grounded strictly in your empirical data.
    """,
    "Keyword": """
        ### 📖 Keyword Analysis Guide
        
        Keywords are words whose frequency in a target (study) corpus is statistically higher (or lower) than in a reference corpus.
        
        #### 1. Reference Corpus Selection
        * **Pre-built or Uploaded**: Select a reference corpus (e.g. BNC or Brown) or upload a custom text/frequency file to serve as the baseline comparison.
        
        #### 2. Settings & Analysis Basis
        * **P-Value Cutoff**: Restrict keywords to those meeting significance thresholds (e.g., 0.01 or 0.05).
        * **Analysis Basis (Optional)**:
          * **By Individual File**: Checked to generate and compare separate keyword lists for every unique file in the study corpus (e.g., creating 4 lists for 4 files).
          * **By Sub-corpora Attributes**: Checked to generate separate keyword lists for each XML attribute group (e.g. producing lists for every year, genre, or sentiment attribute value).
          * *Note: If these options are unchecked, the system calculates and displays only the 1 overall corpus-level keyword list, optimizing performance.*
        
        #### 3. Interpretation
        * **Positive Keywords (High Keyness)**: Words used significantly **more** in the target than the reference (reflecting target-specific themes).
        * **Negative Keywords (Low Keyness)**: Words used significantly **less** or completely absent in the target.
    """,
    "Distribution": """
        ### 📖 Distribution User Guide
        
        The Distribution module displays how words, lemmas, or metadata tags are distributed across different segments of your corpus:
        
        #### 1. Variable Mapping
        * Select an target query word and choose a metadata attribute (e.g., *genre*, *year*, *subcorpus*) as the mapping axis.
        * Displays absolute and relative frequencies partitioned by each sub-corpus category.
        
        #### 2. Plotting & Exporting
        * Renders bar charts and distribution tables to visualize frequency variations, and supports Excel export.
    """,
    "Statistical Testing": """
        ### 📖 Statistical Testing User Guide
        
        This module applies formal hypothesis testing to check if differences in word counts or readability metrics are statistically significant.
        
        #### 1. Selecting Test Variables
        * Choose the categories/sub-corpora to compare, and set the target frequencies or readability indices.
        * **Test Options**: Runs **Chi-Square Tests** for categorical distributions, or **t-tests / Mann-Whitney U** for numerical metrics.
        
        #### 2. Interpreting p-values
        * The interface highlights the test statistic, degrees of freedom, and the **p-value**. If $p < 0.05$, the differences are marked as statistically significant.
    """,
    "Summarisation": """
        ### 📖 Summarisation User Guide
        
        Generate concise summaries of documents or sections using natural language processing:
        
        #### 1. Input Source
        * Select specific documents, XML structural sections, or load custom text buffers.
        * **Length Adjuster**: Use sliders to define the target sentence limit or word count range for the summary output.
        
        #### 2. Model Selection
        * Choose between extractive summarizers (selecting key source sentences) or generative AI summarization.
    """,
    "Quiz Creation": """
        ### 📖 Quiz Creation User Guide
        
        Automatically compile interactive vocabulary and grammar quizzes based on your active corpus texts:
        
        #### 1. Question Types
        * Select question structures: **Multiple Choice**, **Fill in the Blanks**, or **Matching Definitions**.
        
        #### 2. Vocabulary & PASSAGE Selection
        * Define target words or choose source reading passages. The system automatically extracts distractors from the corpus vocabulary to build options.
        * **Export Quizzes**: Save questions and answer keys to text files or PDFs.
    """,
    "Word Trend": {
        "Word Tracker": """
            ### 📈 Word Tracker Guidelines
            
            This tab allows you to visualize and statically analyze the trajectory of specific words over time.
            1. **Metadata Configuration:** Select a temporal metadata column (e.g., Year, Decade, Date) from the **Time Attribute** dropdown.
            2. **Search Modes:**
                *   **Simple Mode:** Perfect for quick tracking. Select the **Simple** radio button, then type your target words into the input box separated by commas (e.g., `technology, computer, internet`).
                *   **Advanced Mode:** Allows for highly specific linguistic queries. Select the **Advanced** radio button. Click the **+ Add Query** button to dynamically stack multiple query boxes. The syntax mirrors the Advanced Concordance:
                    *   **POS Tags:** `can_NN` (finds "can" used strictly as a noun).
                    *   **Lemmas:** `[be]` (finds all inflections like is, am, are, was, were).
                    *   **Wildcards:** `*ing` (finds any token ending in "ing").
                    *   **Unions:** `can|could` (finds instances of either word).
            3. **Output Basis (Advanced Only):**
                *   **Word:** Click the **Word** radio button to explode your query into its individual matched tokens. For example, querying `[see]` will plot separate lines for `see`, `saw`, `seen`, and `seeing`.
                *   **Lemma:** Click the **Lemma** radio button to group all matched tokens into a single aggregated line representing your core query (e.g., one line for `[see]`).
            4. **Statistics & Interpretation:**
                *   Scroll down to the **Inferential Statistics & Interpretation** section.
                *   Select an analysis type from the radio buttons (e.g., Correlation, Trend Comparison).
                *   Click the **Generate Chart & Analysis** button to calculate the results and generate AI interpretation.
        """,
        "Exclusive Words": """
            ### 💎 Exclusive Words Guidelines
            
            This tab identifies vocabulary that is **entirely unique** to one specific metadata period and completely absent from all others.
            *   **Usage:** Useful for finding highly specialized terminology or jargon isolated to a specific genre, author, or era.
            *   **Configuration:** 
                1. Select a **Time Attribute** from the dropdown. 
                2. Review the detected chronological order in the box below it.
            *   **Part-of-Speech Filter (Optional):** You can apply POS filters to narrow down the results. 
                1. Click the **Include** radio button if you only want to analyze words that match your selected tags (e.g., Select Nouns and Verbs in the dropdown below).
                2. Click the **Exclude** radio button if you want to analyze everything *except* your selected tags (e.g., Select punctuation tags in the dropdown below to ignore them).
        """,
        "Emerging Words": """
            ### 🌱 Emerging Words Guidelines
            
            This tab tracks the **first appearance** of words in your corpus across an ordered timeline. 
            *   **Usage:** Ideal for neologism research and tracking language evolution. It highlights when a word was introduced into the corpus and measures its relative frequency at the time of emergence.
            *   **Configuration:** 
                1. Select a **Time Attribute** from the dropdown. 
                2. Review the detected chronological order in the box below it.
            *   **Part-of-Speech Filter (Optional):** You can apply POS filters to narrow down the results. 
                1. Click the **Include** radio button if you only want to analyze words that match your selected tags (e.g., Select Nouns and Verbs in the dropdown below).
                2. Click the **Exclude** radio button if you want to analyze everything *except* your selected tags (e.g., Select punctuation tags in the dropdown below to ignore them).
        """
    }
}

def render_guidelines(module_name, sub_tab=None, key_prefix=""):
    """
    Renders guidelines in a side-by-side sticky column if guidelines are toggled on.
    Returns: col_main, col_guide
    """
    # Guidelines Toggle Button
    state_key = f'{module_name.lower().replace(" ", "_")}_show_guidelines'
    show_guide = get_state(state_key, False)
    
    if st.button("📖 Show Guidelines" if not show_guide else "✖ Hide Guidelines", key=f"btn_guide_{module_name.lower().replace(' ', '_')}_{key_prefix}"):
        show_guide = not show_guide
        set_state(state_key, show_guide)
        st.rerun()

    if show_guide:
        col_main, col_guide = st.columns([5, 3])
        with col_guide:
            st.markdown("""
            <style>
            div[data-testid="column"]:has(.sticky-guidelines) {
                position: -webkit-sticky !important;
                position: sticky !important;
                top: 80px !important;
                align-self: flex-start !important;
            }
            </style>
            <div class="sticky-guidelines"></div>
            """, unsafe_allow_html=True)
            with st.container(border=True):
                # Retrieve the markdown content
                if module_name in ["Overview", "Word Trend"] and sub_tab:
                    content = GUIDELINES.get(module_name, {}).get(sub_tab, "No instructions available.")
                else:
                    content = GUIDELINES.get(module_name, "No instructions available.")
                st.markdown(content)
    else:
        col_main = st.container()
        
    return col_main
