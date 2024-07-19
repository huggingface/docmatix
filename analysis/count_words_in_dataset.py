from collections import Counter
import string

def count_words(df, column_name):
    # Initialize a Counter to count all words
    overall_counter = Counter()
    
    # List to store word counts per row
    word_counts = []
    
    for text in df[column_name]:
        text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        words = text.lower().split()
        word_count = len(words)
        word_counts.append(word_count)
        
        # Update overall word counter
        overall_counter.update(words)
    
    # Add word counts as a new column
    df['word_count'] = word_counts
    
    # Get the most common words
    most_common_words = overall_counter.most_common(100)
    
    return df, most_common_words

# df_texts, most_common_words = count_words(df_texts, 'question')