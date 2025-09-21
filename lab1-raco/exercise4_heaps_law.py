import nltk
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from scipy import stats
import urllib.request

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

FIGSIZE = (5, 3)

def load_don_quijote():
    """
    Load Don Quijote text from local file or URL.
    Returns the text as a string.
    """
    try:
        with open("don_quijote.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print("Local file not found, downloading from URL...")
        target_url = "https://fegalaz.usc.es/~gamallo/aulas/lingcomputacional/corpus/quijote-en.txt"
        try:
            with urllib.request.urlopen(target_url) as response:
                return response.read().decode('latin-1')
        except Exception as e:
            print(f"Error downloading file: {e}")
            print("Please ensure 'don_quijote.txt' is in the same directory as this script.")
            return ""

def preprocess_text(text):
    """
    Basic preprocessing: tokenization and lowercasing.
    Keeps all words including stopwords as per exercise instructions.
    """
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    return tokens

def calculate_vocabulary_growth(tokens, step_size=500):
    """
    Calculate vocabulary size at different text lengths.
    
    Args:
        tokens: List of preprocessed tokens
        step_size: How often to sample vocabulary size
    
    Returns:
        text_lengths: List of text lengths (N values)
        vocab_sizes: List of vocabulary sizes (d values)
    """
    text_lengths = []
    vocab_sizes = []
    seen_words = set()
    
    for i in range(step_size, len(tokens) + 1, step_size):
        chunk = tokens[:i]
        seen_words.update(chunk)
        
        text_lengths.append(i)
        vocab_sizes.append(len(seen_words))
    
    if len(tokens) not in text_lengths:
        seen_words = set(tokens)
        text_lengths.append(len(tokens))
        vocab_sizes.append(len(seen_words))
    
    return text_lengths, vocab_sizes

def plot_vocabulary_growth(text_lengths, vocab_sizes, log_scale=False):
    """
    Plot vocabulary size vs text length.
    
    Args:
        text_lengths: List of text lengths (N values)
        vocab_sizes: List of vocabulary sizes (d values)  
        log_scale: Whether to use log-log scale
    """
    plt.figure(figsize=FIGSIZE)
    
    if log_scale:
        plt.loglog(text_lengths, vocab_sizes, 'bo-', markersize=4, linewidth=1)
        plt.xlabel('Text Length (N) - Log Scale')
        plt.ylabel('Vocabulary Size (d) - Log Scale')
        plt.title("Heap's Law: Vocabulary Growth (Log-Log Scale)")
        plt.grid(True, alpha=0.3)
    else:
        plt.plot(text_lengths, vocab_sizes, 'bo-', markersize=4, linewidth=1)
        plt.xlabel('Text Length (N)')
        plt.ylabel('Vocabulary Size (d)')
        plt.title("Heap's Law: Vocabulary Growth (Linear Scale)")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def estimate_heaps_parameters(text_lengths, vocab_sizes):
    """
    Estimate k and β parameters of Heap's law using linear regression.
    
    Heap's law: d = k * N^β
    Taking log: log(d) = log(k) + β * log(N)
    
    Returns:
        k: scaling factor
        beta: exponent parameter
        r_squared: coefficient of determination
    """
    log_N = np.log(text_lengths)
    log_d = np.log(vocab_sizes)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_N, log_d)
    
    beta = slope
    k = np.exp(intercept)
    r_squared = r_value ** 2
    
    return k, beta, r_squared

def plot_heaps_law_fit(text_lengths, vocab_sizes, k, beta):
    """
    Plot actual vocabulary growth vs Heap's law fit.
    """
    plt.figure(figsize=FIGSIZE)
    
    plt.loglog(text_lengths, vocab_sizes, 'bo', markersize=4, label='Actual Data', alpha=0.7)
    
    N_range = np.linspace(min(text_lengths), max(text_lengths), 100)
    d_fit = k * (N_range ** beta)
    plt.loglog(N_range, d_fit, 'r-', linewidth=2, 
              label=f"Heap's Law: d = {k:.1f} * N^{beta:.3f}")
    
    plt.xlabel('Text Length (N) - Log Scale')
    plt.ylabel('Vocabulary Size (d) - Log Scale')
    plt.title("Heap's Law Fit vs Actual Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main(text):
    """
    Main function to run Exercise 4 analysis.
    """
    print("Exercise 4: Heap's Law Analysis for Don Quijote")
    print("=" * 50)
    
    print("Loading Don Quijote text...")
    
    if not text:
        print("Error: Could not load text. Please check file availability.")
        return
    
    print("Preprocessing text...")
    tokens = preprocess_text(text)
    print(f"Total tokens after preprocessing: {len(tokens):,}")
    
    print("Calculating vocabulary growth...")
    text_lengths, vocab_sizes = calculate_vocabulary_growth(tokens, step_size=1000)
    
    print(f"Text length range: {min(text_lengths):,} to {max(text_lengths):,} words")
    print(f"Vocabulary size range: {min(vocab_sizes):,} to {max(vocab_sizes):,} unique words")
    
    print("\n1. Plotting vocabulary growth (linear scale)...")
    plot_vocabulary_growth(text_lengths, vocab_sizes, log_scale=False)
    
    print("2. Plotting vocabulary growth (log-log scale)...")
    plot_vocabulary_growth(text_lengths, vocab_sizes, log_scale=True)
    
    print("3. Estimating Heap's law parameters...")
    k, beta, r_squared = estimate_heaps_parameters(text_lengths, vocab_sizes)
    
    print(f"\nHeap's Law Parameters:")
    print(f"k (scaling factor): {k:.2f}")
    print(f"β (exponent): {beta:.4f}")
    print(f"R² (goodness of fit): {r_squared:.4f}")
    
    print("4. Plotting Heap's law fit vs actual data...")
    plot_heaps_law_fit(text_lengths, vocab_sizes, k, beta)
    

if __name__ == "__main__":
    main(load_don_quijote())
