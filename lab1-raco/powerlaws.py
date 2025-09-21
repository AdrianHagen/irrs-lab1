# Utils script for the first exercise
import nltk

nltk.download("wordnet")
import pprint
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
from typing import Dict
import urllib.request
from collections import Counter
import numpy as np

FIGSIZE = (5, 3)

# target_url = (
#     "https://fegalaz.usc.es/~gamallo/aulas/lingcomputacional/corpus/quijote-en.txt"
# )

# quijote_text = urllib.request.urlopen(target_url)

# # tokenized with no pre process
# tokenized_text = word_tokenize(quijote_text)

# # remove stopwords
# english_sw = set(stopwords.words("english") + list(string.punctuation))

# filtered_tokenized_text = [
#     w.lower() for w in tokenized_text if w.lower() not in english_sw
# ]

# pprint.pprint(filtered_tokenized_text)


def get_word_frequencies(
    text: str,
    tokenize: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    remove_stopwords: bool = False,
):
    """
    Returns the word frequencies for a text in descending order.

    The text will be lowercased and optionally tokenized, lemmatized, and/or stemmed.

    Args:
        text (str): The text to analyze.
        tokenize (bool): Whether to tokenize the text using nltk.tokenize.word_tokenize(). Defaults to True.
        lemmatize (bool): Whether to lemmatize the tokens. Defaults to False.
        stem (bool): Whether to stem the tokens. Defaults to False.
        remove_stopwords (bool): Whether to remove English stopwords and punctuation. Defaults to False.

    Returns:
        Dict[str, int]: A dictionary containing each word and its frequency.
    """
    text = text.lower()

    if tokenize:
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]
    else:
        # Simple split on whitespace if not tokenizing
        tokens = text.split()
        tokens = [token for token in tokens if token.isalpha()]

    if lemmatize:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if stem:
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(token) for token in tokens]

    if remove_stopwords:
        english_stopwords = set(stopwords.words("english") + list(string.punctuation))
        tokens = [token for token in tokens if token.lower() not in english_stopwords]

    word_frequencies = Counter(tokens)
    sorted_frequencies = word_frequencies.most_common()

    return sorted_frequencies


def plot_frequencies(
    frequencies: Dict[str, int], log_x: bool = True, log_y: bool = True
):
    """
    Plots the word frequencies of the text.

    X-Axis will contain the Rank of each word.

    Y-Axis will contain the frequency for each word.

    Args:
        frequencies (Dict[str, int]): Dictionary with each word and its corresponding frequency.
        log_x: Whether to plot the x-axis using log-scale. Defaults to true.
        log_y: Whether to plot the y-axis using log-scale. Defaults to true.
    """
    # Extract frequencies and ranks for plotting
    freq_values = [freq for word, freq in frequencies]
    ranks = list(range(1, len(freq_values) + 1))

    plt.figure(figsize=FIGSIZE)

    if log_x and log_y:
        plt.loglog(ranks, freq_values, "b-", linewidth=1)
        plt.xlabel("Word Rank (log scale)")
        plt.ylabel("Frequency (log scale)")
        plt.title("Word Frequency vs Rank (Log-Log Scale)")
    elif log_x and not log_y:
        plt.semilogx(ranks, freq_values, "b-", linewidth=1)
        plt.xlabel("Word Rank (log scale)")
        plt.ylabel("Frequency")
        plt.title("Word Frequency vs Rank (Semi-Log X Scale)")
    elif not log_x and log_y:
        plt.semilogy(ranks, freq_values, "b-", linewidth=1)
        plt.xlabel("Word Rank")
        plt.ylabel("Frequency (log scale)")
        plt.title("Word Frequency vs Rank (Semi-Log Y Scale)")
    else:
        plt.plot(ranks, freq_values, "b-", linewidth=1)
        plt.xlabel("Word Rank")
        plt.ylabel("Frequency")
        plt.title("Word Frequency vs Rank (Linear Scale)")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_frequencies_and_plot(
    text: str,
    tokenize: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    remove_stopwords: bool = False,
    log_x: bool = True,
    log_y: bool = True,
):
    """
    Analyzes word frequencies from text and creates a frequency vs rank plot.

    This is a convenience function that combines get_word_frequencies() and plot_frequencies()
    to analyze and visualize word frequency distributions in a single call.

    Args:
        text (str): The text to analyze.
        tokenize (bool): Whether to tokenize the text using nltk.tokenize.word_tokenize(). Defaults to True.
        lemmatize (bool): Whether to lemmatize the tokens. Defaults to False.
        stem (bool): Whether to stem the tokens. Defaults to False.
        remove_stopwords (bool): Whether to remove English stopwords and punctuation. Defaults to False.
        log_x (bool): Whether to plot the x-axis using log-scale. Defaults to True.
        log_y (bool): Whether to plot the y-axis using log-scale. Defaults to True.
    """
    frequencies = get_word_frequencies(
        text, tokenize, lemmatize, stem, remove_stopwords
    )
    plot_frequencies(frequencies, log_x, log_y)


def plot_power_law(a: float, c: float, x_range: tuple = (1, 1000)):
    """
    Plots a power law equation of the form y = c * x^a.

    Args:
        a (float): The exponent parameter of the power law.
        c (float): The coefficient parameter of the power law.
        x_range (tuple): Tuple of (x_min, x_max) for the range to plot. Defaults to (1, 1000).
    """
    # Generate x values for the power law curve
    x_min, x_max = x_range
    x_values = np.linspace(x_min, x_max, 1000)

    # Calculate y values using the power law: y = c * x^a
    y_values = c * (x_values**a)

    plt.figure(figsize=FIGSIZE)
    plt.loglog(x_values, y_values, "r-", linewidth=2)
    plt.xlabel("Word Rank (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Power Law: y = {c:.1f} * x^{a:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_power_law_and_frequencies():
    plot_power_law()
    get_frequencies_and_plot()


def plot_combined_power_law_and_frequencies(
    text: str,
    a: float,
    c: float,
    tokenize: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    remove_stopwords: bool = False,
    x_range: tuple = (1, 10000),
):
    """
    Plots both the power law curve and actual word frequencies on the same plot for comparison.

    Args:
        text (str): The text to analyze for word frequencies.
        a (float): The exponent parameter of the power law.
        c (float): The coefficient parameter of the power law.
        tokenize (bool): Whether to tokenize the text using nltk.tokenize.word_tokenize(). Defaults to True.
        lemmatize (bool): Whether to lemmatize the tokens. Defaults to False.
        stem (bool): Whether to stem the tokens. Defaults to False.
        remove_stopwords (bool): Whether to remove English stopwords and punctuation. Defaults to False.
        x_range (tuple): Tuple of (x_min, x_max) for the range to plot. Defaults to (1, 1000).
    """
    frequencies = get_word_frequencies(
        text, tokenize, lemmatize, stem, remove_stopwords
    )

    freq_values = [freq for word, freq in frequencies]
    ranks = list(range(1, len(freq_values) + 1))

    x_min, x_max = x_range
    x_values = np.linspace(x_min, min(x_max, len(freq_values)), 1000)
    y_values = c * (x_values**a)

    plt.figure(figsize=FIGSIZE)

    plt.loglog(ranks, freq_values, "b-", linewidth=1, label="Actual Frequencies")

    plt.loglog(
        x_values,
        y_values,
        "r-",
        linewidth=2,
        label=f"Power Law: y = {c:.1f} * x^{a:.3f}",
    )

    plt.xlabel("Word Rank (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title("Word Frequency vs Rank: Actual vs Power Law")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
