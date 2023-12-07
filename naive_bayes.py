from sklearn.naive_bayes import CategoricalNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import nltk
from nltk.corpus import treebank



# Adds up to 1
#
# P(+) = 0.6
# P(-) = 0.4

# Doesn't necessarily add up to 1
#
#       "with"  "to"    "bad"   "from"
# P(+)  0.6     0.75    0.1       0.41
# P(-)  0.1     0.2     0.3       0.11


df = pd.read_csv("emails.csv")

# Shuffle the dataframe's rows to improve results
df = df.sample(frac = 1, random_state = 4).reset_index()

df_train = df.iloc[:int(0.75 * len(df)), :]
df_test = df.iloc[int(0.75 * len(df)):, :]

# print(len(df))
# print(len(df_train))
# print(len(df_test))

# print(df_train)
# print(df_test)

df_test.reset_index(drop=True, inplace=True)

# text = df["text"]
# spam = df["spam"]

# t = treebank.parsed_sents('wsj_0001.mrg')[0]
# t.draw()

#print(text.value_counts())

#print(nltk.word_tokenize(text[0]))

word_dictionary = dict()

columns = df.columns

#print(len(df.iloc[0]))

#print(df["spam"][0])

# This counts the number of spam vs authentic emails
num_spam_emails = len(df_train[df_train.spam == 1])
num_authentic_emails = len(df_train[df_train.spam == 0])

# Calculates the probability of spam vs authentic emails
p_spam = num_spam_emails / (num_spam_emails + num_authentic_emails)
p_authentic = num_authentic_emails / (num_spam_emails + num_authentic_emails)

# This keeps track of the number of words that occur that are either spam or authentic (including duplicates)
num_spam_words = 0
num_authentic_words = 0


# Perform training on our dataset
for row in range(len(df_train)):

    sentence = df_train["text"][row]
    classification = df_train["spam"][row]

    #print(sentence)

    parsed_sentence = nltk.word_tokenize(sentence)

    for word in parsed_sentence:

        if word in word_dictionary:

            # Is Spam
            if classification:

                word_dictionary[word][1] += 1

                num_spam_words += 1

            # Isn't Spam
            else:

                word_dictionary[word][0] += 1

                num_authentic_words += 1

        else:

            # Is Spam
            if classification:

                # [+, -]
                word_dictionary[word] = [0, 1]

                num_spam_words += 1

            # Isn't Spam
            else:

                # [+, -]
                word_dictionary[word] = [1, 0]

                num_authentic_words += 1


word_dict_keys = list(word_dictionary.keys())

# False Positive and True Positive
tp = 0
fp = 0
tn = 0
fn = 0


# Use the training values on the testing set
for row in range(len(df_test)):

    # These are for testing
    sentence = df_test["text"][row]
    classification = df_test["spam"][row]

    parsed_sentence = nltk.word_tokenize(sentence)

    # Count the number of all non-repeated words used in both the training set and this current email
    word_union = len(set(word_dict_keys + parsed_sentence))

    # These are the probabilities that the entire test email is spam or not
    # These values start out as being the probability of an email being spam/authentic
    p_test_spam = p_spam
    p_test_authentic = p_authentic

    for word in parsed_sentence:

        # Count the number of instances of the current word within our test email
        word_count = parsed_sentence.count(word)

        if word in word_dictionary:

            # Calculate the probability of our current word as both spam/authentic
            p_word_spam = (word_dictionary[word][1] + word_count) / (num_spam_words + word_union)
            p_word_authentic = (word_dictionary[word][0] + word_count) / (num_authentic_words + word_union)

            # Update the probability of this current email being spam/authentic
            p_test_spam *= p_word_spam
            p_test_authentic *= p_word_authentic

    # This means we concluded the email is spam
    if p_test_spam > p_test_authentic:

        # If the email is actually authentic
        if classification == 0:

            fp += 1

        # If the email is actually spam
        else:

            tp += 1

    # This means we concluded the email is authentic
    else:

        # If the email is actually authentic
        if classification == 0:

            tn += 1

        # If the email is actually spam
        else:

            fn += 1

print("\n(When run on Testing data)")
print("True Positive Count: " + str(tp))
print("False Positive Count: " + str(fp))
print("True Negative Count: " + str(tn))
print("False Negative Count: " + str(fn))

# False Positive and True Positive
tp = 0
fp = 0
tn = 0
fn = 0

# Use the training values on the testing set
for row in range(len(df_train)):

    # These are for testing
    sentence = df_train["text"][row]
    classification = df_train["spam"][row]

    parsed_sentence = nltk.word_tokenize(sentence)

    # Count the number of all non-repeated words used in both the training set and this current email
    word_union = len(set(word_dict_keys + parsed_sentence))

    # These are the probabilities that the entire test email is spam or not
    # These values start out as being the probability of an email being spam/authentic
    p_test_spam = p_spam
    p_test_authentic = p_authentic

    for word in parsed_sentence:

        # Count the number of instances of the current word within our test email
        word_count = parsed_sentence.count(word)

        if word in word_dictionary:

            # Calculate the probability of our current word as both spam/authentic
            p_word_spam = (word_dictionary[word][1] + word_count) / (num_spam_words + word_union)
            p_word_authentic = (word_dictionary[word][0] + word_count) / (num_authentic_words + word_union)

            # Update the probability of this current email being spam/authentic
            p_test_spam *= p_word_spam
            p_test_authentic *= p_word_authentic

    # This means we concluded the email is spam
    if p_test_spam > p_test_authentic:

        # If the email is actually authentic
        if classification == 0:

            fp += 1

        # If the email is actually spam
        else:

            tp += 1

    # This means we concluded the email is authentic
    else:

        # If the email is actually authentic
        if classification == 0:

            tn += 1

        # If the email is actually spam
        else:

            fn += 1

print("\n(When run on Training data)")
print("True Positive Count: " + str(tp))
print("False Positive Count: " + str(fp))
print("True Negative Count: " + str(tn))
print("False Negative Count: " + str(fn))

#print(word_dictionary.keys())

#print(word_dictionary.values())

print("Done!")