import nltk
import nltk.corpus

# Tokenize the text
text = "Xin chào, hôm nay bạn có khỏe không?"
tokens = nltk.word_tokenize(text)

# Use NLTK's pos_tag function to tag the tokens using the VTB tagset
tags = nltk.pos_tag(tokens, tagset="vietnamese")

# print(tags)  # Output: [('Xin', 'V'), ('chào', 'N'), (',', 'CH'), ('hôm', 'N'), ('nay', 'N'), ('bạn', 'N'), ('có', 'V'), ('khỏe', 'A'), ('không', 'R'), ('?', 'CH')]

# # Download the VNERE dataset from the nltk data repository
# nltk.download("vnere")
#
# # Load the VNERE dataset
# vnere_corpus = nltk.corpus.vnere
#
# # Split the dataset into training and test sets
# train_sents = vnere_corpus.tagged_sents()[:1000]
# test_sents = vnere_corpus.tagged_sents()[1000:]
#
# # Train a sequence classifier on the training set
# classifier = nltk.classify.maxent.MaxentClassifier.train(train_sents)
#
# # Test the classifier on the test set
# print(nltk.classify.accuracy(classifier, test_sents))
#
# # Define a function to extract named entities from a Vietnamese text
# def extract_entities(text):
#   tokens = nltk.word_tokenize(text)
#   tags = classifier.classify_many(tokens)
#   entities = []
#   for token, tag in tags:
#     if tag != "O":
#       entities.append((token, tag))
#   return entities
#
# # Test the function on a Vietnamese text
# text = "Vietnam là một nước Đông Nam Á có nhiều văn hóa và lịch sử, cũng như các bãi biển và khu vực thiên nhiên đẹp. Thủ đô của Vietnam là Hà Nội và nước này có hơn 96 triệu người."
#
# entities = extract_entities(text)
# print(entities)
