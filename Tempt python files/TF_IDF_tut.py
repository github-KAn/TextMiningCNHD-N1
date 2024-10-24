
# from sklearn.feature_extraction.text import TfidfVectorizer
# # Example documents
# docs = ["The cat jumped",
#         "The white tiger roared",
#         "Bird flying in the sky"]
# # Create a TfidfVectorizer object
# vectorizer = TfidfVectorizer()
# # Use the fit_transform method to transform the documents into a TF-IDF matrix
# tfidf = vectorizer.fit_transform(docs)
# # Print the vocabulary (features) of the TF-IDF matrix
# # print(vectorizer.get_feature_names_out())
# features_names=vectorizer.get_feature_names_out()
# print(features_names)
# # print(list(map('{:10}'.format,features_names)))
# # Print the TF-IDF matrix
# print(f"{tfidf.toarray()}")

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["good boy", "good girl", "good boy girl"]
vectorizer = TfidfVectorizer()
vectorizer.smooth_idf=False

X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())