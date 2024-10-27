
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
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["good boy", "good girl", "good boy girl"]
vectorizer = TfidfVectorizer()
vectorizer.smooth_idf=False

data={}
X = vectorizer.fit_transform(corpus)
ft_names=vectorizer.get_feature_names_out()
print(ft_names)
a=X.toarray()
print(len(ft_names))
for i in range(len(ft_names)):
    data[corpus[i]]=a[i]
df=pd.DataFrame(data,index=ft_names)
print(df.to_string())
print(df.transpose())
# print(f'a:{a},\n a[0] {a[0]}')
# print(X.toarray())
