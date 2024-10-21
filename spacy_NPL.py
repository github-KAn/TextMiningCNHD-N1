import spacy
from spacy.lang.vi import Vietnamese
from pyvi import ViTokenizer, ViPosTagger


# # Load Vietnamese tokenizer, tagger, parser, NER and word vectors
nlp = Vietnamese()

with open('sample_text_vi.txt', 'r', encoding='utf-8') as file:
    text = file.read()
print(ViTokenizer.tokenize(u"Text mining (khai thác văn bản) là quá trình sử dụng các công cụ và kỹ thuật để trích xuất thông tin có giá trị từ các nguồn dữ liệu văn bản lớn, không có cấu trúc từ các nguồn có sẵn như trang web, mạng xã hội, báo cáo, bài viết, khảo sát,..")
)
my_doc=ViTokenizer.tokenize(u"Text mining (khai thác văn bản) là quá trình sử dụng các công cụ và kỹ thuật để trích xuất thông tin có giá trị từ các nguồn dữ liệu văn bản lớn, không có cấu trúc từ các nguồn có sẵn như trang web, mạng xã hội, báo cáo, bài viết, khảo sát,..")

print(my_doc)

#  "nlp" Object is used to create documents with linguistic annotations.
my_doc = nlp(text)

# Create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)

from spacy.lang.vi.stop_words import STOP_WORDS

# Create list of word tokens after removing stopwords
filtered_sentence =[]

for word in token_list:
    lexeme = nlp.vocab[word]
    if not lexeme.is_stop:
        filtered_sentence.append(word)
print("token list",token_list)
print("\n\nfiltered sentence",filtered_sentence)