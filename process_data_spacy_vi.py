import spacy,regex,re,string,nltk
from spacy.lang.vi import Vietnamese
from spacy.lang.en import English
from pyvi import ViTokenizer, ViPosTagger

def remove_punctuation(text):
    punctuation_free="".join([i for i in text if i not in string.punctuation])
    return punctuation_free

# Đối tượng "nlp"  được dùng để tạo documents với chú thích ngôn ngữ (linguistic annotations).
nlp_vi = Vietnamese()
nlp_en=English()

#Khai báo tên file path để đọc
file_name="CNHD_textmining_gt"
# Đọc dữ liệu từ file .txt
with open('Vi_sample\\'+file_name+".txt", 'r', encoding='utf-8') as file:
    text = file.read()


#Loại bỏ từ
print(" văn bản gốc:\n",text)
text=remove_punctuation(text)
text=re.sub("[-\n]","",text)

my_doc = nlp_vi(text)

# Tạo danh sách tokens từ
token_list = []
for token in my_doc:
    token_list.append(token.text)


# Tạo danh sách tokens sau khi loại bỏ từ dùng
filtered_words =[]
for word in token_list:
    # print(word)
    lexeme_vi = nlp_vi.vocab[word]
    lexeme_en= nlp_en.vocab[word]
    if (not (lexeme_vi.is_stop==True or lexeme_en.is_stop==True or word.isspace() or (word in string.punctuation))):
        filtered_words.append(word)

#In ra list các tokens và list các token sau khi loại bỏ từ dừng
print("danh sách token :\n",token_list)
print("\n\n danh sách words tokens sau khi loại bỏ từ dừng:\n",filtered_words)

processed_text = ' '.join(filtered_words)
#Ghi kết quả ra file
with open('Processed Results\\'+file_name+"_preprocessed.txt", 'w', encoding='utf-8') as file:
    file.write(processed_text)

# print(ViTokenizer.tokenize(u"Text mining (khai thác văn bản) là quá trình sử dụng các công cụ và kỹ thuật để trích xuất thông tin có giá trị từ các nguồn dữ liệu văn bản lớn, không có cấu trúc từ các nguồn có sẵn như trang web, mạng xã hội, báo cáo, bài viết, khảo sát,..")
# )
# my_doc=ViTokenizer.tokenize(u"Text mining (khai thác văn bản) là quá trình sử dụng các công cụ và kỹ thuật để trích xuất thông tin có giá trị từ các nguồn dữ liệu văn bản lớn, không có cấu trúc từ các nguồn có sẵn như trang web, mạng xã hội, báo cáo, bài viết, khảo sát,..")
#
# print(my_doc)