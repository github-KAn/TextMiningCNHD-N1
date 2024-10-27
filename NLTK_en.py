# pip install nltk spacy sklearn textblob gensim
import nltk, pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import ssl

# Bỏ qua chứng chỉ SSL để tránh lỗi tải xuống
ssl._create_default_https_context = ssl._create_unverified_context

# Tải các công cụ cần thiết cho NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def process_text(_text_data):
    # 1. Chuyển thành chữ thường
    _text_data = _text_data.lower()

    # 2. Tokenization (Tách từ)
    _tokens = word_tokenize(_text_data)

    # 3. Loại bỏ từ dừng (Stopwords)
    _tokens = [word for word in _tokens if word.isalpha() and word not in stopwords.words('english')]
    # 4. Lemmatization (Chuyển về thể gốc)
    lemmatizer = WordNetLemmatizer()
    _tokens = [lemmatizer.lemmatize(token) for token in _tokens]

    # Kết hợp lại chuỗi văn bản đã xử lý
    _processed_text = ' '.join(_tokens)
    return _processed_text,_tokens

#Khai báo tên file path để đọc
file_name='book_plates'

# Đọc dữ liệu từ file .txt
with open('En_sample\\'+file_name+".txt", 'r', encoding='utf-8') as file:
    text_data = file.read()
result=process_text(text_data)
processed_text=result[0]
tokens=result[1]

print(f"filtered tokens:\n {tokens}")
# In kết quả
print(f"Văn bản sau khi xử lý: {processed_text}")

#Ghi kết quả ra file
with open('Processed Results\\'+file_name+"_preprocessed.txt", 'w', encoding='utf-8') as file:
    file.write(processed_text)
