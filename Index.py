# pip install nltk spacy sklearn textblob gensim
import nltk,pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import ssl
from spacy.lang.vi.stop_words import STOP_WORDS

# Bỏ qua chứng chỉ SSL để tránh lỗi tải xuống
ssl._create_default_https_context = ssl._create_unverified_context

# Tải các công cụ cần thiết cho NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Đọc dữ liệu từ file .txt
with open('sample_text_vi.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

# 1. Chuyển thành chữ thường
text_data = text_data.lower()

# 2. Tokenization (Tách từ)
tokens = word_tokenize(text_data)

# 3. Loại bỏ từ dừng (Stopwords)
tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
# tokens = [word for word in tokens if word.isalpha() and word not in STOP_WORDS]
# 4. Lemmatization (Chuyển về thể gốc)
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(token) for token in tokens]

# Kết hợp lại chuỗi văn bản đã xử lý
processed_text = ' '.join(tokens)

# 5. Phân tích cảm xúc (Sentiment Analysis) với TextBlob
blob = TextBlob(processed_text)
sentiment = blob.sentiment

# In kết quả
print(f"Văn bản sau khi xử lý: {processed_text}")
print(f"Phân tích cảm xúc: {sentiment}")

#Ghi kết quả ra file
with open('processed_text.txt', 'w', encoding='utf-8') as file:
    file.write(processed_text)
pickle.dump(tokens,open("data/filtered_words_en.pkl","wb"))