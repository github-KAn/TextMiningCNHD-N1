import spacy,regex,re,string,nltk,pickle,os,gensim
from spacy.lang.vi import Vietnamese
from spacy.lang.en import English
from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import numpy as np
def remove_punctuation(data):
    punctuation_free="".join([i for i in data if i not in string.punctuation])
    return punctuation_free
def process_data(data):
    # Đối tượng "nlp"  được dùng để tạo documents với chú thích ngôn ngữ (linguistic annotations).
    nlp_vi = Vietnamese()
    data = remove_punctuation(data)
    data = re.sub("[-\n]", " ", data)

    my_doc = nlp_vi(data)

    # Tạo danh sách tokens từ
    token_list = []
    for token in my_doc:
        token_list.append(token.text)

    # Tạo danh sách tokens sau khi loại bỏ từ dùng
    filtered_words = []
    for word in token_list:
        # print(word)
        lexeme_vi = nlp_vi.vocab[word]
        if (not (lexeme_vi.is_stop == True or word.isspace() or (word in string.punctuation))):
            filtered_words.append(word)
        processed_text = ' '.join(filtered_words)
    return filtered_words

dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'TextMiningCNHD-N1')

def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-8") as f:
                # text=f.read()
                # X.append(process_data(text))
                # y.append(path)
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)

                X.append(lines)
                y.append(path)
    return X, y

train_path = os.path.join(dir_path, 'VNTC-master/Data/10Topics/Ver1.1/Train_Full')
X_data, y_data = get_data(train_path)

pickle.dump(X_data, open('data/X_data.pkl', 'wb'))
pickle.dump(y_data, open('data/y_data.pkl', 'wb'))

test_path = os.path.join(dir_path, 'VNTC-master/Data/10Topics/Ver1.1/Test_Full')
X_test, y_test = get_data(test_path)

pickle.dump(X_test, open('data/X_test.pkl', 'wb'))
pickle.dump(y_test, open('data/y_test.pkl', 'wb'))




