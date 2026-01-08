import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import re

# 下载必要的nltk数据
nltk.download('punkt')
nltk.download('punkt_tab')

def preprocess_text(text):
    """文本预处理函数"""
    # 转换为小写
    text = str(text).lower()
    # 移除特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    tokens = word_tokenize(text)
    return tokens

def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 合并标题和评论
    df['text'] = df.iloc[:, 1] + " " + df.iloc[:, 2]
    
    # 预处理所有文本
    corpus = []
    for text in df['text']:
        tokens = preprocess_text(text)
        corpus.append(tokens)
    
    return corpus, df.iloc[:, 0].values  # 返回处理后的文本和标签

def train_word2vec(corpus):
    """训练Word2Vec模型"""
    model = Word2Vec(sentences=corpus,
                    vector_size=100,  # 词向量维度
                    window=5,         # 上下文窗口大小
                    min_count=1,      # 词频阈值
                    workers=4)        # 训练的线程数
    return model

def get_document_vector(text, model):
    """获取文档的词向量表示（取平均）"""
    tokens = preprocess_text(text)
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)


def main():
    # 1. 加载并预处理训练集数据
    train_corpus, train_labels = load_and_preprocess_data('dataset/train.csv')

    # 2. 训练 Word2Vec 模型（只用训练集）
    model = train_word2vec(train_corpus)

    # 3. 获取训练集文档向量
    train_df = pd.read_csv('dataset/train.csv')
    train_texts = train_df.iloc[:, 1] + " " + train_df.iloc[:, 2]
    X_train = np.array([get_document_vector(text, model) for text in train_texts])
    y_train = train_df.iloc[:, 0].values

    print("训练集文档向量形状:", X_train.shape)
    print("训练集标签形状:", y_train.shape)

    # 4. 获取测试集文档向量
    test_df = pd.read_csv('dataset/test.csv')
    test_texts = test_df.iloc[:, 1] + " " + test_df.iloc[:, 2]
    X_test = np.array([get_document_vector(text, model) for text in test_texts])
    y_test = test_df.iloc[:, 0].values

    print("测试集文档向量形状:", X_test.shape)
    print("测试集标签形状:", y_test.shape)

    # 5. 保存模型（可选）
    model.save("word2vec_sentiment.model")

    # 示例：查看某些词的相似词
    word = "great"
    if word in model.wv:
        similar_words = model.wv.most_similar(word)
        print(f"\n与'{word}'最相似的词:")
        for w, score in similar_words:
            print(f"{w}: {score:.4f}")


if __name__ == "__main__":
    main()