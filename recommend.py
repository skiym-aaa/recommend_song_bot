# ライブラリのインポート
import pandas as pd
import numpy as np
import csv
import os
import glob
import pathlib
import re
import codecs

# テキストクリーニング用ライブラリ
import mojimoji

# MeCab
import MeCab

# 形態素解析ライブラリ
import janome
import jaconv

# TF-IDF、コサイン類似度計算用ライブラリ
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# データの読み込み
df = pd.read_csv('AkimotoYasushi_all.csv', index_col=0)

# テキストクリーニング
def clean_text(text):
    text = mojimoji.han_to_zen(text, digit=False, ascii=False)
    text = mojimoji.zen_to_han(text, kana=False)
    text = text.lower()
    return text

# MeCabで単語を分かち書き
def text2wakati(text):
    # 取り出したい品詞
    select_conditions = ['動詞', '形容詞', '名詞']

    # 分かち書きオブジェクト
    tagger = MeCab.Tagger()

    # Neologdの指定版 最新語に対応する
    # tagger = MeCab.Tagger('-d /usr/lib64/mecab/dic/mecab-ipadic-neologd')

    # 安定するらしい
    tagger.parse('')

    # 分けてノードごとにする
    node = tagger.parseToNode(text)
    terms = []

    while node:

        # 単語
        term = node.surface

        # 品詞
        pos = node.feature.split(',')[0]

        # もし品詞が条件と一致してたら
        if pos in select_conditions:
            terms.append(term)

        node = node.next

    # 連結おじさん
    text_result = ' '.join(terms)
    return text_result


# コーパスの作成
def create_title_corpus():
    # タイトルのコーパスの作成
    title_corpus = []

    # Song_Titleを抽出してリストを作成
    for text in df.values[:, 0]:
        text_list = text.split(' ')
        for wakachi in text_list:
              text = clean_text(wakachi)
              text = text2wakati(wakachi)
              title_corpus.append(text)
    return title_corpus

def create_content_corpus():
    # 歌詞のコーパスの作成
    content_corpus = []

    # Lyricを抽出してリストを作成
    for text in df.values[:, 4]:
        text_list = text.split(' ')
        for wakachi in text_list:
              text = clean_text(wakachi)
              text = text2wakati(wakachi)
              content_corpus.append(text)

    return content_corpus

# Doc2Vecモデルの構築
def doc2vec(content_corpus):
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(content_corpus)]
    model = Doc2Vec(documents, dm =1, vector_size=300, window=5, min_count=1, workers=4)
    return model

# Doc2Vec用にクエリを分かち書き
def tokenize(text):
    wakati = MeCab.Tagger("-Owakati")
    wakati.parse("")
    return wakati.parse(text).strip().split()

# コーパス
title_corpus = create_title_corpus()
content_corpus = create_content_corpus()
# Doc2Vecモデル
model = doc2vec(content_corpus)

# 検索
def recommend(query):
    # 類似度の計算
    ### TF-IDFの計算 ###
    # タイトル用
    title_vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    title_transformer = TfidfTransformer()
    # 歌詞用
    content_vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    content_transformer = TfidfTransformer()

    # タイトルのTF-IDFを計算
    title_tf = title_vectorizer.fit_transform(title_corpus) # 単語の出現頻度を計算
    title_tfidf = title_transformer.fit_transform(title_tf) # 各タイトルのtfidfを計算

    # 歌詞のTF-IDFを計算
    content_tf = content_vectorizer.fit_transform(content_corpus) # 単語の出現頻度を計算
    content_tfidf = content_transformer.fit_transform(content_tf) # 各歌詞のtfidfを計算

    ### クエリ ###
    # 形態素解析
    tagger = MeCab.Tagger('-Owakati')
    # 分かち書きしたものをリストに入れて渡す
    title_query_tf = title_vectorizer.transform([tagger.parse(query).strip()])
    content_query_tf = content_vectorizer.transform([tagger.parse(query).strip()])
    # クエリのTF-IDFを計算する
    title_query_tfidf = title_transformer.transform(title_query_tf)
    content_query_tfidf = content_transformer.transform(content_query_tf)

    ### Doc2Vecの類似度を算出 ###
    doc_similarity = model.docvecs.most_similar([model.infer_vector(tokenize(query))], topn = len(content_corpus))
    # 昇順にソート
    doc_similarity.sort()
    # 昇順にソートしたDoc2Vecをリスト化
    sort_doc_similarity = []
    for i in range (len(doc_similarity)):
         sort_doc_similarity.append(doc_similarity[i][1])
    # Doc2Vecの類似度
    sort_doc_similarity = np.array(sort_doc_similarity)

    # タイトルのコサイン類似度の計算
    title_similarity = cosine_similarity(title_query_tfidf, title_tfidf)[0]
    # 歌詞のコサイン類似度の計算
    content_similarity = cosine_similarity(content_query_tfidf, content_tfidf)[0]
    # タイトルと歌詞のコサイン類似度とDoc2Vecの類似度の平均
    similarity = np.mean([title_similarity, content_similarity,sort_doc_similarity], axis=0)

    ### 結果 ###
    # タイトルに類似している楽曲を上から順に5個見つける
    topn_indices = np.argsort(similarity)[::-1][:10]
    return topn_indices