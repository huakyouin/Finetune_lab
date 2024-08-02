'''
用于在RAG中对文本进行分片的函数，向量化需额外实现
'''

import re
import numpy as np


def split_text_to_sentences(text,split_chars=".?!。\n！："):
    ''' 按分割符切割文本成句子'''
    pattern = re.compile(f'(?<=[{split_chars}])\s+')
    single_sentences_list = pattern.split(text)

    # 去除列表中的空字符串项，并记录每个句子在原文中的起始和结束位置
    start_idx = 0

    sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list)]

    for item in sentences:
        item['start_idx'] = start_idx
        start_idx += len(item['sentence'])
        # 跳过分隔符长度
        while start_idx < len(text) and text[start_idx] in split_chars:
            start_idx += 1
    
    return sentences
    

def combine_sentences(sentences, buffer_size=1):
    '''按窗口大小合并句子 '''
    combined_sentences = [
        ' '.join(sentences[j]['sentence'] for j in range(max(i - buffer_size, 0), min(i + buffer_size + 1, len(sentences))))
        for i in range(len(sentences))
    ]   
    # 更新原始字典列表，添加组合后的句子
    for i, combined_sentence in enumerate(combined_sentences):
        sentences[i]['combined_sentence'] = combined_sentence

    return sentences

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def calculate_cosine_distances(sentence_embeddings):
    """
    根据前后句向量的余弦相似度生成距离
    Input:
        sentence_embeddings: 句子的embedding向量构成的列表
    """
    distances = []
    for i in range(len(sentence_embeddings) - 1):
        embedding_current = sentence_embeddings[i]
        embedding_next = sentence_embeddings[i + 1]
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding_current, embedding_next)
        # Convert to cosine distance
        distance = 1 - similarity
        distances.append(distance)
    return distances

def build_chunks(text, sentences,distances,breakpoint_percentile_threshold=95):
    '''
    根据距离把文本分片
    Input:
        text:       原文
        sentences:  句子列表
        distances:  跟下句的相似距离
        breakpoint_percentile_threshold：   距离分割阈值分位数（0-100）
    '''
    if len(sentences)<=1: # 全文被分为一句话
        return [{"start_idx":sentences[0]["start_idx"], "text": text}]

    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

    chunks = []
    start_sent_idx = 0
    for index in indices_above_thresh:
        # The end index is the current breakpoint
        last_sent_idx = index

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_sent_idx:last_sent_idx + 1]
        combined_text = text[sentences[start_sent_idx]["start_idx"]: sentences[last_sent_idx]["start_idx"]+len(sentences[last_sent_idx]["sentence"])]
        chunk = {"start_idx":group[0]["start_idx"], "text":combined_text}
        chunks.append(chunk)
        
        # Update the start index for the next group
        start_sent_idx = index + 1

    # 处理剩余句子
    if start_sent_idx < len(sentences):
        group = sentences[start_sent_idx:]
        combined_text = text[sentences[start_sent_idx]["start_idx"]: sentences[-1]["start_idx"]+len(sentences[-1]["sentence"])]
        chunk = {"start_idx":group[0]["start_idx"], "text":combined_text}
        chunks.append(chunk)
    return chunks