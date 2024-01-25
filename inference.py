import argparse
from transformers import AutoConfig, AutoModel, AutoTokenizer,pipeline
import torch
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import dot
from numpy.linalg import norm
#爬虫
import requests
from bs4 import BeautifulSoup
#向量数据库
import chromadb


parser = argparse.ArgumentParser()
parser.add_argument("--pt-checkpoint", type=str, default=None, help="The checkpoint path")
parser.add_argument("--model", type=str, default=None, help="main model weights")
parser.add_argument("--tokenizer", type=str, default=None, help="main model weights")
parser.add_argument("--pt-pre-seq-len", type=int, default=128, help="The pre-seq-len used in p-tuning")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max-new-tokens", type=int, default=128)


model_m3e = SentenceTransformer('moka-ai/m3e-base')
args = parser.parse_args()

if args.tokenizer is None:
    args.tokenizer = args.model

if args.pt_checkpoint:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, pre_seq_len=args.pt_pre_seq_len)
    model = AutoModel.from_pretrained(args.model, config=config, trust_remote_code=True).cuda()
    prefix_state_dict = torch.load(os.path.join(args.pt_checkpoint, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)

model = model.to(args.device)


while True:
    
    max_similarity = 0
    # 待检测新闻
    det_article = "LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in \"Harry Potter and the Order of the Phoenix\" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. \"I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar,\" he told an Australian interviewer earlier this month. \"I don't think I'll be particularly extravagant. \"The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs.\" At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film \"Hostel: Part II,\" currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. \"I'll definitely have some sort of party,\" he said in an interview. \"Hopefully none of you will be reading about it.\" Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. \"People are always looking to say 'kid star goes off the rails,'\" he told reporters last month. \"But I try very hard not to go that way because it would be too easy for them.\" His latest outing as the boy wizard in \"Harry Potter and the Order of the Phoenix\" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films.  Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called \"My Boy Jack,\" about author Rudyard Kipling and his son, due for release later this year. He will also appear in \"December Boys,\" an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's \"Equus.\" Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: \"I just think I'm going to be more sort of fair game,\" he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed."
    # 获取关键字
    inputs = tokenizer("Take three keywords from the following news story.The output should consist of only three keywords, not any other extraneous information.news:"+det_article, return_tensors="pt")
    inputs = inputs.to(args.device)
    response = model.generate(input_ids=inputs["input_ids"], max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens)
    response = response[0, inputs["input_ids"].shape[-1]:]
    keyword = tokenizer.decode(response, skip_special_tokens=True)[10:]
    print("Keyword:"+keyword)
    print("---------------------------------------------------------------------------------------")
    #获取待检测新闻摘要
    inputs = tokenizer("Summarise the content of the news after # in one sentence, just output only the summary, not any other irrelevant information.#"+det_article, return_tensors="pt")
    inputs = inputs.to(args.device)
    response = model.generate(input_ids=inputs["input_ids"], max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens)
    response = response[0, inputs["input_ids"].shape[-1]:]
    det_summary = tokenizer.decode(response, skip_special_tokens=True)[1:]
    print("Summary:"+det_summary)
    print("---------------------------------------------------------------------------------------")
    #爬虫
    theme = keyword
    page = "1"
    start = "0"
    end = "3"
    url = "https://search.prod.di.api.cnn.io/content?q="+theme+"&size="+end+"&from="+start+"&page="+page+"&sort=relevance&types=article"

    headers = {

        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }
    count = 0
    response = requests.get(url,headers=headers)
    jsons = response.json()
    data_list = jsons['result']
    num = len(data_list)
    crawl = []
    final_paragraphs=[]
    for i in range(0,num):
        url = data_list[i]['path']
        res = requests.get(url, headers)
        soup = BeautifulSoup(res.content,"lxml")
        title = soup.find(name="div",attrs={"class":"headline__wrapper"})  #标题
        if title == None:
            continue   #不是一篇规范文章
        else:
            title =title.text.replace("\n","")
        body = soup.find(name="div",attrs={"class":"article__content"})     #新闻内容
        content = body.find_all(name="p",attrs={"class":"paragraph inline-placeholder"})
        article = ""
        paragraphs = []
        for paragraph in content:
            p = paragraph.text.strip()
            paragraphs.append(p)
            article = article + p
        #情绪分析
        if count == 0:
            prompt = "You are a psychologist who specialises in analysing the emotions embedded in the news and your task is to conduct an emotional analysis based on the two news stories provided below, comparing and contrasting which one is more emotional:real_news#"+article+"det_news#"+det_article
            response, history = model.chat(tokenizer, prompt, history=[])
            emotion = response
            
        print("emotion:"+emotion)
        print("---------------------------------------------------------------------------------------")
        # 获取爬虫爬下来的真实新闻摘要
        inputs = tokenizer("Summarise the content of the news after # in one sentence, just output only the summary, not any other irrelevant information.#"+article, return_tensors="pt")
        inputs = inputs.to(args.device)
        response = model.generate(input_ids=inputs["input_ids"], max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens)
        response = response[0, inputs["input_ids"].shape[-1]:]
        summary = tokenizer.decode(response, skip_special_tokens=True)
        print("crawl_news_summary{}:".format(count)+summary)
        print("---------------------------------------------------------------------------------------")
        # 相似性比较 判断是否为描述同一件事情的新闻
        embeddings1 = model_m3e.encode(det_summary)
        embeddings2 = model_m3e.encode(summary)
        similarity = dot(embeddings1, embeddings2)/(norm(embeddings1)*norm(embeddings2))
        print("similarity{}:".format(count)+str(similarity))
        max_similarity = max(max_similarity,similarity)
        print("---------------------------------------------------------------------------------------")
        if similarity>0.7:#描述同一件事情的新闻
            for item in paragraphs:
                final_paragraphs.append(item)
        count+=1
    #向量数据库
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="my_collection")
    collection.add(
        documents=final_paragraphs,
        ids=[f"id{i}" for i in range(len(final_paragraphs))] )
    #幻觉检测 看看有没有证据：
    sentences_list = det_article.split('.')
    sentences_list = [sentence.strip() for sentence in sentences_list if sentence]  
    cnt = 0
    for sentence in sentences_list:
        results = collection.query(
            query_texts=[sentence],
            n_results=1)
        embeddings1 = model_m3e.encode(sentence)
        embeddings2 = model_m3e.encode(results["documents"][0][0])
        similarity = dot(embeddings1, embeddings2)/(norm(embeddings1)*norm(embeddings2))
        if similarity<0.7:
            cnt+=1
    hallucination = "There is no basis for the presence of "+str(cnt/len(sentences_list)*100)+"% in the news to be detected"
    print(hallucination)
    print("---------------------------------------------------------------------------------------")
    #final_prompt="你作为文本分类器，需要基于下述相似性以及情绪分析将输入的文本分为“human written”、“ai generated”中的一种。在输出时，只输出类别，不要输出其它任何无关的信息。#similarity:"+hallucination+"#emotion:"+emotion+"question:"+det_article
    final_prompt="As a text classifier, you need to classify the input text into one of 'human written', 'ai generated' based on the following similarity and sentiment analysis. In the output, only the category is output, not any other irrelevant information.#similarity:"+hallucination+"#emotion:"+emotion+"question:"+det_article
    print(final_prompt)
    print("---------------------------------------------------------------------------------------")
    inputs = tokenizer(final_prompt, return_tensors="pt")
    inputs = inputs.to(args.device)
    response = model.generate(input_ids=inputs["input_ids"], max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens)
    response = response[0, inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)
    print(answer)
    break
    
        
        
    