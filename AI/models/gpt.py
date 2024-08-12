# coding : utf-8
import json

from openai import OpenAI
from AI.common.keys import my_keys
from AI.common.utils import YouTubeCaptionCrawler
#from textrank.keysentences import textrank_keysentences

# OpenAI GPT for Video Summarization
class VS_GPT:
    def __init__(self, caption, model='gpt-3.5-turbo'):
        self.api_key = my_keys('openai')
        self.client = OpenAI(api_key=self.api_key)
        self.caption = caption  # video url
        self.model = model  # default: gpt-3.5-turbo / possible: gpt-4-turbo, gpt-4o-mini
    
    #def _create_prompt(self):
        # # TextRank로 중요 문장 추출
        # summary_data = textrank_keysentences(self.url)
        
        # prompt = ""
        # for i, line in enumerate(summary_data):
        #     sentence_info = f'{i+1}번째 문장의 중요도는 {line["importance"]}이며, 내용은 "{line["sentence"]}"입니다.\n'
        #     prompt += sentence_info
        
        #return prompt
    
    def generate_summary(self):
        #prompt = self._create_prompt()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "너는 긴 유튜브 영상을 5개의 문장으로 요약하는 모델이야."},
                {"role": "user", "content": "다음 내용을 요약문으로 작성해줘."},
                {"role": "user", "content": f"{self.caption}"}
            ]
        )
        
        return completion.choices[0].message.content

    
    