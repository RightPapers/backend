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

    def generate_summary(self):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "너는 긴 유튜브 영상을 요약하는 모델이야. 5문장으로 요약하고, 첫번째 문장은 영상 전체를 핵심을 담아 짧게 구성해줘. 4문장은 영상의 상세 내용을 문장 형태로 요약해줘"},
                {"role": "user", "content": "다음 내용을 요약문으로 작성해줘."},
                {"role": "user", "content": f"{self.caption}"}
            ]
        )
        
        return completion.choices[0].message.content

    
    