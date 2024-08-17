from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone, timedelta
import os
import sys 
from dotenv import load_dotenv
import pymysql
from sqlalchemy import create_engine

pymysql.install_as_MySQLdb()

load_dotenv()

# 현재 프로젝트 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#필요한 모듈 가져오기
from AI.common.utils import *
from AI.common.inference import model_inference  
from AI.models.gpt import VS_GPT

app = Flask(__name__)

# 환경에 따라 설정 적용
env = os.environ.get('FLASK_ENV', 'development')
if env == 'production':
    app.config.from_object('config.ProductionConfig')
elif env == 'testing':
    app.config.from_object('config.TestingConfig')
else:
    app.config.from_object('config.DevelopmentConfig')

#모델 경로 설정
model_base_path = app.config['MODEL_PATH']
roberta_model_path = os.path.join(model_base_path, 'reberta_ACC_0.9265.pth')
transformer_model_path = os.path.join(model_base_path, 'transformer_ACC_0.9231.pth')

#데이터 베이스 설정

DATABASE_URI = os.getenv('DATABASE_URI')
if not DATABASE_URI:
    raise ValueError("DATABASE_URI 환경 변수가 설정되지 않았습니다.")

engine = create_engine(DATABASE_URI)

try:
    with engine.connect() as connection:
        result = connection.execute("SELECT DATABASE();")
        print("Connected to:", result.fetchone())
except Exception as e:
    print("Error connecting to the database:", e)


app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# JSON 파일 저장 경로 설정
json_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_data')

# 디렉토리가 존재하지 않으면 생성
if not os.path.exists(json_save_path):
    os.makedirs(json_save_path)

db = SQLAlchemy(app)
class Feedback(db.Model):
    __tablename__ = 'FEEDBACK'

    feedback_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    feedback_text = db.Column(db.String(600), nullable=False)
    video_id = db.Column(db.String(255), nullable=False)
    submitted_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone(timedelta(hours=9))))

    def __repr__(self):
        return f'<Feedback {self.feedback_id}>'

def analyze_video(url):
    # Use YouTubeCaptionCrawler to get video details and captions
    crawler = YouTubeCaptionCrawler(url)
    video_id = crawler.get_video_id(url)
    video_details = crawler.get_metadata(video_id)

    if video_details and isinstance(video_details, dict):
        video_details["captions"] = crawler.get_caption()

        # JSON 파일 저장
        json_file_path = os.path.join(json_save_path, f'{video_id}.json')
        crawler.save_to_json(json_file_path)

        # Use model_inference to get the probability
        probability = model_inference(url)

        # Use the GPT model to generate the summary
        gpt = VS_GPT(video_details["captions"])
        summary = gpt.generate_summary()

        related_news = related_articles(summary)

        #json에서 캡션(자막)은 제거
        del video_details["captions"]
        del video_details["hashtags"]
        
        analysis_result = {
            "fake_probability": probability,
            "summary": summary
        }
        
        return {
            "youtube_info": video_details,
            "analysis_result": analysis_result,
            "related_articles": related_news
        }
    else:
        raise ValueError("Failed to retrieve video details")

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if request.content_type != 'application/json':
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({"error": "Invalid URL."}), 400

    try:
        # 모델을 통해 유튜브 URL 분석
        analysis_data = analyze_video(url)

        # 분석 결과 반환
        return jsonify(analysis_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json

    feedback_text = data.get('feedback_text', "")
    video_id = data.get('video_id')

    if not video_id:
        return jsonify({"error": "Invalid feedback data. 'video_id' is required."}), 400

    new_feedback = Feedback(feedback_text=feedback_text, video_id=video_id)
    db.session.add(new_feedback)
    db.session.commit()

    return jsonify({"message": "Feedback received. Thank you!"}), 200

if __name__ == '__main__':
    app.run(debug=True)
