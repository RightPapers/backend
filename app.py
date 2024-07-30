from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# 환경에 따라 설정 적용
env = os.environ.get('FLASK_ENV', 'development')
if env == 'production':
    app.config.from_object('config.ProductionConfig')
elif env == 'testing':
    app.config.from_object('config.TestingConfig')
else:
    app.config.from_object('config.DevelopmentConfig')

#모델 불러오기
model_base_path = app.config['MODEL_PATH']
gpt_model_path = os.path.join(model_base_path, 'gpt')
roberta_model_path = os.path.join(model_base_path, 'roberta')
transformer_model_path = os.path.join(model_base_path, 'transformer')

model = tf.keras.models.load_model(transformer_model_path)

@app.route('/api/analyze/<youtube_id>', methods=['POST'])
def analyze(youtube_id):
    # youtube_id 유효성 검사
    if not youtube_id:
        return jsonify({"error": "Invalid YouTube ID."}), 400
    
    # 유튜브 ID로부터 정보 추출 (예시 데이터 사용)
    youtube_info = {
        "id": youtube_id,
        "thumbnail": f"https://img.youtube.com/vi/{youtube_id}/0.jpg",
        "title": "Example Video Title",
        "channel_name": "Example Channel"
    }
    
    # 분석 결과 (예시 데이터 사용)
    analysis_result = {
        "accuracy": 95.5,
        "summary": "This is a summary of the video content."
    }
    
    # 관련 기사 (예시 데이터 사용)
    related_articles = [
        {
            "source": "Example News",
            "upload_time": "2024-07-28T10:00:00Z",
            "title": "Example Article Title",
            "link": "https://www.examplenews.com/article/example"
        }
    ]
    
    response = {
        "youtube_info": youtube_info,
        "analysis_result": analysis_result,
        "related_articles": related_articles
    }
    
    return jsonify(response), 200

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    was_helpful = data.get('was_helpful')
    feedback = data.get('feedback', "")
    
    if was_helpful is None or (was_helpful is False and not feedback):
        return jsonify({"error": "Invalid feedback data."}), 400
    
    # 피드백 저장 또는 처리 (예시로 단순 메시지 반환)
    return jsonify({"message": "Feedback received. Thank you!"}), 200

@app.route('/api/main', methods=['GET'])
def main():
    try:
        # 메인 화면 데이터 생성 (예시 데이터 사용)
        main_content = "Welcome to the main page!"
        highlights = [
            {"title": "Highlight 1", "description": "Description of highlight 1"},
            {"title": "Highlight 2", "description": "Description of highlight 2"}
        ]
        
        response = {
            "main_content": main_content,
            "highlights": highlights
        }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": "Failed to load main content."}), 500

@app.route('/api/help', methods=['GET'])
def help():
    try:
        # 도움말 화면 데이터 생성 (예시 데이터 사용)
        help_content = "This is the help page content."
        faq = [
            {"question": "How to use this service?", "answer": "You can use this service by doing X, Y, and Z."},
            {"question": "Who can I contact for support?", "answer": "You can contact support at support@example.com."}
        ]
        
        response = {
            "help_content": help_content,
            "faq": faq
        }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": "Failed to load help content."}), 500

if __name__ == '__main__':
    app.run(debug=True)
