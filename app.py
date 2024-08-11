from flask import Flask, request, jsonify
import os
import sys

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


# JSON 파일 저장 경로 설정
json_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_data')

# 디렉토리가 존재하지 않으면 생성
if not os.path.exists(json_save_path):
    os.makedirs(json_save_path)

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
        add_probability_to_json(json_file_path, probability)

        # Use the GPT model to generate the summary
        gpt = VS_GPT(file_path=json_file_path)
        summary = gpt.generate_summary()

        analysis_result = {
            "fake_probability": probability,
            "summary": summary
        }
        
        return {
            "youtube_info": video_details,
            "analysis_result": analysis_result,
            "json_file_path": json_file_path,  # JSON 파일 경로 추가
            "related_articles": []  # Placeholder for future implementation
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
    was_helpful = data.get('was_helpful')
    feedback = data.get('feedback', "")
    
    if was_helpful is None or (was_helpful is False and not feedback):
        return jsonify({"error": "Invalid feedback data."}), 400
    
    # 피드백 저장 또는 처리 (예시로 단순 메시지 반환)
    return jsonify({"message": "소중한 피드백 감사합니다!"}), 200

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
