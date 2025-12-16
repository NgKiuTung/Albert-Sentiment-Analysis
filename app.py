from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer
from nets import ALBERT
import time
import numpy as np

app = Flask(__name__)
CORS(app)

# 配置
MODEL_PATH = "models/albert_best.pth"
PRETRAINED_MODEL = "albert-chinese-small"
MAX_LENGTH = 200
NUM_CLASSES = 7

# 情感标签
LABEL_NAMES = ['like', 'sadness', 'fear', 'anger', 'disgust', 'happiness', 'surprise']
LABEL_NAMES_ZH = ['喜欢', '悲伤', '恐惧', '愤怒', '厌恶', '快乐', '惊讶']
LABEL_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
LABEL_EMOJIS = ['👍', '😢', '😨', '😠', '🤢', '😄', '😲']

# 加载模型
print("正在加载模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
model = ALBERT(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.to(device)
model.eval()
print(f"模型加载完成！使用设备: {device}")

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        # 获取输入文本
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({
                'success': False,
                'error': '请输入文本内容'
            }), 400

        # 记录开始时间
        start_time = time.time()

        # 文本预处理
        info_dict = tokenizer(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = info_dict["input_ids"].to(device)
        attention_mask = info_dict["attention_mask"].to(device)

        # 模型推理
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        # 计算推理时间
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒

        # 准备返回数据
        probabilities_list = probabilities.cpu().numpy().tolist()

        # 构建详细结果
        emotions = []
        for i, (label_en, label_zh, prob, color, emoji) in enumerate(zip(
            LABEL_NAMES, LABEL_NAMES_ZH, probabilities_list, LABEL_COLORS, LABEL_EMOJIS
        )):
            emotions.append({
                'label_en': label_en,
                'label_zh': label_zh,
                'probability': round(prob * 100, 2),
                'color': color,
                'emoji': emoji,
                'is_predicted': i == predicted_class
            })

        # 排序（按概率从高到低）
        emotions_sorted = sorted(emotions, key=lambda x: x['probability'], reverse=True)

        # 计算统计信息
        tokens = tokenizer.tokenize(text)
        token_count = len(tokens)
        valid_token_count = attention_mask.sum().item()

        return jsonify({
            'success': True,
            'result': {
                'text': text,
                'predicted_emotion': {
                    'label_en': LABEL_NAMES[predicted_class],
                    'label_zh': LABEL_NAMES_ZH[predicted_class],
                    'emoji': LABEL_EMOJIS[predicted_class],
                    'color': LABEL_COLORS[predicted_class]
                },
                'confidence': round(confidence * 100, 2),
                'emotions': emotions_sorted,
                'statistics': {
                    'inference_time': round(inference_time, 2),
                    'text_length': len(text),
                    'token_count': token_count,
                    'valid_token_count': valid_token_count,
                    'device': str(device),
                    'model': 'ALBERT-Chinese-Small'
                }
            }
        })

    except Exception as e:
        print(f"预测错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'预测失败: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model': 'ALBERT',
        'device': str(device),
        'num_classes': NUM_CLASSES
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
