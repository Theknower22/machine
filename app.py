from flask import Flask, request, render_template
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    uploaded_file = request.files['codefile']
    if uploaded_file.filename != '':
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(filepath)

        features = extract_features(filepath)
        result = simple_ai_model(features)

        return render_template('result.html', filename=uploaded_file.filename, result=result)

    return 'no file has been uploaded'

def extract_features(filepath):
    with open(filepath, 'r') as f:
        code = f.read()
    num_lines = code.count('\n')
    num_chars = len(code)
    return [num_lines, num_chars]

def simple_ai_model(features):
    lines, chars = features
    if lines > 50 or chars > 1000:
        return "there might be a vulnerability in your code"
    else:
        return "your code is clean"

if __name__ == '__main__':
    app.run(debug=True)
