from flask import Flask, render_template, request, jsonify
# import voice_to_chatgpt
import assistant

app = Flask('__name__')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_text_prompt', methods=['POST'])
def process_text_prompt():
    user_prompt = request.form['user_prompt']
    response = assistant.process_text_prompt(user_prompt)
    return jsonify(response)

@app.route('/process_voice_command', methods=['POST'])
def process_voice_command():
    response = assistant.process_voice_command()
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
