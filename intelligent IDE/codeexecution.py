from flask import Flask, request, jsonify
import subprocess

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app
# app = Flask(__name__)

@app.route('/code/submit', methods=['POST'])
def submit_code():
    user_code = request.json['code']
    output, errors = execute_code(user_code)
    store_errors(user_code, errors)
    return jsonify({'output': output, 'errors': errors})

def execute_code(code):
    try:
        # Execute the user's code in a sandboxed environment using exec()
        # IMPORTANT: Be cautious when using exec() as it can execute arbitrary code.
        # Consider implementing additional security measures like code analysis and filtering.
        exec(code, globals())
        return '', ''
    except Exception as e:
        # Capture any errors that occur during execution
        error_message = str(e)
        return '', error_message

def store_errors(code, errors):
    # Store error feedback data in a database for learning purposes
    pass

if __name__ == '__main__':
    app.run(debug=True)
