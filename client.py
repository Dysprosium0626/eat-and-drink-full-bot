from flask import Flask, request, jsonify
from interact import *
import logging


logging.basicConfig(filename='chat.log', filemode='w', format="%(asctime)s|%(levelname)s|%(filename)s:%(lineno)s|%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)

app = Flask(__name__)
args = set_args()
inference = Inference(args.model_dir, args.device, args.max_history_len, args.max_len,
                      args.repetition_penalty, args.temperature)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('query')
        logging.info(f"query: {query}")
        if query.strip() in ('break', 'end'):
            raise ValueError("exit")

        text = inference.predict(query)
        response = {
            'bot': text
        }
        logging.info(f"response: {text}")
        return jsonify(response)
    except (ValueError, EOFError):
        return jsonify({'error': 'An error occurred during the conversation.'}), 500


if __name__ == '__main__':
    app.run()
