import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

import logging

from src.inference_ocr.utils.exceptions import MessageHandlerError

from .. import bootstrap

bootstrap_dict = bootstrap.bootstrap()
message_handler = bootstrap_dict["message_handler"]

log = logging.getLogger("Flask_app")


def run_callback(image):

    try:

        final_results = message_handler(image)
        return jsonify({'results': final_results})
    
    except MessageHandlerError as error:
        log.error("Error while handling message: %s", repr(error))
    except Exception as error:
        log.error("Error: %s", repr(error))


@app.route('/ocr', methods=['POST'])
def ocr():
    image = request.files['image']
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

    return run_callback(image)
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded = False)
