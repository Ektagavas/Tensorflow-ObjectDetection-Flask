##### export TFHUB_CACHE_DIR=/hub_module_cache

from flask import Flask, render_template, request, send_file
from object_detection_1 import ObjectDetection


app = Flask(__name__)

instance = None


@app.before_first_request
def do_something_only_once():
    global instance
    instance = ObjectDetection()

@app.route('/')
def index():
    print('index')
    return render_template('index.html')


@app.route('/image', methods=['POST'])
def result():
    print('result')
    # instance = ObjectDetection()
    image = request.form['imageurl']
    result, op = instance.detect_objects(image)
    path = "./output.jpg"

    try:
        return send_file(path, as_attachment=True)
    except Exception as e:
        return '', 400
    return 'Success', 200




if __name__ == '__main__':
    print('in main')
    app.run(debug=True, host='172.16.2.164')  # Server ip
