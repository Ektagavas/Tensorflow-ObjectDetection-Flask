from flask import Flask, render_template
from object_detection_1 import ObjectDetection

app = Flask(__name__)


# image_url = "https://farm3.staticflickr.com/7621/16236302203_35f0ded2cc_o.jpg"
# instance = ObjectDetection()
# result, op = instance.detect_objects(image_url)
#
# image_url = "https://farm1.staticflickr.com/4032/4653948754_c0d768086b_o.jpg"
# result, op = instance.detect_objects(image_url)

instance = None


@app.before_first_request
def do_something_only_once():
    global instance
    instance = ObjectDetection()

@app.route('/')
def index():
    # instance = ObjectDetection()
    print('index')
    # global instance
    # instance = ObjectDetection()
    return render_template('index.html')


@app.route('/image', methods=['POST'])
def result():
    print('result')
    c = instance.detect_objects("")
    print(c)
    return "Done", 200


if __name__ == '__main__':
    print('in main')
    app.run(debug=True)  # host='172.16.2.164'