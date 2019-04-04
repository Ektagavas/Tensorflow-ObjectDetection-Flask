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
    # instance = ObjectDetection()
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
    app.run(debug=True, host='172.16.2.164')  # host='172.16.2.164'








# from flask import Flask, send_file
# from flask_restful import Api, Resource, reqparse
# from object_detection_1 import ObjectDetection
#
#
# app = Flask(__name__)
# api = Api(app)
#
# # users =[
# #     {
# #         "name":"abc",
# #         "age":42
# #     },
# #     {
# #         "name":"def",
# #         "age":23
# #     }
# # ]
#
# # class User(Resource):
# #     def get(self, name):
# #         for user in users:
# #             if(name == user["name"]):
# #                 return user, 200
# #         return "User not found", 404
# #
# #     def post(self, name):
# #         parser = reqparse.RequestParser()
# #         parser.add_argument("age")
# #         args = parser.parse_args()
# #
# #         for user in users:
# #             if (name == user["name"]):
# #                 return "User with name {} already exists".format(name), 400
# #
# #         user = {
# #             "name": name,
# #             "age": args["age"]
# #         }
# #         users.append(user)
# #         return user, 201
#
#
# class Res(Resource):
#     def post(self):
#         parser = reqparse.RequestParser()
#         parser.add_argument("imageurl")
#         args = parser.parse_args()
#         instance = ObjectDetection()
#         result, op = instance.detect_objects(args["imageurl"])
#         path = "./output.jpg"
#         if path is None:
#             self.Error(400)
#         try:
#             return send_file(path, as_attachment=True)
#         except Exception as e:
#                 self.log.exception(e)
#                 self.Error(400)
#         return 'Done', 200
#
#
# api.add_resource(Res, "/")
#
# # api.add_resource(User, "/user/<string:name>")
#
# # @app.route("/hello")
# # def hello():
# #     return "Hello World!"
# #
# #
# # if __name__ == '__main__':
# app.run(debug=True)





