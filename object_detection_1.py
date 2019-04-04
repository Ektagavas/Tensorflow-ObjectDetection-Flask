# Import Libraries
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import matplotlib.pyplot as plt

from urllib import request as request


from PIL import Image as Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont


class ObjectDetection:
    """
    class for Object Detection
    """

    image_url = ""  # Image url
    img_localpath = "input.jpg"  # Path where image will be downloaded locally
    img_width, img_height = 1600, 1280


    """
    Build graph in constructor
    """
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Create a placeholder
            self.image_raw_placeholder = tf.placeholder(tf.string, name="image_raw_placeholder")
            self.decoded_image = tf.image.decode_jpeg(self.image_raw_placeholder, channels=3, name="decoded_image")
            print('decoded image')
            self.resized_image = tf.image.resize_images(self.decoded_image, [self.img_width, self.img_height])
            self.image_tensor1 = tf.image.convert_image_dtype(self.resized_image, dtype=tf.float32)
            self.image_tensor2 = tf.cast(self.image_tensor1, tf.float32) * (1. / 255)
            print('resized images')
            self.image_tensor = tf.reshape(self.image_tensor2, [1, self.img_width, self.img_height, 3])
            print('reshaped images')

            print('Downloading.........')
            # TFHub Module
            ObjectDetection.detector = hub.Module("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")

            print('detector')


            self.detector_output = ObjectDetection.detector(self.image_tensor, as_dict=True)
            print('run detector')

            init_ops = [tf.initialize_all_variables(), tf.tables_initializer()]


    """
    Download image and read it as raw
    # returns: 
    raw_image: raw form of image data
    """

    def get_raw_image(self):
        self.download_image()
        image_file = open(self.img_localpath, 'rb')
        image_raw = image_file.read()
        return image_raw

    """
    Download image from url and save it locally
    """

    def download_image(self):
        response = request.urlretrieve(self.image_url, self.img_localpath)

    """
    Display image
    # Arguments
    image: image to display
    """

    def display_image(self, image):
        fig = plt.figure(figsize=(20, 15))
        plt.grid(False)
        plt.imshow(image)
        plt.imsave("output.jpg", image)

    """
    Main method for object detection
    Create a graph and perform object detection on downloaded image and display image with object labels
    # Arguments
    image_url: url of image on which object detection is to be done
  
    # Returns
    result: result of object detector
    ops: output of decoded image
    """

    def detect_objects(self, image_url):
        print('Set image url')
        self.image_url = image_url

        # Create session
        sess = tf.Session(graph=self.graph)

        # Initialize session
        sess.run([self.graph.get_operation_by_name('init'), self.graph.get_operation_by_name('init_all_tables')])

        # Get raw image
        image_raw = self.get_raw_image()
        print('got raw image')

        print('run module')

        # Feed image raw data to module
        result, op = sess.run([self.detector_output, self.decoded_image], feed_dict={self.image_raw_placeholder: image_raw})
        print("Found %d objects." % len(result["detection_scores"]))

        sess.close()


        # Draw boxes
        image_with_boxes = self.draw_boxes(np.array(op), result["detection_boxes"], result["detection_class_entities"],result["detection_scores"])

        # Display image
        self.display_image(image_with_boxes)
        return result, op



    """
    Helper functions to draw boxes around objects with labels on image
    """

    def draw_boxes(self, image, boxes, class_names, scores, max_boxes=20, min_score=0.5):
        """Overlay labeled boxes on an image with formatted scores and label names."""
        colors = list(ImageColor.colormap.values())

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except IOError:
            print("Font not found, using default font.")
            font = ImageFont.load_default()

        for i in range(min(boxes.shape[0], max_boxes)):
            if scores[i] >= min_score:
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                               int(100 * scores[i]))
                color = colors[hash(class_names[i]) % len(colors)]
                image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
                self.draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font,
                                                display_str_list=[display_str])
                np.copyto(image, np.array(image_pil))
        return image

    def draw_bounding_box_on_image(self, image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
        """Adds a bounding box to an image."""
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=thickness,
                  fill=color)

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height
        # Reverse list and print from bottom to top.
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                            (left + text_width, text_bottom)],
                           fill=color)
            draw.text((left + margin, text_bottom - text_height - margin),
                      display_str,
                      fill="black",
                      font=font)
            text_bottom -= text_height - 2 * margin


# image_url = "https://farm3.staticflickr.com/7621/16236302203_35f0ded2cc_o.jpg"
#
# instance = ObjectDetection()
# result, op = instance.detect_objects(image_url)
# print('Hello')

"""### Caching Modules
When creating a module from a URL, the module content is downloaded and cached in the local system temporary directory. The location where modules are cached can be overridden using TFHUB_CACHE_DIR environment variable.

For example, setting TFHUB_CACHE_DIR to /my_module_cache:

$ export TFHUB_CACHE_DIR=/hub_module_cache
and then creating a module from a URL:

m = hub.Module("https://tfhub.dev/google/progan-128/1")
results in downloading and unpacking the module into /my_module_cache.
"""