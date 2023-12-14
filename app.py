from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

app = Flask(__name__)

class_mapping = {0: 'Bottle', 1: 'Can',2:"Cardboard", 3:'Shoes', 4:"Textile"}
# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

def image_preprocess(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    array_image = img_to_array(img)
    array_image = np.expand_dims(array_image, axis=0)
    array_image /= 255.
    return array_image

def predict_single_image(image_path, model, target_size=(225, 225)):
    # Load and preprocess the single image
    img_array = image_preprocess(image_path, target_size)

    # Make prediction
    pred_array = model.predict(img_array, batch_size=1, verbose=1)

    # Get the predicted class index
    results_ = np.argmax(pred_array, axis=1)

    # Get the probability scores for all classes
    probability_scores = pred_array[0]

    return results_, probability_scores



def waste_classifier(img_path_,model):
    result_,probability_scores = predict_single_image(img_path_, model,(224, 224) )
    # Assuming class_mapping is a dictionary mapping class indices to labels
    class_mapping = {0: 'Bottle', 1: 'Can',2:"Cardboard", 3:'Shoes', 4:"Textile"}

    # Get the predicted class index
    predicted_class_index = result_[0]

    # Get the corresponding label from the class_mapping
    predicted_label = class_mapping.get(predicted_class_index, 'Unknown')

    return predicted_label,probability_scores




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # model_path = load_model("best3_vgg16_10__epochs_.h5")
        loaded_model_ = load_model("best3_vgg16_10__epochs_.h5")
        # model_path = os.path.join(script_dir, "model/best3_vgg16_10__epochs_.h5")

        # Make prediction
        predicted_label,probability_scores =waste_classifier(file_path,loaded_model_)
        
        # Set the thresholds for accuracy
        lower_threshold = 0.5
        upper_threshold = 1.0  # Assuming 1.0 as the maximum valid probability
        result=""
        # Check if any accuracy is less than the lower threshold or greater than the upper threshold
        if(max(probability_scores) <lower_threshold or max(probability_scores)>upper_threshold):
            result = "Maybe " + predicted_label +" Or Other Waste"
        else:
            result = "The Image is Predicted as " + predicted_label +" Recyclable"
                
       
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)