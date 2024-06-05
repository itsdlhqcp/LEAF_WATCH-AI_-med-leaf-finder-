import base64
import datetime
from io import BytesIO
import json
from flask import Flask, render_template, request
import os
from keras.models import load_model
import numpy as np
import scipy
from PIL import Image
import skimage
import skimage.io
import tensorflow.keras as keras
import tensorflow as tf

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH, 'static/models/')
UPLOAD_FOLDER = os.path.join(BASE_PATH, 'static/image/')

##------------------------------LOAD MODELS -----------------------------
# model_sgd_path = os.path.join(MODEL_PATH,'dsa_image_classification_7_sgd.pickle')
# scaler_path = os.path.join(MODEL_PATH,'dsa_scaler_7.pickle')
# model_sgd = pickle.load(open(model_sgd_path,'rb'))
# scaler = pickle.load(open(scaler_path,'rb'))
model_sgd_path = os.path.join(MODEL_PATH,'resnet50_new.h5')
model = load_model(model_sgd_path)

# uploaded Images folder path
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'image')
# Check if the folder directory exists, if not then create it
if not os.path.exists(app.config['UPLOAD_FOLDER'] ):
    os.makedirs(app.config['UPLOAD_FOLDER'] )


@app.errorhandler(404)
def error404(error):
    message="error404"
    return render_template("error.html",message=message)

@app.errorhandler(405)
def error405(error):
    message="error405"
    return render_template("error.html",message=message)

@app.errorhandler(500)
def error500(error):
    message="error500"
    return render_template("error.html",message=message)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "POST":
        upload_file =request.files['image_name']
        filename=upload_file.filename
        print('The filename has been uploaded =',filename)
        #knows the extensions of the file
        ext = filename.split('.')[-1]
        print('The extention of the filename=',ext)
        if ext.lower() in ['png','jpg','jpeg']:
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print('File saved sucessfully')
            # send to pipeline model
            results = pipeline_model(path_save,model)
            hei = getheight(path_save)
            print(results)
            return render_template('upload.html',fileupload=True,extension=False,data=results,image_filename=filename,height=hei)
        else:
            print('Use only the extention with .jpg, .png, .jpeg')

            return render_template('upload.html',extension=True,fileupload=False)


    else:
        return render_template('upload.html',fileupload=False)

# @app.route('/')
# def about():
#     return render_template("capture.html")

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    filename = ''  # using filename variable to display video feed and captured image alternatively on the same page
    image_data_url = request.form.get('image')

    if request.method == 'POST' and image_data_url:
        try:
            # Decode the base64 data URL to obtain the image data
            image_data = base64.b64decode(image_data_url.split(',')[1])
            
            # Create an image from the decoded data
            img = Image.open(BytesIO(image_data))
            
            # Convert the image to RGB mode if it's in RGBA mode
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Generate a filename with the current date and time
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            upload_file_name = f"leaf_{timestamp}.jpg"
            
            # Save the image to the upload folder
            upload_file_path = os.path.join(UPLOAD_PATH, upload_file_name)
            img.save(upload_file_path, 'JPEG')
            print('File saved successfully')

            # Send to the pipeline model
            results = pipeline_model(upload_file_path,model)
            hei = getheight(upload_file_path)
            print(results)

            # Display the results on the template
            return render_template('capture.html', fileupload=True, extension=False, data=results,image_filename=upload_file_name,height=hei)
            
        except IndexError as e:
            error_message = f'Error processing image: {str(e)}'
            return render_template('capture.html', filename=filename, error_message=error_message)

    return render_template('capture.html', filename=filename)
    

def getheight(path):
    img = skimage.io.imread(path)
    h,w,_ =img.shape
    aspect = h/w
    given_width = 100
    height = given_width*aspect
    return height


      
    
# MODELING A PIPELINE
def pipeline_model(path,model):
    # pipeline model
    # have already loaded the image
    # image = skimage.io.imread(path)
    # RESIZE AND PROCESS IMAGE
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    # CONVERTING IMAGE INTO ARRAY
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.array([img_array])
    # generate predictions for samples
    predictions = model.predict(img_array)
    # define classes name
    # class_names = ['Aadaloodakam','Aaryavepp','Arutha','Asoca','Ayapana','Ayurvedic(Morooti_like)','Ayyappala','Changalamparanda','Chethikkoduveli','Illipa','KrishnaTulasi','Kumizhu','Moovila','NaagaDanthi','Nannari','nonAyurvedic (Mango_like)','nonAyurvedic (Mexican Mint  like)','nonAyurvedic (Tulsi_like)','nonAyurvedic(Arive-Dantu_like)','nonAyurvedic(Basale_like)','nonAyurvedic(Betel_like)','nonAyurvedic(Crape Jasmine_like)','nonAyurvedic(Curry_like)','nonAyurvedic(Drumstick_like)','nonAyurvedic(Fenugreek_like)','nonAyurvedic(Guava_like)','nonAyurvedic(Indian Beech like)','nonAyurvedic(Indian Mustard_like)','nonAyurvedic(Jackfruit_like)','nonAyurvedic(Jamaica Cherry-Gasagase_like)','nonAyurvedic(Jamun_like)','nonAyurvedic(Jasmine_like)','nonAyurvedic(Karanda_like)','nonAyurvedic(Lemon_like)','nonAyurvedic(Mint_like)','nonAyurvedic(Oleander_like)','nonAyurvedic(Parijata_like)','nonAyurvedic(Peepal Tree_like)','nonAyurvedic(Pomegranate_like)','nonAyurvedic(Rasna_like)','nonAyurvedic(Rose Apple_like)','nonAyurvedic(Roxburgh fig_like)','nonAyurvedic(Sandalwood_like)','nonAyurvedic_x','nonAyurvedic1','nonAyurvedic2','nonAyurvedic3','nonAyurvedic4','nonAyurvedic5','nonAyurvedic6','nonAyurvedic7','nonAyurvedic8','nonAyurvedic9','nonAyurvedic10','nonAyurvedic11','nonAyurvedic12','nonAyurvedic13','nonAyurvedic14','nonAyurvedic15','nonAyurvedic16','nonAyurvedic17','nonAyurvedic18','nonAyurvedic19','nonAyurvedic20','nonAyurvedic21','nonAyurvedic22','nonAyurvedic23','nonAyurvedic24','nonAyurvedic25','nonAyurvedic26','nonAyurvedic27','nonAyurvedic28','nonAyurvedic29','nonAyurvedic30','nonAyurvedic31','nonAyurvedic32','nonAyurvedic-Betel_like','nonAyurvedic-Hibiscus_like','nonAyurvedic-Honge_like','nonAyurvedic-Lemongrass-like','nonAyurvedic-Parijatha-like','nonAyurvedic-Pea-like','nonAyurvedic-Pepper-like','nonAyurvedic-Raddish-like','nonAyurvedic-Turmeric-like','Orila','Panichakam','paniKoorka','Thipalli']
    # generate argmax for predictions
    # class_names = ['Aadaloodakam','Arutha','Ayapana','nonAyurvedic (Mango_like)','nonAyurvedic (Tulsi_like)','nonAyurvedic(Arive-Dantu_like)','nonAyurvedic(Basale_like)','nonAyurvedic(Betel_like)','nonAyurvedic(Crape Jasmine_like)','nonAyurvedic(Curry_like)','nonAyurvedic(Drumstick_like)','Vaatamkolli']
    # class_names = ['Aadaloodakam','Aaryavepp','Arutha','Asoca','Ayapana','Ayurvedic(Morooti_like)','Ayyappala','Brahmi','Changalamparanda','Chethikkoduveli','Chitamruth','Illipa','KrishnaTulasi','Kumizhu','Moovila','Morikooti','NaagaDanthi','Nannari','nonAyurvedic (Mango_like)','nonAyurvedic (Mexican Mint  like)','nonAyurvedic (Tulsi_like)','nonAyurvedic(Arive-Dantu_like)','nonAyurvedic(Basale_like)','nonAyurvedic(Betel_like)','nonAyurvedic(Crape Jasmine_like)','nonAyurvedic(Curry_like)','nonAyurvedic(Drumstick_like)','nonAyurvedic(Fenugreek_like)','nonAyurvedic(Guava_like)','nonAyurvedic(Indian Beech like)','nonAyurvedic(Indian Mustard_like)','nonAyurvedic(Jackfruit_like)','nonAyurvedic(Jamaica Cherry-Gasagase_like)','nonAyurvedic(Jamun_like)','nonAyurvedic(Jasmine_like)','nonAyurvedic(Karanda_like)','nonAyurvedic(Lemon_like)','nonAyurvedic(Mint_like)','nonAyurvedic(Oleander_like)','nonAyurvedic(Parijata_like)','nonAyurvedic(Peepal Tree_like)','nonAyurvedic(Pomegranate_like)','nonAyurvedic(Rasna_like)','nonAyurvedic(Rose Apple_like)','nonAyurvedic(Roxburgh fig_like)','nonAyurvedic(Sandalwood_like)','nonAyurvedic_x','nonAyurvedic1','nonAyurvedic2','nonAyurvedic3','nonAyurvedic4','nonAyurvedic5','nonAyurvedic6','nonAyurvedic7','nonAyurvedic8','nonAyurvedic9','nonAyurvedic10','nonAyurvedic11','nonAyurvedic12','nonAyurvedic13','nonAyurvedic14','nonAyurvedic15','nonAyurvedic16','nonAyurvedic17','nonAyurvedic18','nonAyurvedic19','nonAyurvedic20','nonAyurvedic21','nonAyurvedic22','nonAyurvedic23','nonAyurvedic24','nonAyurvedic25','nonAyurvedic26','nonAyurvedic27','nonAyurvedic28','nonAyurvedic29','nonAyurvedic30','nonAyurvedic31','nonAyurvedic32','nonAyurvedic33', 'nonAyurvedic34', 'nonAyurvedic35', 'nonAyurvedic36', 'nonAyurvedic37', 'nonAyurvedic38', 'nonAyurvedic39', 'nonAyurvedic40', 'nonAyurvedic41', 'nonAyurvedic42', 'nonAyurvedic43', 'nonAyurvedic44', 'nonAyurvedic45', 'nonAyurvedic46', 'nonAyurvedic47', 'nonAyurvedic48', 'nonAyurvedic49', 'nonAyurvedic50', 'nonAyurvedic51', 'nonAyurvedic52', 'nonAyurvedic53', 'nonAyurvedic54', 'nonAyurvedic55', 'nonAyurvedic56', 'nonAyurvedic57', 'nonAyurvedic58', 'nonAyurvedic59', 'nonAyurvedic60', 'nonAyurvedic61', 'nonAyurvedic62','nonAyurvedic-Betel_like','nonAyurvedic-Hibiscus_like','nonAyurvedic-Honge_like','nonAyurvedic-Lemongrass-like','nonAyurvedic-Parijatha-like','nonAyurvedic-Pea-like','nonAyurvedic-Pepper-like','nonAyurvedic-Raddish-like','nonAyurvedic-Turmeric-like','Orila','Panichakam','paniKoorka','Thipalli','Vaatamkolli']
    class_names = ['Aadaloodakam','Aaryavepp','Arutha','Asoca','Ayapana','Ayurvedic(Morooti_like)','Ayyappala','Brahmi','Changalamparanda','Chethikkoduveli','Chitamruth','Illipa','KrishnaTulasi','Kumizhu','Moovila','Morikooti','NaagaDanthi','Nannari','nonAyurvedic (Mango_like)','nonAyurvedic (Mexican Mint  like)','nonAyurvedic (Tulsi_like)','nonAyurvedic(Arive-Dantu_like)','nonAyurvedic(Basale_like)','nonAyurvedic(Betel_like)','nonAyurvedic(Crape Jasmine_like)','nonAyurvedic(Curry_like)','nonAyurvedic(Drumstick_like)','nonAyurvedic(Fenugreek_like)','nonAyurvedic(Guava_like)','nonAyurvedic(Indian Beech like)','nonAyurvedic(Indian Mustard_like)','nonAyurvedic(Jackfruit_like)','nonAyurvedic(Jamaica Cherry-Gasagase_like)','nonAyurvedic(Jamun_like)','nonAyurvedic(Jasmine_like)','nonAyurvedic(Karanda_like)','nonAyurvedic(Lemon_like)','nonAyurvedic(Mint_like)','nonAyurvedic(Oleander_like)','nonAyurvedic(Parijata_like)','nonAyurvedic(Peepal Tree_like)','nonAyurvedic(Pomegranate_like)','nonAyurvedic(Rasna_like)','nonAyurvedic(Rose Apple_like)','nonAyurvedic(Roxburgh fig_like)','nonAyurvedic(Sandalwood_like)','nonAyurvedic_x','nonAyurvedic1','nonAyurvedic2','nonAyurvedic3','nonAyurvedic4','nonAyurvedic5','nonAyurvedic6','nonAyurvedic7','nonAyurvedic8','nonAyurvedic9','nonAyurvedic10','nonAyurvedic11','nonAyurvedic12','nonAyurvedic13','nonAyurvedic14','nonAyurvedic15','nonAyurvedic16','nonAyurvedic17','nonAyurvedic18','nonAyurvedic19','nonAyurvedic20','nonAyurvedic21','nonAyurvedic22','nonAyurvedic23','nonAyurvedic24','nonAyurvedic25','nonAyurvedic26','nonAyurvedic27','nonAyurvedic28','nonAyurvedic29','nonAyurvedic30','nonAyurvedic31','nonAyurvedic32','nonAyurvedic33', 'nonAyurvedic34', 'nonAyurvedic35', 'nonAyurvedic36', 'nonAyurvedic37', 'nonAyurvedic38', 'nonAyurvedic39', 'nonAyurvedic40', 'nonAyurvedic41', 'nonAyurvedic42', 'nonAyurvedic43', 'nonAyurvedic44', 'nonAyurvedic45', 'nonAyurvedic46', 'nonAyurvedic47', 'nonAyurvedic48', 'nonAyurvedic49', 'nonAyurvedic50', 'nonAyurvedic51', 'nonAyurvedic52', 'nonAyurvedic53', 'nonAyurvedic54', 'nonAyurvedic55', 'nonAyurvedic56', 'nonAyurvedic57', 'nonAyurvedic58', 'nonAyurvedic59', 'nonAyurvedic60', 'nonAyurvedic61', 'nonAyurvedic62','nonAyurvedic-Betel_like','nonAyurvedic-Hibiscus_like','nonAyurvedic-Honge_like','nonAyurvedic-Lemongrass-like','nonAyurvedic-Parijatha-like','nonAyurvedic-Pea-like','nonAyurvedic-Pepper-like','nonAyurvedic-Raddish-like','nonAyurvedic-Turmeric-like','Orila','Panichakam','paniKoorka','Thipalli','Vaatamkolli']
    # class_names = ['Aadaloodakam','Aaryavepp','Arutha','Asoca','Ayapana','Ayurvedic(Morooti_like)','Ayyappala','Changalamparanda','Chethikkoduveli','Illipa','KrishnaTulasi','Kumizhu','Moovila','NaagaDanthi','Nannari','nonAyurvedic (Mango_like)','nonAyurvedic (Mexican Mint  like)','nonAyurvedic (Tulsi_like)','nonAyurvedic(Arive-Dantu_like)','nonAyurvedic(Basale_like)','nonAyurvedic(Betel_like)','nonAyurvedic(Crape Jasmine_like)','nonAyurvedic(Curry_like)','nonAyurvedic(Drumstick_like)','nonAyurvedic(Fenugreek_like)','nonAyurvedic(Guava_like)','nonAyurvedic(Indian Beech like)','nonAyurvedic(Indian Mustard_like)','nonAyurvedic(Jackfruit_like)','nonAyurvedic(Jamaica Cherry-Gasagase_like)','nonAyurvedic(Jamun_like)','nonAyurvedic(Jasmine_like)','nonAyurvedic(Karanda_like)','nonAyurvedic(Lemon_like)','nonAyurvedic(Mint_like)','nonAyurvedic(Oleander_like)','nonAyurvedic(Parijata_like)','nonAyurvedic(Peepal Tree_like)','nonAyurvedic(Pomegranate_like)','nonAyurvedic(Rasna_like)','nonAyurvedic(Rose Apple_like)','nonAyurvedic(Roxburgh fig_like)','nonAyurvedic(Sandalwood_like)','nonAyurvedic_x','nonAyurvedic1','nonAyurvedic2','nonAyurvedic3','nonAyurvedic4','nonAyurvedic5','nonAyurvedic6','nonAyurvedic7','nonAyurvedic8','nonAyurvedic9','nonAyurvedic10','nonAyurvedic11','nonAyurvedic12','nonAyurvedic13','nonAyurvedic14','nonAyurvedic15','nonAyurvedic16','nonAyurvedic17','nonAyurvedic18','nonAyurvedic19','nonAyurvedic20','nonAyurvedic21','nonAyurvedic22','nonAyurvedic23','nonAyurvedic24','nonAyurvedic25','nonAyurvedic26','nonAyurvedic27','nonAyurvedic28','nonAyurvedic29','nonAyurvedic30','nonAyurvedic31','nonAyurvedic32','nonAyurvedic33', 'nonAyurvedic34', 'nonAyurvedic35', 'nonAyurvedic36', 'nonAyurvedic37', 'nonAyurvedic38', 'nonAyurvedic39', 'nonAyurvedic40', 'nonAyurvedic41', 'nonAyurvedic42', 'nonAyurvedic43', 'nonAyurvedic44', 'nonAyurvedic45', 'nonAyurvedic46', 'nonAyurvedic47', 'nonAyurvedic48', 'nonAyurvedic49', 'nonAyurvedic50', 'nonAyurvedic51', 'nonAyurvedic52', 'nonAyurvedic53', 'nonAyurvedic54', 'nonAyurvedic55', 'nonAyurvedic56', 'nonAyurvedic57', 'nonAyurvedic58', 'nonAyurvedic59', 'nonAyurvedic60', 'nonAyurvedic61', 'nonAyurvedic62','nonAyurvedic-Betel_like','nonAyurvedic-Hibiscus_like','nonAyurvedic-Honge_like','nonAyurvedic-Lemongrass-like','nonAyurvedic-Parijatha-like','nonAyurvedic-Pea-like','nonAyurvedic-Pepper-like','nonAyurvedic-Raddish-like','nonAyurvedic-Turmeric-like','Orila','Panichakam','paniKoorka','Thipalli']
    # generate argmax for predictions
    class_id = np.argmax(predictions, axis = 1)
    predictions = predictions.flatten()
    # cal. z score
    z = scipy.stats.zscore(predictions)
    prob_value = scipy.special.softmax(z)
    # getting top three probabilty values
    top_2_prob_ind = prob_value.argsort()[::-1][:3]
    # for making pipeline clasnames should be defined in array
    top_labels = [class_names[class_id.item()] for class_id in top_2_prob_ind]
    top_prob = prob_value[top_2_prob_ind]
    top_dict = dict()
    for key,val in zip(top_labels,top_prob):
    #     top_dict.update({key:np.round(val,3)})
        top_dict.update({key: {"probability": np.round(val, 3), "details": get_plant_details(key)}})
    return top_dict
def get_plant_details(plant_name):
    # Corrected file path
    json_file_path = os.path.join('static', 'js', 'plant.json')
    # Load plant details from JSON file
    with open(json_file_path, 'r') as file:
        plant_details = json.load(file)

    return plant_details.get(plant_name, {"description": "Details not available."})       

if __name__ == "__main__":
    app.run(debug=True)