import cv2
import os
import glob
import pickle
from keras_vggface.vggface import VGGFace
import numpy as np
from keras.preprocessing import image
# import utils
from keras.applications.resnet50 import preprocess_input

def pickle_stuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()

class FaceExtractor(object):
    """
    Singleton class to extraction face images from video files
    """
    CASE_PATH = "pretrained_models/haarcascade_frontalface_alt.xml"

    def __new__(cls, weight_file=None, face_size=224):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceExtractor, cls).__new__(cls)
        return cls.instance

    def __init__(self, face_size=224):
        self.face_size = face_size

    def crop_face(self, imgarray, section, margin=20, size=224):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)
    
    def detect_face(self,img,save_folder):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame=image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=10,
            minSize=(64, 64)
        )
        # placeholder for cropped faces
        face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
        for i, face in enumerate(faces):
            face_img, cropped = self.crop_face(frame, face, margin=10, size=self.face_size)
            (x, y, w, h) = cropped
            imgfile = "1.png"
            imgfile = os.path.join(save_folder, imgfile)
            cv2.imwrite(imgfile, face_img)

def image2x(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x

def cal_mean_feature(image_folder):
    face_images = list(glob.iglob(os.path.join(image_folder, '*.*')))

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    batch_size = 32
    face_images_chunks = chunks(face_images, batch_size)
    fvecs = None
    for face_images_chunk in face_images_chunks:
        images = np.concatenate([image2x(face_image) for face_image in face_images_chunk])
        batch_fvecs = resnet50_features.predict(images)
        if fvecs is None:
            fvecs = batch_fvecs
        else:
            fvecs = np.append(fvecs, batch_fvecs, axis=0)
    return np.array(fvecs).sum(axis=0) / len(fvecs)

if __name__ == "__main__":
    resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                                pooling='avg')
    
    FACE_IMAGES_FOLDER = "./data/face_images"
    extractor = FaceExtractor()
    # name='DSC01368.jpg'
    # sub_dir='name1'
    sub_dir_path=os.path.join(FACE_IMAGES_FOLDER,'original')
    sub_dirs=os.listdir(sub_dir_path)

    precompute_features = []
    for sub_dir in sub_dirs:
        sub_dir_image_files=os.path.join(sub_dir_path,sub_dir)
        if(not os.path.isdir(sub_dir_image_files)):
            continue
        print(sub_dir_image_files)
        images=os.listdir(sub_dir_image_files)
        images=[item for item in images if item.split('.')[-1]=='jpg']
        print(images)
        name=images[0]

        save_folder = os.path.join(FACE_IMAGES_FOLDER,'faces',sub_dir)
        image_path=os.path.join(FACE_IMAGES_FOLDER,'original',sub_dir,name)
        os.makedirs(save_folder, exist_ok=True)

        extractor.detect_face(image_path, save_folder)
        
        mean_features = cal_mean_feature(image_folder=save_folder)
        precompute_features.append({"name": sub_dir, "features": mean_features})

    pickle_stuff("./data/precompute_features.pickle", precompute_features)
        
