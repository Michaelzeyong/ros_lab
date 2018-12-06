#coding=utf-8
import cv2
import face_recognition
import numpy
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
from face_recognition.face_recognition_cli import image_files_in_folder
import time

ALLOWED_EXTENSIONS = {'png','jpg','jpeg','JPG'}

class face_classification(object):

    def train(self , train_dir):

        """
        Trains a k-nearest neighbors classifier for face recognition.

        :param train_dir: directory that contains a sub-directory for each known person, with its name.

         (View in source code to see train_dir example tree structure)

         Structure:
            <train_dir>/
            ├── <person1>/
            │   ├── <somename1>.jpeg
            │   ├── <somename2>.jpeg
            │   ├── ...
            ├── <person2>/
            │   ├── <somename1>.jpeg
            │   └── <somename2>.jpeg
            └── ...

        :param model_save_path: (optional) path to save model on disk
        :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
        :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
        :param verbose: verbosity of training
        :return: returns knn classifier that was trained on the given data.
        """
        
        print("Training KNN classifier...")

        model_save_path="knn_file.clf"
        n_neighbors=2
        knn_algo='ball_tree'
        verbose=False

        X = []
        y = []

        # Loop through each person in the training set
        for class_dir in os.listdir(train_dir):
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            # Loop through each training image for the current person
            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    y.append(class_dir)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)

        print("Training complete!")

    def predict(self,img_path , draw_image = True):
        """
        Recognizes faces in given image using a trained KNN classifier

        :param img_path: path to image to be recognized
        :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
        :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
        :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
               of mis-classifying an unknown person as a known one.
        :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
            For faces of unrecognized persons, the name 'unknown' will be returned.
        """
        model_path="knn_file.clf"
        distance_threshold=0.6

        if not os.path.isfile(img_path) or os.path.splitext(img_path)[1][1:] not in ALLOWED_EXTENSIONS:
            raise Exception("Invalid image path: {}".format(img_path))

        if model_path is None:
            raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
        else:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)

        # Load image file and find face locations
        X_img = face_recognition.load_image_file(img_path)
        X_face_locations = face_recognition.face_locations(X_img)
        print type(X_img)
        print type(X_face_locations)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
        
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        if draw_image:
            print("Looking for faces in %s and draw the image!!!" %img_path)

            pil_image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(pil_image)

            for name, (top, right, bottom, left) in predictions:
                # Draw a box around the face using the Pillow module
                draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

                # There's a bug in Pillow where it blows up with non-UTF-8 text
                # when using the default bitmap font
                name = name.encode("UTF-8")

                # Draw a label with a name below the face
                text_width, text_height = draw.textsize(name)
                draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

            # Remove the drawing library from memory as per the Pillow docs
            del draw

            # Display the resulting image
            pil_image.show()
        else:
            print("Do not draw the image %s" %img_path)

        if predictions[0][0] is "unknown":
            return False
        else:
            return True

    def video_predict(self,gaust,draw_image = True):
        
        model_path="/home/zhangzeyong/robotics/src/project/scripts/face_knn/knn_file.clf"
        distance_threshold=0.3

        if model_path is None:
            raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
        else:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)

        # Load video file and find face locations
        video = cv2.VideoCapture(0)
        
        X_face_locations = []
        face_encodings = []
        process_this_frame = True
        break_while = False

        time_start = time.time()

        while True:
            ret, frame = video.read()
            rgb_small_frame = frame[:,:,::-1]

            X_img = rgb_small_frame
            X_face_locations = face_recognition.face_locations(rgb_small_frame)

            # If no faces are found in the image, return an empty result.
            if len(X_face_locations) == 0:
                X_face_locations = [(frame.shape[0]/4,frame.shape[1]*3/4,frame.shape[0]*3/4,frame.shape[1]/4)]

            # Find encodings for faces in the test iamge
            faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

            # Use the KNN model to find the best matches for the test face
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

            # Predict classes and remove classifications that aren't within the threshold
            predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

            #display the resul
            time_end = time.time()
            spend_time = time_end - time_start

            if draw_image:
                print("Looking for faces...")
                for name, (top, right, bottom, left) in predictions:
                    print("- Found {} at ({}, {})".format(name, left, top))
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), thickness=1)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                cv2.namedWindow("video")
                cv2.imshow("video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                for name, (top, right, bottom, left) in predictions:
                    if name == gaust:
                        time.sleep(3)
                        break_while = True
                        break
                if break_while:
                    break
                if spend_time>60:
                    video.release()
                    cv2.destroyAllWindows()
                    return False

            else:
                print("Do not draw the image!")
                for name, (top, right, bottom, left) in predictions:
                    print("- Found {} at ({}, {})".format(name, left, top))
                    if name == gaust:
                        break_while = True
                        break
                if break_while:
                    break
                if spend_time>60:
                    video.release()
                    cv2.destroyAllWindows()
                    return False

        video.release()
        cv2.destroyAllWindows()

        if break_while:
            return True
        else:
            return False

