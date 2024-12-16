import os
import tensorflow as tf
import cv2
import numpy as np
import time
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.get_logger().setLevel('ERROR')

class CartoonConverter:
    def __init__(self, tflite_model_path):
        if not os.path.exists(tflite_model_path):
            raise FileNotFoundError(f"Model file '{tflite_model_path}' does not exist.")

        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]

    def convert_to_cartoon(self, frame):
        input_frame = cv2.resize(frame, tuple(self.input_shape))
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        input_frame = input_frame.astype(np.float32) / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_frame)
        self.interpreter.invoke()

        cartoon_frame = self.interpreter.get_tensor(self.output_details[0]['index'])
        cartoon_frame = np.squeeze(cartoon_frame, axis=0)
        cartoon_frame = np.clip(cartoon_frame * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cartoon_frame, cv2.COLOR_RGB2BGR)

class VideoFilter:
    def __init__(self, tflite_model_path):
        self.cartoon_converter = CartoonConverter(tflite_model_path)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 15)  
        self.mode = 'cartoon'

    def run(self):
        frame_skip = 2  
        frame_count = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % frame_skip == 0:  
                frame = cv2.flip(frame, 1)

                if self.mode == 'cartoon':
                    try:
                        start_time = time.time()
                        frame = self.cartoon_converter.convert_to_cartoon(frame)
                        print(f"Processing time: {time.time() - start_time:.2f}s")
                    except Exception:
                        traceback.print_exc()

                cv2.imshow('Cartoon Video Filter', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.mode = 'cartoon'
            elif key == ord('o'):
                self.mode = 'original'

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    tflite_model_path = '1.tflite'
    try:
        filter_app = VideoFilter(tflite_model_path)
        filter_app.run()
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    main()
    

    #enjoy
    
