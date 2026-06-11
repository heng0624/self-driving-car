import numpy as np

class CameraSensor:

    def __init__(self):

        self.image = None

    def callback(self, image):

        array = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        )

        array = array.reshape(
            image.height,
            image.width,
            4
        )

        self.image = array[:, :, :3]