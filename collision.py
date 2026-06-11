class CollisionSensor:

    def __init__(self):

        self.collision = False

    def callback(self, event):

        self.collision = True