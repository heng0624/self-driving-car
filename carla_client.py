import carla

def connect():
    client = carla.Client(
        "localhost",
        2000
    )

    client.set_timeout(10.0)

    return client