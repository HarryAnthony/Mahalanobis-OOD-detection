import torchvision.transforms as T

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.5), std=(0.5),)
])

classes = ("bedroom",
            "bridge",
            "church_outdoor",
            "classroom",
            "conference_room",
            "dining_room",
            "kitchen",
            "living_room",
            "restaurant",
            "tower")


