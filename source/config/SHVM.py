import torchvision.transforms as T

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614),)
])

mean = (0.4376821, 0.4437697, 0.47280442)

std = (0.19803012, 0.20101562, 0.19703614)

classes = (0,1,2,3,4,5,6,7,8,9)
