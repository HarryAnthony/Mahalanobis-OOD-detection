import torchvision.transforms as T
import torchvision

#Training parameters
num_epochs = 400
momentum = 0.9 
weight_decay = 1e-10 
lr_milestones = [int(num_epochs*0.5),int(num_epochs*0.75)]
lr_gamma = 0.2
criterion = 'CrossEntropyLoss'
initialisation_method = 'he'

# network architecture
dropout = 0.3 
depth = 28
widen_factor = 10

# data parameters
image_size = 224

image_size = 32


transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

ID_dataset = torchvision.datasets.CIFAR10(
    root='../../data/cifar10', train=False, download=True, transform=transform_test)

train_ID = torchvision.datasets.CIFAR10(
    root='../../data/cifar10', train=True, download=True, transform=transform_train)

mean = {
    'setting0': (0.4914, 0.4822, 0.4465),
}

std = {
    'setting0': (0.2023, 0.1994, 0.2010),
}

df_name='cifar10'

# Classes for cifar10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')