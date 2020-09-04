import torch
import torchvision.utils as vutils
from torch.autograd import Variable
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torchvision import transforms
from PIL import Image

from data import ImagePipeline

device = 'cuda'

datapath = '/content/drive/My Drive/Testset/Test'
transform = transforms.Compose([transforms.ToTensor()])
# Generate frontal images from the test set
def frontalize(model, features, datapath, mtest):
    
    test_pipe = ImagePipeline(datapath, image_size=128, random_shuffle=False, batch_size=mtest)
    test_pipe.build()
    test_pipe_loader = DALIGenericIterator(test_pipe, ["profiles", "frontals"], mtest)
    with torch.no_grad():
        for data in test_pipe_loader:
            profile = data[0]['profiles']
            frontal = data[0]['frontals']
            generated = model(Variable(profile.type('torch.FloatTensor').to(device)), features)
    vutils.save_image(torch.cat((profile, generated.data, frontal)), 'output/test.jpg', nrow=mtest, padding=2, normalize=True)
    return

# Load a pre-trained Pytorch model
saved_model = torch.load("./output/netG_199.pt")
saved_autoencoder = torch.load("./output/autoencoder_199.pt")

tensor = transforms.ToTensor()
img_tensor = tensor(Image.open('/content/drive/My Drive/Testset/Test/100.jpg'))
img_tensor = img_tensor.unsqueeze(0)
input_f = Variable(img_tensor).type('torch.FloatTensor').to(device)
with torch.no_grad():
    output, features = saved_autoencoder(input_f)

frontalize(saved_model, features, datapath, 5)

