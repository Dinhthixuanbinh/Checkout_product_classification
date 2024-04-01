from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import EfficientNet_B0_Weights as b0_weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import torchvision.transforms as transforms
import torch


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


WeightsEnum.get_state_dict = get_state_dict


def load_model():
    model = getattr(models, "efficientnet_b0")(weights=b0_weights.DEFAULT)
    nodes = {"avgpool": "features"}
    model = create_feature_extractor(model,
                                     return_nodes=nodes).cuda().eval()
    return model


def process_img():
    return transforms.Compose([transforms.Resize((512, 512)),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])])


class Img2Vec(object):

    def __init__(self):
        self.intermediate_layer_model = load_model()
        self.transform = process_img()

    def get_vec(self, img):
        """ Gets a vector embedding from an image.
        :param image_path: path to image on filesystem
        :returns: numpy ndarray
        """

        # img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        img = img.unsqueeze(0).cuda()
        features = self.intermediate_layer_model(img)["features"]
        intermediate_output = torch.flatten(features, start_dim=1).cuda()

        return intermediate_output.cpu().detach().numpy()
