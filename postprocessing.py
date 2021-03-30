from utils import * 

def aug(file):
    if file=='train':
        return al.Compose([
            al.VerticalFlip(p=0.5),
            al.HorizontalFlip(p=0.5),
            al.RandomRotate90(p=0.5),
            al.OneOf([
                al.GaussNoise(0.002, p=0.5),
                al.IAAAffine(p=0.5),
            ], p=0.2),
            al.OneOf([
                al.Blur(blur_limit=(3, 10), p=0.4),
                al.MedianBlur(blur_limit=3, p=0.3),
                al.MotionBlur(p=0.3)
            ], p=0.3),
            al.OneOf([
                al.RandomBrightness(p=0.3),
                al.RandomContrast(p=0.4),
                al.RandomGamma(p=0.3)
            ], p=0.5),
            al.Cutout(num_holes=20, max_h_size=20, max_w_size=20, p=0.5),
#             al.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.3),
            al.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(p=1)
        ])

    elif file=='validation':
        return al.Compose([
            al.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(p=1)
        ])

    elif file=='test':
        return al.Compose([
            al.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(p=1)
        ], p=1)

class EfficientnetCus(nn.Module):
    
    def __init__(self, model, num_class, model_weight=None, is_train=True):
        super(EfficientnetCus, self).__init__()
        
        self.is_train = is_train
        self.model = timm.create_model(f'tf_efficientnet_{model}_ns', pretrained=is_train,
                                       in_chans=3, num_classes=num_class)
        if model_weight is not None:
            new_keys = self.model.state_dict().keys()
            values = torch.load(model_weight, map_location=lambda storage, loc: storage).values()
            self.model.load_state_dict(OrderedDict(zip(new_keys, values)))
                
    def forward(self, image):
        if self.is_train:
            out = self.model(image)
            return out.squeeze(-1)
        else:
            vertical = image.flip(1)
            horizontal = image.flip(2)
            rotate90 = torch.rot90(image, 1, (1, 2))
            rotate90_ = torch.rot90(image, 1, (2, 1))
            out = torch.stack([image, vertical, horizontal, rotate90, rotate90_])
            return torch.sigmoid(self.model(out)).mean()

