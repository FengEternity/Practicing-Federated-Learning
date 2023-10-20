
import torch 
from torchvision import models

def get_model(name="vgg16", pretrained=True):
	if name == "resnet18":
		model = models.resnet18(pretrained=pretrained)
	elif name == "resnet50":
		model = models.resnet50(pretrained=pretrained)	
	elif name == "densenet121":
		model = models.densenet121(pretrained=pretrained)		
	elif name == "alexnet":
		model = models.alexnet(pretrained=pretrained)
	elif name == "vgg16":
		model = models.vgg16(pretrained=pretrained)
	elif name == "vgg19":
		model = models.vgg19(pretrained=pretrained)
	elif name == "inception_v3":
		model = models.inception_v3(pretrained=pretrained)
	elif name == "googlenet":		
		model = models.googlenet(pretrained=pretrained)
	elif name == "transformer":
		# 为CIFAR数据集设置参数
		patch_size = 4
		embed_dim = 96
		depths = [2, 2, 18, 2]  # 这个深度值是Swin Transformer的一个示例，可以根据需要进行调整
		num_heads = [6, 12, 24, 48]
		window_size = [8, 8]

		# 创建SwinTransformer模型
		model = models.SwinTransformer(
			# img_size = 32,  # CIFAR图像的大小为32x32
			patch_size = (patch_size, patch_size),
			# in_chans= 3,  # 输入通道数，通常为3（RGB图像）
			embed_dim = embed_dim,
			depths = depths,
			num_heads = num_heads,
			window_size = window_size,
			num_classes = 10,  # 对于CIFAR-10，类别数是10
		)
		
	if torch.cuda.is_available():
		return model.cuda()
	else:
		return model 