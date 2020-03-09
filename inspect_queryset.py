from attack import datasets

queryset_name = "ImageNet1k"
modelfamily = datasets.dataset_to_modelfamily[queryset_name]
transform = datasets.modelfamily_to_transforms[modelfamily]['test']
queryset = datasets.__dict__[queryset_name](train=True, transform=transform)