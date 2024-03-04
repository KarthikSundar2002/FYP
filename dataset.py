import os
from torchvision import datasets, transforms
from RFMID import RFMiD
from timm.data import create_transform

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if is_train:
        dataset = RFMiD(image_dir='dataset/RFMID/Training',label_dir='dataset/RFMID/train.csv',transform=transform)
    else:
        dataset = RFMiD(image_dir='dataset/RFMID/resize/testset',label_dir='dataset/RFMID/test.csv',transform=transform)
    nb_classes = args.nb_classes

    print('Number of Classes = %d' % nb_classes)
    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    input_size = args.input_size
    
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            # mean=mean,
            # std=std
        )
        print(transforms)
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform
    
    t = []
    if resize_im:

        if args.input_size >= 384:
            t.append(
                transforms.Resize((args.input_size,args.input_size), interpolation=transforms.InterpolationMode.BICUBIC)
            )
        else:
            if args.crop_pct is None:
                args.crop_pct = 224/256
                size = int(args.input_size / args.crop_pct)
                t.append(
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                )
                t.append(transforms.CenterCrop(args.input_size))
    
    t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)