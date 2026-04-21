from imgaug import augmenters as iaa

seg = iaa.Sequential(
    [
        iaa.Fliplr(p=0.5),
        iaa.Affine(rotate=(-30, 30)),
    ]
)
