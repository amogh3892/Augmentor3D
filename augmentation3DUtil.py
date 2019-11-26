import SimpleITK as sitk 
import numpy as np 
import matplotlib.pyplot as plt 
from enum import Enum
import random 

class Transforms(Enum):
    TRANSLATE = 0 
    ROTATE2D = 1
    SHEAR = 2  
    FLIPHORIZONTAL = 3 
    FLIPVERTICAL = 4 

class Augmentation3DUtil(object):
    DIMENSION = 3
    def __init__(self,img,mask=None):
        """
        img : input should be sitk image 
        mask : sitk image of binary segmentation mask (if binary mask is involved)

        A class implemented for the purpose of augmenting 3D volumes. 
        This class uses transformations in SimpleITK library to perform augmentations. 

        Currently rotation, translation and shear implemented 

        To use, define an object of this class and add transformation. 
        Note :  Use the Transforms Enum class to define the transformation. 

        Ex. 
        au = Augmentation3DUtil(img,mask=mask)
        au.add(Transforms.SHEAR,probability = 0.75, magnitude = (0.1,0.1))
        au.add(Transforms.TRANSLATE,probability = 0.75, offset = (2,2,0))
        au.add(Transforms.ROTATE2D,probability = 0.75, degrees = 15)
        ret = au.process(10)


        The above code produces 10 augmented samples by randomly combining 
        all the transformations defined with respect to their probabilities defined.
 
        """
        self.transforms = [] 

        if mask is not None:
            mask.SetDirection(img.GetDirection())
            mask.SetOrigin(img.GetOrigin())
            mask.SetSpacing(img.GetSpacing())
            mask = self._flipimage(mask)
            
        self.mask = mask 
        img = self._flipimage(img)


        self.img = sitk.Cast(img,sitk.sitkFloat64)

        self.reference_image = None 
        self._define_reference_image()

    def _define_reference_image(self):
        img = self.img 
        size = img.GetSize()
        dimension = Augmentation3DUtil.DIMENSION
        origin = img.GetOrigin()
        direction = img.GetDirection()
        spacing = img.GetSpacing()

        reference_image = sitk.Image(size, img.GetPixelIDValue())
        reference_image.SetOrigin(origin)
        reference_image.SetSpacing(spacing)
        reference_image.SetDirection(direction)

        self.reference_image =  reference_image

    def _translate(self,kwargs):

        offset = kwargs["offset"]
        dimension = Augmentation3DUtil.DIMENSION
        transform = sitk.TranslationTransform(dimension)
        transform.SetOffset(offset) 
        return transform
 
    def _affine_transform(self,transformationmatrix):

        dimension = Augmentation3DUtil.DIMENSION
        img = self.img
        reference_image = self.reference_image
        img_center = np.array(img.GetSize())/2.0
        reference_center = np.array(reference_image.GetSize())/2.0

        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(img.GetDirection())
        transform.SetTranslation(np.array(img.GetOrigin()) - np.array(reference_image.GetOrigin()))

        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(img_center))
        reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(reference_center))

        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform.AddTransform(centering_transform)
        
        flipped_transform = sitk.AffineTransform(dimension)    
        flipped_transform.SetCenter(reference_center)

        flipped_transform.SetMatrix(transformationmatrix.ravel())
        centered_transform.AddTransform(flipped_transform)
        
        return centered_transform


    def _flipHorizontal(self):
        tranformationmatrix = np.array([-1,0,0,0,1,0,0,0,1]).astype(float)
        return self._affine_transform(tranformationmatrix)

    def _flipVertical(self):
        tranformationmatrix = np.array([1,0,0,0,-1,0,0,0,1]).astype(float)
        return self._affine_transform(tranformationmatrix)
    
    
    def _rotate2D(self,kwargs):
        degrees = kwargs["degrees"]
        dimension = Augmentation3DUtil.DIMENSION
        transform = sitk.AffineTransform(dimension)
        radians = -np.pi * degrees / 180.
        rotation = np.eye(dimension)
        _rotation = np.array([[np.cos(radians), -np.sin(radians)],[np.sin(radians), np.cos(radians)]])
        rotation[:2,:2] = _rotation

        return self._affine_transform(rotation)

    def _shear(self,kwargs):

        magnitude = kwargs["magnitude"]

        dimension = Augmentation3DUtil.DIMENSION
        transform = sitk.AffineTransform(dimension)
        new_transform = sitk.AffineTransform(transform) 
        matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
        matrix[0,1] = -magnitude[0]
        matrix[1,0] = -magnitude[1]
        new_transform.SetMatrix(matrix.ravel())
        return new_transform


    def _resample(self, img, transform, interpolator):
        # Output image Origin, Spacing, Size, Direction are taken from the reference
        # image in this call to Resample

        img = self.img
        reference_image = self.reference_image
        default_value = 0

        origin = reference_image.GetOrigin()
        new_origin = transform.TransformPoint(origin)

        resampled = sitk.Resample(img, reference_image, transform,
                            interpolator, default_value)
        resampled.SetOrigin(new_origin)
        resampled.SetSpacing(reference_image.GetSpacing())
        resampled.SetDirection(reference_image.GetDirection())

        return resampled

    def _getTransform(self,transform):

        if transform[0] == Transforms.TRANSLATE:
            return self._translate(transform[2])
    
        elif transform[0] == Transforms.ROTATE2D:
            return self._rotate2D(transform[2])

        elif transform[0] == Transforms.SHEAR:
            return self._shear(transform[2])

        elif transform[0] == Transforms.FLIPHORIZONTAL:
            return self._flipHorizontal()

        elif transform[0] == Transforms.FLIPVERTICAL:
            return self._flipVertical()



    def _composite(self,transforms):

        dimension = Augmentation3DUtil.DIMENSION
        composite = sitk.Transform(dimension, sitk.sitkComposite) 
        for transform in transforms:
            composite.AddTransform(self._getTransform(transform)) 
        
        img = self.img 
        mask = self.mask 

        augmented_mask = None 

        if mask is not None:
            augmented_mask = self._resample(mask,composite,interpolator=sitk.sitkNearestNeighbor)

        augmented = self._resample(img,composite,interpolator=sitk.sitkLinear)

        return (augmented,augmented_mask)


    def add(self,transformname,probability,**kwargs):
        self.transforms.append((transformname,probability,kwargs))


    def process(self,samples):
        img = self.img
        augmentations = [] 

        sampling = np.zeros((len(self.transforms),samples)) 

        for i in range(len(self.transforms)):
            per = self.transforms[i][1]
            ind = tuple(random.sample(range(samples),int(per*samples)))
            np.put(sampling[i],ind,1)

        for i in range(samples):

            if not sampling[:,i].sum() == 0:

                ind = tuple(np.where(sampling[:,i]== 1)[0])

                subtransforms = [self.transforms[i] for i in range(len(self.transforms)) if i in ind]
                augmentations.append(self._composite(subtransforms))


        return img,augmentations


    def _flipimage(self,img):
        arr = sitk.GetArrayFromImage(img)
        direction = np.array(img.GetDirection()).reshape(3,3)

        if direction[0].sum() == -1:
            arr = np.flip(arr,2)
        if direction[1].sum() == -1:
            arr = np.flip(arr,1)

        fimg = sitk.GetImageFromArray(arr)
        fimg.SetDirection(np.eye(3).ravel())
        fimg.SetOrigin(img.GetOrigin())
        fimg.SetSpacing(img.GetSpacing())

        return fimg 



