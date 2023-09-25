import torch
from typing import Tuple, List
import math
import random
import skimage
import numpy as np
import math, PIL.Image, PIL.ImageDraw, PIL.ImageFont, PIL.ImageColor
from matplotlib import font_manager
import torchvision.transforms as T
import scipy


def modify_transforms(new_transform, transform_list, where_to_insert='end', insert_transform=None):
    """
    Modifies the transforms applied to the dataset.

    Parameters
    ----------
    new_transform : torchvision.transforms
        The transform to be added to the transform list.
    transform_list : torchvision.transforms
        The transform list to be modified.
    where_to_insert : str
        Where to insert the new transform. Options are 'end', 'insert_after', 'insert_before', or 'replace'. Default is 'end'.
    insert_transform : torchvision.transforms
        The transform to insert after, before, or replace. Required if 'where_to_insert' is 'insert_after', 'insert_before', or 'replace'. Default is None.

    Returns
    -------
    torchvision.transforms
        The new transform list.
    """
    if where_to_insert == 'end':
        # Add the new transform to the end of the list
        new_transforms = transform_list.transforms + [new_transform]
    elif where_to_insert == 'insert_after' or where_to_insert == 'insert_before':
        if insert_transform is None:
            raise ValueError("When inserting before or after, 'insert_transform' must be specified.")
        
        insert_index = None
        for idx, transform in enumerate(transform_list.transforms):
            if isinstance(transform, type(insert_transform)):
                insert_index = idx if where_to_insert == 'insert_before' else idx + 1
                break
        
        if insert_index is None:
            raise ValueError(f"Transform {insert_transform} not found in the transform list.")
        
        new_transforms = (
            transform_list.transforms[:insert_index] + [new_transform] +
            transform_list.transforms[insert_index:]
        )
    elif where_to_insert == 'replace':
        if insert_transform is None:
            raise ValueError("When replacing, 'insert_transform' must be specified.")
        
        replace_index = None
        for idx, transform in enumerate(transform_list.transforms):
            if isinstance(transform, type(insert_transform)):
                replace_index = idx
                break
        
        if replace_index is None:
            raise ValueError(f"Transform {insert_transform} not found in the transform list.")
        
        new_transforms = (
            transform_list.transforms[:replace_index] + [new_transform] +
            transform_list.transforms[replace_index + 1:]
        )
    else:
        raise ValueError("Invalid 'insert' argument. Use 'None', 'insert_after', 'insert_before', or 'replace'.")
    
    new_transform_test = T.Compose(new_transforms)
    return new_transform_test


class RandomErasing_square(torch.nn.Module):
    """
    Class that randomly erases a square patch from an image. Inspired by the paper 'Random Erasing Data Augmentation' by Zhong et al.

    Parameters
    ----------
    p : float
        probability that the operation will be performed.
    scale : Tuple[float,float]
        range of proportion of erased area against input image.
    ratio : Tuple[float,float]
        range of aspect ratio of erased area.
    value : str
        erasing value. Default is '0'.
        If float, it is used to erase the area.
        If str, it must be one of the following:
            'random_gaussian_noise': random Gaussian noise with mean (self.noise_mean) and std (self.noise_std).
            'random_uniform_noise': random uniform noise between minimum and maximum value of image.
            'foreign_texture': uses foreign texture (self.foreign_texture).
            'image_replace': uses another part of the image.
            'image_replace_no_overlap': uses another part of the image, but without overlapping with the erased area.
    setting : str
        Position of erased area. Default is 'random'. Options are:
            'random': random position of square patch
            'centred': square patch in the centre
            'near_centre': square patch near the center (Gaussian distribution)
            'periphery': square patch in the periphery (image outline)
            'corners': square patch in the corners
            'near_corners': square patch near the corners (Gaussian distribution)
            'near_periphery': square patch near the periphery (Gaussian distribution)
    noise_mean : str
        Mean of Gaussian noise. Default is 'img_mean', or should be a float.
    noise_std : str
        Standard deviation of Gaussian noise. Default is 'img_std', or should be a float.
    noise_coarseness : float
        Coarseness of noise or foreign texture (>=1). Default is 1.
    rotation_angle : float
        Angle of rotation of the erased area. Default is 0. 'random' will rotate the erased area by a random angle.
    foreign_texture : torch.tensor
        Foreign texture (2D or 3D) to be used to fill the erased area. Default is a 10x10 checkerboard pattern.
    gaussian_filter_sigma : float
        Standard deviation of Gaussian filter to be applied to the erased area to smooth the edges of the erased area. Default is 0.
    make_transparent : bool
        Whether to make the erased area transparent. Default is False.
    transparency_power : float
        Power to raise the transparency power to. Default is 5.
    """

    def __init__(self, p:float=1, scale:Tuple[float,float]=(0.1,0.1), ratio:Tuple[float,float]=(1,1), value:str='0', setting:str='random',noise_mean:str='img_mean',
                 noise_std:str='img_std',noise_coarseness:float=1,rotation_angle:float=0,foreign_texture = torch.tensor(np.kron([[1,0]*5,[0,1]*5]*5, np.ones((10, 10))),dtype=torch.float32),
                 gaussian_filter_sigma:float=0,make_transparent:bool=False,transparency_power:float=5,**kwargs):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.setting = setting
        self.mean = noise_mean
        self.coarseness = max(noise_coarseness,1)
        self.std = noise_std
        self.rotation_angle = rotation_angle
        self.foreign_texture = foreign_texture
        self.gaussian_filter_sigma = gaussian_filter_sigma
        self.transparency_power = transparency_power
        self.make_transparent = make_transparent

        # Set additional attributes based on **kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


    def forward(self, img: torch.Tensor,**kwargs):
        """
        Forward pass of the transformation.

        Parameters
        ----------
        img : torch.Tensor
            Image to be transformed.

        Returns
        -------
        torch.Tensor
            Transformed image.
        """
        if random.uniform(0, 1) < self.p:
            img = self.erase(img,**kwargs)
        return img
    

    def erase(self,img,**kwargs):
        """
        Erases the shape from an image.

        Parameters
        ----------
        img : torch.Tensor
            Image to be transformed.

        Returns
        -------
        torch.Tensor
            Transformed image.
        """
        #Get area of image to erase
        _, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        h,w = self._get_artefact_area(img_h,img_w,**kwargs)
        i, j = self._get_erased_coordinates(img_h, img_w, h, w,**kwargs)
        mask = self._get_bool_mask(img,h,w,i,j,**kwargs)
        mask = self._rotate_mask(mask,**kwargs)

        # Used for making the mask transparent or have softening transition between erased and non-erased areas
        if self.make_transparent == True or self.gaussian_filter_sigma > 0:
            original_image = img.clone()
            if self.gaussian_filter_sigma > 0:
                mask_blurred = scipy.ndimage.gaussian_filter(mask.float(), sigma=self.gaussian_filter_sigma)

        # Erase the area
        img = self._erase_mask(img,mask,h,w,**kwargs)

        #Make the mask transparent or have softening transition between erased and non-erased areas
        if self.gaussian_filter_sigma > 0: #For softening the edges of the mask
            transparent = (mask_blurred < 1.0) & (mask_blurred >= 0.0)
            transparent = torch.from_numpy(transparent).bool()
            transparent = transparent.expand(img.size(0),-1,-1)
            img[transparent] = original_image[transparent] * (1 - (mask_blurred[transparent])**self.transparency_power)+ img[transparent
                                ] * (mask_blurred[transparent])**self.transparency_power
        elif self.make_transparent == True: #For making whole mask transparent
            img[mask] = original_image[mask] * (0.5)**(self.transparency_power) + img[mask] * 1-((0.5)**self.transparency_power)

        return img
    

    def _erase_mask(self,img,mask,h,w,**kwargs):
        #For filling the mask with a constant value
        if self._check_if_string_is_number(str(self.value)): 
            img[mask] = float(self.value)

        #For filling the mask with random noise or a foreign texture
        elif self.value in ['random_Gaussian_noise','random_gaussian_noise','random_uniform_noise','foreign_texture']:
            if self.value in ['random_Gaussian_noise','random_gaussian_noise']:
                noise_mean = float(self.mean) if self._check_if_string_is_number(self.mean) == True else  torch.mean(img.flatten()).item()
                noise_std = float(self.std) if self._check_if_string_is_number(self.std) == True else  torch.std(img.flatten()).item()
                noise_box = torch.normal(noise_mean, noise_std, size=(int(img.shape[-2] / self.coarseness), int(img.shape[-1] / self.coarseness)))
            elif self.value == 'random_uniform_noise':
                noise_min = torch.min(img.flatten()).item()
                noise_max = torch.max(img.flatten()).item()
                noise_box = (noise_max - noise_min) * torch.rand(size=(int(img.shape[-2] / self.coarseness), int(img.shape[-1] / self.coarseness))) + noise_min
            else:
                if isinstance(self.foreign_texture, (np.ndarray, torch.Tensor)):
                    if len(self.foreign_texture.shape) == 2 and self.foreign_texture.shape[0] == self.foreign_texture.shape[1]:
                        self.coarseness = img.shape[-1] / self.foreign_texture.shape[-1]
                    else:
                        raise ValueError("Custom texture size must be 2d square")
                else:
                    raise ValueError("Custom texture must be a numpy array or a torch tensor")
                noise_box = self.foreign_texture
            if self.coarseness > 1:
                noise_box = torch.nn.functional.interpolate(noise_box.unsqueeze(0).unsqueeze(1), size=(img.shape[-2], img.shape[-1]), mode='bilinear').squeeze(0)
            img[mask] = noise_box.expand(img.size(0),-1,-1)[mask]

        #For replacing the mask with a different part of the image    
        elif self.value in ['image_replace','image_replace_no_overlap']:
                img_copy = img.clone()
                new_mask = torch.zeros_like(mask)  # Create a new mask with the same shape as the original mask
                attempts = 0
                while attempts < 20:
                    new_i = torch.randint(0, new_mask.shape[0] - mask.shape[0] + 1, ())
                    new_j = torch.randint(0, new_mask.shape[1] - mask.shape[1] + 1, ())
                    new_mask = self._get_bool_mask(img,h,w,new_i,new_j,**kwargs)  # Set the new mask at random coordinates
                    if torch.sum(new_mask & mask) == 0 or self.value == 'image_replace':
                        break  # Exit the loop if the masks don't overlap
                    else:
                        new_mask.zero_()  # Reset the new mask if it overlaps
                    attempts += 1
                if attempts == 20:
                    raise RuntimeError('Could not find a valid location for RandomErasing which does not overlap with the original mask')
                img[mask] = img_copy[new_mask]  # Replace masked area with data from the new mask's location
        else:
            raise ValueError('Invalid value for RandomErasing: {}'.format(self.value))
        
        return img
    

    def _rotate_mask(self,mask):
        if self.rotation_angle == 'random' or self._check_if_string_is_number(str(self.rotation_angle)) == False:
            self.rotation_angle = random.randint(0, 360)
        return T.functional.rotate(mask, float(self.rotation_angle))
    

    def _check_if_string_is_number(self,s):
        try:
            float(s) or int(s)
            return True
        except ValueError:
            return False
    

    def _get_artefact_area(self,img_h,img_w):
        area = img_h * img_w
        erase_area = area * random.uniform(self.scale[0], self.scale[1])
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
        h = int(round(math.sqrt(erase_area * aspect_ratio)))
        w = int(round(math.sqrt(erase_area / aspect_ratio)))
        return h,w
    

    def _get_bool_mask(self,img,h,w,i,j):
        mask = torch.zeros_like(img, dtype=torch.int)
        mask[..., i:i + w, j:j + h] = 1
        return mask.bool()
    

    def _get_erased_coordinates(self, img_h, img_w, h, w):
        i = -1  # Begin with -1 so that the while loop is always entered
        j = -1
        while (i < 0) or (i + h > img_h) or (j < 0) or (j + w > img_w):  # Enter a loop until shape is in bounds
            i, j = self._get_coordinates(img_h, img_w, h, w)
        return i, j
    

    def _get_coordinates(self, img_h, img_w, h, w):
        settings_map = {
            'random': lambda img_h, img_w, h, w: (random.randint(0, img_h - h), random.randint(0, img_w - w)),
            'centred': lambda img_h, img_w, h, w: (int((img_h - h) / 2), int((img_w - w) / 2)),
            'near_centre': lambda img_h, img_w, h, w: (int((img_h - h) / 2) + int(random.normalvariate(0, int(img_h / (4 * 5)))),
                            int((img_w - w) / 2) + int(random.normalvariate(0, int(img_w / (4 * 5))))),
            'periphery': self._get_periphery_coordinates,
            'corners': self._get_corner_coordinates,
            'near_corners': self._get_near_corner_coordinates,
            'near_periphery': self._get_near_periphery_coordinates,
        }
        if self.setting in settings_map:
            return settings_map[self.setting](img_h, img_w, h, w)
        else:
            raise ValueError('Invalid setting for Random Erasing')
        

    def _get_periphery_coordinates(self, img_h, img_w, h, w):
        side = random.randint(0, 3)
        if side == 0:
            return 0, random.randint(0, img_w - w)
        elif side == 1:
            return img_h - h, random.randint(0, img_w - w)
        elif side == 2:
            return random.randint(0, img_h - h), 0
        else:
            return random.randint(0, img_h - h), img_w - w
    

    def _get_corner_coordinates(self, img_h, img_w, h, w):
        corner = random.randint(0, 3)
        if corner == 0:
            return 0, 0
        elif corner == 1:
            return 0, img_w - w
        elif corner == 2:
            return img_h - h, 0
        else:
            return img_h - h, img_w - w
        

    def _get_near_corner_coordinates(self, img_h, img_w, h, w):
        corner = random.randint(0, 3)
        if corner == 0:
            return int(random.normalvariate(0, int(img_h / (4 * 5)))
                       ),int(random.normalvariate(0, int(img_w / (4 * 5))))
        elif corner == 1:
            return int(random.normalvariate(0, int(img_h / (4 * 5)))
                       ),img_w - w + int(random.normalvariate(0, int(img_w / (4 * 5))))
        elif corner == 2:
            return img_h - h + int(random.normalvariate(0, int(img_h / (4 * 5)))
                                   ),int(random.normalvariate(0, int(img_w / (4 * 5))))
        else:
            return img_h - h + int(random.normalvariate(0, int(img_h / (4 * 5)))
                                   ),img_w - w + int(random.normalvariate(0, int(img_w / (4 * 5))))
        

    def _get_near_corner_coordinates(self, img_h, img_w, h, w):
        corner = random.randint(0, 3)
        if corner == 0:
            return int(random.normalvariate(0, int(img_h / (4 * 5)))
                       ),int(random.normalvariate(0, int(img_w / (4 * 5))))
        elif corner == 1:
            return int(random.normalvariate(0, int(img_h / (4 * 5)))
                       ),img_w - w + int(random.normalvariate(0, int(img_w / (4 * 5))))
        elif corner == 2:
            return img_h - h + int(random.normalvariate(0, int(img_h / (4 * 5)))
                                   ),int(random.normalvariate(0, int(img_w / (4 * 5))))
        else:
            return img_h - h + int(random.normalvariate(0, int(img_h / (4 * 5)))
                                   ),img_w - w + int(random.normalvariate(0, int(img_w / (4 * 5))))


    def _get_near_periphery_coordinates(self, img_h, img_w, h, w):
        side = random.randint(0, 3)
        if side == 0:
            return int(random.normalvariate(0, int(img_h / (4 * 5)))), random.randint(0, img_w - w)
        elif side == 1:
            return img_h - h + int(random.normalvariate(0, int(img_h / (4 * 5)))), random.randint(0, img_w - w)
        elif side == 2:
            return random.randint(0, img_h - h), int(random.normalvariate(0, int(img_w / (4 * 5))))
        else:
            return random.randint(0, img_h - h), img_w - w + int(random.normalvariate(0, int(img_w / (4 * 5))))
        

class RandomErasing_triangle(RandomErasing_square):
    """
    Class that randomly erases a triangle patch from an image. Subclass that inherits from RandomErasing_square.

    Parameters:
    ------------
        p : float
        probability that the operation will be performed.
    scale : Tuple[float,float]
        range of proportion of erased area against input image.
    ratio : Tuple[float,float]
        range of aspect ratio of erased area.
    value : str
        erasing value. Default is '0'.
        If float, it is used to erase the area.
        If str, it must be one of the following:
            'random_gaussian_noise': random Gaussian noise with mean (self.noise_mean) and std (self.noise_std).
            'random_uniform_noise': random uniform noise between minimum and maximum value of image.
            'foreign_texture': uses foreign texture (self.foreign_texture).
            'image_replace': uses another part of the image.
            'image_replace_no_overlap': uses another part of the image, but without overlapping with the erased area.
    setting : str
        Position of erased area. Default is 'random'. Options are:
            'random': random position of square patch
            'centred': square patch in the centre
            'near_centre': square patch near the center (Gaussian distribution)
            'periphery': square patch in the periphery (image outline)
            'corners': square patch in the corners
            'near_corners': square patch near the corners (Gaussian distribution)
            'near_periphery': square patch near the periphery (Gaussian distribution)
    noise_mean : str
        Mean of Gaussian noise. Default is 'img_mean', or should be a float.
    noise_std : str
        Standard deviation of Gaussian noise. Default is 'img_std', or should be a float.
    noise_coarseness : float
        Coarseness of noise or foreign texture (>=1). Default is 1.
    rotation_angle : float
        Angle of rotation of the erased area. Default is 0. 'random' will rotate the erased area by a random angle.
    foreign_texture : torch.tensor
        Foreign texture (2D or 3D) to be used to fill the erased area. Default is a 10x10 checkerboard pattern.
    gaussian_filter_sigma : float
        Standard deviation of Gaussian filter to be applied to the erased area to smooth the edges of the erased area. Default is 0.
    make_transparent : bool
        Whether to make the erased area transparent. Default is False.
    transparency_power : float
        Power to raise the transparency power to. Default is 5.
    triangle_type : str
        Type of triangle to be erased. Default is 'equilateral'. Options are:
            'equilateral': equilateral triangle
            'scaline_60': scaline triangle with 60 degree angle
            'scaline': scaline triangle with random angle
            'right_angle': right angle triangle
    """

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        kwargs.setdefault('triangle_type','equilateral')
        self.triangle_type=kwargs['triangle_type']

    def _get_bool_mask(self,img,h,w,i,j):
        if self.triangle_type=='equilateral':
            mask = skimage.draw.polygon2mask((img.shape[-2],img.shape[-1]),np.array([[j+h, i], [j, i+(w/2)], [j+h, i+w]]))
        elif self.triangle_type=='scaline_60':
            mask = skimage.draw.polygon2mask((img.shape[-2],img.shape[-1]),np.array([[j+h, i], [j, int(i+np.random.randint(0,w))], [j+h, i+w]]))
        elif self.triangle_type=='scaline':
            mask = skimage.draw.polygon2mask((img.shape[-2],img.shape[-1]),np.array([[int(j+np.random.randint(0,h)), i], [j, int(i+np.random.randint(0,w))], [j+h, i+w]]))
        elif self.triangle_type=='right_angle':
            mask = skimage.draw.polygon2mask((img.shape[-2],img.shape[-1]),np.array([[j+h, i], [j, i], [j+h, i+w]]))
        else:
            raise ValueError('triangle_type must be one of "equilateral", "scaline_60", "scaline", "right_angle"')

        mask = torch.from_numpy(mask.astype(int))
        mask=mask.unsqueeze(0)
        mask=mask.expand(img.size(0),-1,-1)

        return mask.bool()
    

class RandomErasing_polygon(RandomErasing_square):
    """
    Class that randomly erases a polygon patch from an image. Subclass that inherits from RandomErasing_square.

    Parameters:
    ------------
    p : float
        probability that the operation will be performed.
    scale : Tuple[float,float]
        range of proportion of erased area against input image.
    ratio : Tuple[float,float]
        range of aspect ratio of erased area.
    value : str
        erasing value. Default is '0'.
        If float, it is used to erase the area.
        If str, it must be one of the following:
            'random_gaussian_noise': random Gaussian noise with mean (self.noise_mean) and std (self.noise_std).
            'random_uniform_noise': random uniform noise between minimum and maximum value of image.
            'foreign_texture': uses foreign texture (self.foreign_texture).
            'image_replace': uses another part of the image.
            'image_replace_no_overlap': uses another part of the image, but without overlapping with the erased area.
    setting : str
        Position of erased area. Default is 'random'. Options are:
            'random': random position of square patch
            'centred': square patch in the centre
            'near_centre': square patch near the center (Gaussian distribution)
            'periphery': square patch in the periphery (image outline)
            'corners': square patch in the corners
            'near_corners': square patch near the corners (Gaussian distribution)
            'near_periphery': square patch near the periphery (Gaussian distribution)
    noise_mean : str
        Mean of Gaussian noise. Default is 'img_mean', or should be a float.
    noise_std : str
        Standard deviation of Gaussian noise. Default is 'img_std', or should be a float.
    noise_coarseness : float
        Coarseness of noise or foreign texture (>=1). Default is 1.
    rotation_angle : float
        Angle of rotation of the erased area. Default is 0. 'random' will rotate the erased area by a random angle.
    foreign_texture : torch.tensor
        Foreign texture (2D or 3D) to be used to fill the erased area. Default is a 10x10 checkerboard pattern.
    gaussian_filter_sigma : float
        Standard deviation of Gaussian filter to be applied to the erased area to smooth the edges of the erased area. Default is 0.
    make_transparent : bool
        Whether to make the erased area transparent. Default is False.
    transparency_power : float
        Power to raise the transparency power to. Default is 5.
    polygon_coordinates : numpy.ndarray
        Coordinates of the polygon to be erased. Default is a 10-sided polygon.
    """
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        kwargs.setdefault('polygon_coordinates', np.array([(112, 50), (118, 80), (142, 80), (124, 98), (140, 122),
    (112, 110), (84, 122), (100, 98), (82, 80), (106, 80)]))
              
        if isinstance(kwargs['polygon_coordinates'],(type(np.array([])),type(np.ndarray([])))) is False:
           raise ValueError('polygon_coordinates must be a numpy array')                    

        self.polygon_coordinates=kwargs['polygon_coordinates']

    def _get_bool_mask(self,img,h,w,i,j):
        mask = skimage.draw.polygon2mask((img.shape[-2],img.shape[-1]),self.polygon_coordinates)
        mask = torch.from_numpy(mask.astype(int))
        mask=mask.unsqueeze(0)
        mask=mask.expand(img.size(0),-1,-1)

        return mask.bool()
    

class RandomErasing_ring(RandomErasing_square):
    """
    Class that randomly erases a ring patch from an image. Subclass that inherits from RandomErasing_square.

    Parameters:
    ------------
    p : float
        probability that the operation will be performed.
    scale : Tuple[float,float]
        range of proportion of erased area against input image.
    ratio : Tuple[float,float]
        range of aspect ratio of erased area.
    value : str
        erasing value. Default is '0'.
        If float, it is used to erase the area.
        If str, it must be one of the following:
            'random_gaussian_noise': random Gaussian noise with mean (self.noise_mean) and std (self.noise_std).
            'random_uniform_noise': random uniform noise between minimum and maximum value of image.
            'foreign_texture': uses foreign texture (self.foreign_texture).
            'image_replace': uses another part of the image.
            'image_replace_no_overlap': uses another part of the image, but without overlapping with the erased area.
    setting : str
        Position of erased area. Default is 'random'. Options are:
            'random': random position of square patch
            'centred': square patch in the centre
            'near_centre': square patch near the center (Gaussian distribution)
            'periphery': square patch in the periphery (image outline)
            'corners': square patch in the corners
            'near_corners': square patch near the corners (Gaussian distribution)
            'near_periphery': square patch near the periphery (Gaussian distribution)
    noise_mean : str
        Mean of Gaussian noise. Default is 'img_mean', or should be a float.
    noise_std : str
        Standard deviation of Gaussian noise. Default is 'img_std', or should be a float.
    noise_coarseness : float
        Coarseness of noise or foreign texture (>=1). Default is 1.
    rotation_angle : float
        Angle of rotation of the erased area. Default is 0. 'random' will rotate the erased area by a random angle.
    foreign_texture : torch.tensor
        Foreign texture (2D or 3D) to be used to fill the erased area. Default is a 10x10 checkerboard pattern.
    gaussian_filter_sigma : float
        Standard deviation of Gaussian filter to be applied to the erased area to smooth the edges of the erased area. Default is 0.
    make_transparent : bool
        Whether to make the erased area transparent. Default is False.
    transparency_power : float
        Power to raise the transparency power to. Default is 5.
    ellipse_parameter : float
        Ellipcticity of the ring (<=1). Default is 1.
    ring_width : float
        Width of the ring. Default is 20.
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        # Set default values for the arguments if they are not provided
        kwargs.setdefault('ellipse_parameter', 1)
        kwargs.setdefault('ring_width', 20)

        self.width=kwargs['ring_width']
        if kwargs['ellipse_parameter'] > 1:
            raise ValueError('ellipse must be less than 1')
        self.ellipse=kwargs['ellipse_parameter']

    def _get_bool_mask(self,img,h,w,i,j):
        canvas = PIL.Image.new('1', ((img.shape[-1], img.shape[-2])))
        letter_canvas = PIL.Image.new('1', (int(h*self.ellipse+1),int(w*1/self.ellipse)))
        draw = PIL.ImageDraw.Draw(letter_canvas)
        draw.ellipse([(0,0), (h*self.ellipse,w*1/self.ellipse)],fill=None,outline='white',width=self.width)
        transform = T.Compose([T.ToTensor()])

        # Adjust i and j to keep the ellipse centered
        ellipse_width = h * (1-self.ellipse)  
        ellipse_height = w * (1-1/self.ellipse) 
        i += int(ellipse_width / 2)
        j += int(ellipse_height / 2)
        adjusted_i = max(0, min(i, canvas.size[0] - letter_canvas.size[0]))
        adjusted_j = max(0, min(j, canvas.size[1] - letter_canvas.size[1]))
        pos=(int(adjusted_i),int(adjusted_j))
        canvas.paste(letter_canvas,pos)
        canvas = canvas.resize((img.shape[-1], img.shape[-2]), PIL.Image.LANCZOS)

        transform = T.Compose([T.ToTensor()])
        mask = transform(canvas).expand(img.size(0),-1,-1)

        return mask.bool()


class RandomErasing_text(RandomErasing_square):
    """
    Class that randomly erases a text patch from an image. Subclass that inherits from RandomErasing_square.

    Parameters:
    ------------
    p : float
        probability that the operation will be performed.
    scale : Tuple[float,float]
        range of proportion of erased area against input image.
    ratio : Tuple[float,float]
        range of aspect ratio of erased area.
    value : str
        erasing value. Default is '0'.
        If float, it is used to erase the area.
        If str, it must be one of the following:
            'random_gaussian_noise': random Gaussian noise with mean (self.noise_mean) and std (self.noise_std).
            'random_uniform_noise': random uniform noise between minimum and maximum value of image.
            'foreign_texture': uses foreign texture (self.foreign_texture).
            'image_replace': uses another part of the image.
            'image_replace_no_overlap': uses another part of the image, but without overlapping with the erased area.
    setting : str
        Position of erased area. Default is 'random'. Options are:
            'random': random position of square patch
            'centred': square patch in the centre
            'near_centre': square patch near the center (Gaussian distribution)
            'periphery': square patch in the periphery (image outline)
            'corners': square patch in the corners
            'near_corners': square patch near the corners (Gaussian distribution)
            'near_periphery': square patch near the periphery (Gaussian distribution)
    noise_mean : str
        Mean of Gaussian noise. Default is 'img_mean', or should be a float.
    noise_std : str
        Standard deviation of Gaussian noise. Default is 'img_std', or should be a float.
    noise_coarseness : float
        Coarseness of noise or foreign texture (>=1). Default is 1.
    rotation_angle : float
        Angle of rotation of the erased area. Default is 0. 'random' will rotate the erased area by a random angle.
    foreign_texture : torch.tensor
        Foreign texture (2D or 3D) to be used to fill the erased area. Default is a 10x10 checkerboard pattern.
    gaussian_filter_sigma : float
        Standard deviation of Gaussian filter to be applied to the erased area to smooth the edges of the erased area. Default is 0.
    make_transparent : bool
        Whether to make the erased area transparent. Default is False.
    transparency_power : float
        Power to raise the transparency power to. Default is 5.
    text : str
        Text to be used to fill the erased area. Default is 'OOD'.
    font_family : str
        Font family to be used to fill the erased area. Default is 'sans-serif'.
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        # Set default values for the arguments if they are not provided
        kwargs.setdefault('text', 'OOD')  # Default text is 'OOD'
        kwargs.setdefault('font_family', 'sans-serif')

        self.text = kwargs['text']
        self.font_family = kwargs['font_family']


    def _get_bool_mask(self,img,h,w,i,j):
        font_size = 1
        canvas = PIL.Image.new('1', ((img.shape[-1], img.shape[-2])))
        letter_canvas = PIL.Image.new('1', (int(h),int(w)))

        # Find the largest font size that fits the erased area
        font_size = 1
        while True:
            font_type = font_manager.FontProperties(family=self.font_family, weight='regular')
            file = font_manager.findfont(font_type)
            font = PIL.ImageFont.truetype(file, font_size)
            text_width, text_height = font.getsize(self.text)
            if text_width >= letter_canvas.width or text_height >= letter_canvas.height:
                break  # Exit the loop if the letter fits
            font_size += 1

        # Draw the letter
        draw = PIL.ImageDraw.Draw(letter_canvas)
        draw.text(((letter_canvas.width - text_width) / 2, (letter_canvas.height - text_height) / 2), self.text, font=font, fill='white')
        transform = T.Compose([T.ToTensor()])
        pos=(int(i),int(j))
        canvas.paste(letter_canvas,pos)
        canvas = canvas.resize((img.shape[-1], img.shape[-2]), PIL.Image.LANCZOS)
        transform = T.Compose([T.ToTensor()])
        mask = transform(canvas).expand(img.size(0),-1,-1)

        return mask.bool()


class add_Gaussian_noise(torch.nn.Module):
    """
    Adds Gaussian noise to the image.

    Parameters
    ----------
    p : float
        Probability of applying the transform. Default value is 0.5.
    noise_mean : float
        Mean of the Gaussian distribution added to image. Default value is 0.
    noise_std : float
        Standard deviation of the Gaussian distribution added to image. Default value is 0.1.
    """
    def __init__(self, p:float=0.5, noise_mean:float=0, noise_std:float=0.1, **kwargs):
        super().__init__()
        self.p = p
        self.mean = noise_mean
        self.std = noise_std

    def forward(self, img:torch.Tensor):
        """
        Erases the shape from an image.

        Parameters
        ----------
        img : torch.Tensor
            Image to be transformed.

        Returns
        -------
        torch.Tensor
            Transformed image.
        """
        if random.uniform(0, 1) < self.p:
            img = self.erase(img)
        return img

    def erase(self, img:torch.Tensor):
        """
        Erases the shape from an image.

        Parameters
        ----------
        img : torch.Tensor
            Image to be transformed.

        Returns
        -------
        torch.Tensor
            Transformed image.
        """
        noise_mean = float(self.mean) if self._check_if_string_is_number(self.mean) == True else  torch.mean(img.flatten()).item()
        noise_std = float(self.std) if self._check_if_string_is_number(self.std) == True else  torch.std(img.flatten()).item()
        return img + torch.normal(float(noise_mean), float(noise_std), size=img.shape
                                  )
    
    def _check_if_string_is_number(self,s):
        try:
            float(s) or int(s)
            return True
        except ValueError:
            return False
