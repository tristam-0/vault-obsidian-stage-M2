### 2. D'où vient ce "12%" ? Ce n'est pas arbitraire, c'est statistique

Ce choix n'est pas sorti d'un chapeau, il vient directement de la distribution des données du dataset de référence en vision par ordinateur : **COCO (Common Objects in Context)**.

Dans la littérature scientifique, les objets sont classés en trois catégories selon leur taille en pixels :

- **Small (Petits) :** moins de 32×32 pixels.
    
- **Medium (Moyens) :** entre 32×32 et 96×96 pixels.
    
- **Large (Grands) :** plus de 96×96 pixels.
    

Sur le dataset COCO, **plus de 40% des objets entrent dans la catégorie "Small"**, et environ 35% dans la catégorie "Medium". Les grands objets sont minoritaires.

Si on calcule la taille moyenne d'un côté (w ou h) de ces objets par rapport à la taille standard d'une image (640×640 pixels), on tombe mathématiquement dans une fourchette située **entre 5% et 15%**.



Mais exsitance dans le code
class RandomSizeCrop(object):

def __init__(self, min_size: int, max_size: int):

self.min_size = min_size

self.max_size = max_size

  

def __call__(self, img: PIL.Image.Image, target: dict):

w = random.randint(self.min_size, min(img.width, self.max_size))

h = random.randint(self.min_size, min(img.height, self.max_size))

region = T.RandomCrop.get_params(img, [h, w])

return crop(img, target, region)


dans une taill en % ne fais pas de sense