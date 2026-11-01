import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2

#Toma todos los colores existentes en la imagen
def get_colors(image):
  aux = []
  band = True
  for i in image:
    for j in i:
      for k in aux:
        if j.tolist() == k:
          band = False
          break
      if band:
        aux.append(j.tolist())
      band = True
  return np.array(aux)

#Obtiene los n colores principales mediante kmeans
def principal_colors(img, n_colors):
  photo = img
  photo = photo/255
  w,h,d = tuple(photo.shape)
  #assert d == 3
  photo = photo.reshape(w*h, d)
  photo_sample = shuffle(photo, random_state=0)[-1]
  kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(photo_sample)
  labels = kmeans.predict(photo)
  return labels, kmeans

#Recrea una imagen comprimida por la función de kmaens por un codebook
def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

#Función para transformar y obtener los colores principales de una imagen
def n_colors_img(img, n_colors):
  labels, model = principal_colors(img, n_colors)
  w,h,d = tuple(img.shape)
  print(labels)
  res = recreate_image(model.cluster_centers_, labels, w, h)
  res = np.round(res*255).astype(np.uint8)
  return res

#Función para dada una paleta solo tomar los colores de esa paleta en la imagen
def quantizetopalette(silf, palette, dither=False, mode="P"):
  """Convert an RGB or L mode image to use a given P image's palette."""
  silf.load()
  palette.load()
  im = silf.im.convert(mode, 0, palette.im)
  # the 0 above means turn OFF dithering making solid colors
  return silf._new(im)


#Realiza las operaciones necesarias para obtener una imagen RGB por una paleta de colores
def rgb_quantized(img, palette):
  rows, cols = len(img), len(img[0])
  total_vals = 1
  for i in palette.shape:
    total_vals *= i
  palettedata = palette.reshape(total_vals).tolist()
  palImage = Image.new('P', (rows, cols))
  palImage.putpalette(palettedata*32)
  oldImage = Image.fromarray(img).convert("RGB")
  newImage = quantizetopalette(oldImage,palImage)
  res_image = np.asarray(newImage.convert("RGB"))
  return res_image

def gray_quantized(img, palette):
  rows, cols = len(img), len(img[0])
  total_vals = 1
  for i in palette.shape:
    total_vals *= i
  palettedata = palette.reshape(total_vals).tolist()
  palImage = Image.new('L', (rows, cols))
  palImage.putpalette(palettedata*32)
  oldImage = Image.fromarray(img, 'L')
  newImage = quantizetopalette(oldImage,palImage, mode="L")
  res_image = np.asarray(newImage)
  return res_image

#Transforma una imagen en escala de grises a RGB dada una paleta de colores
def gray_to_rgb_by_pallete(img, palette):
  img_float32 = np.float32(palette)
  palette_gray = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
  rows = len(img)
  cols = len(img[0])
  new_img = np.zeros((rows, cols,3)) #3 color channels
  for i in range(rows):
    for j in range(cols):
      index = np.argmin(np.abs(palette_gray-img[i][j]))
      new_img[i][j] = palette[index]
  return new_img

def balance_img_categories(img, palette, balancer):
  #palette = np.sort(palette)
  rows = len(img)
  cols = len(img[0])
  for i in range(rows):
    for j in range(cols):
      pos = np.where(palette == img[i,j])[0][0]
      img[i,j] = balancer[pos]
  return img

def recolor_greys_image(data, palette):
    rows, cols = len(data), len(data[0])
    aux = np.zeros((rows, cols), dtype=np.uint64)
    for i in range(rows):
        for j in range(cols):
            aux[i,j] = min(palette, key= lambda x:abs(x-data[i,j]))
    return aux