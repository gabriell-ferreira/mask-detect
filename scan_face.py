import os
import shutil
import cv2

# Define o path do dataset e XML do Cascade Classifier
dataset_path = 'dataset'
haar_file = 'haarcascade_frontalface_alt2.xml'

# Captura o nome do usuario para reconhecimento
user_name = input("insira seu nome: ")

# Define o subdiretorio do dataset com o nome do usuario
sub_dir = user_name

# Monta o path do dataset com o nome do usuario informado
path = f"{dataset_path}/{user_name}"

# Verifica se existe um diretorio com mesmo nome e exclui se sim
if os.path.isdir(path):
  shutil.rmtree(path)
  
# Lista os arquivos do dataset
os.mkdir(path)

# Define o tamanho da area de reconhecimento
width = 130
height = 100

# Inicia o treino com faces genericas do XML
face_cascade = cv2.CascadeClassifier(haar_file)

# Instacia a webcam
webcam = cv2.VideoCapture(0)

# Faz a captura de 30 fotos da face do usuario para guardar no dataset de treino
count = 1
while count < 30:
  # Faz a leitura da imagem da camera
  (_, image) = webcam.read()

  # Converte cada imagem para cinza
  gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

  # Realiza a detecção da face e retorna um retangulo com as coordenadas ao redor da face
  faces = face_cascade.detectMultiScale(gray, 1.3, 4)

  # Percorre a face detectada
  for (x, y, w, h) in faces:
    # Exibe o retangulo com ao redor de uma face detectada
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = gray[y:y + h, x:x + w]

    # Realiza o redimensionamento da imagem
    face_resize = cv2.resize(face, (width, height))

    # Salva a imagem no dataset com nome do usuario
    cv2.imwrite('% s/% s.png' % (path, count), face_resize)
  
  count += 1

  # Exibe a janela com a imagem da camera
  cv2.imshow('OpenCV', image)

  # Encerra a aplicação ao apertar Q
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break