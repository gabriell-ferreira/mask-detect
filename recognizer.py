import cv2
import os
import numpy
import pandas
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import warnings

# Carrega um dataframe da biblioteca Pandas com as imagens para treinamento
def load_data_frame():
  # Cria um dict para organizar os dados
  data = {
    "FILE": [],
    "LABEL": [],
    "TARGET": [],
    "IMAGE": []
  }

  # Carrega as imagens com mascara e sem mascara
  mask = os.listdir(f"images{os.sep}maskon")
  no_mask = os.listdir(f"images{os.sep}maskoff")

  # Percorre o diretorio de imagens com mascara e preenche o dict data com as informações de cada imagem
  for file in mask:
    data["FILE"].append(f"images{os.sep}maskon{os.sep}{file}")
    data["LABEL"].append(f"Com mascara")
    data["TARGET"].append(1)
    image = cv2.cvtColor(cv2.imread(f"images{os.sep}maskon{os.sep}{file}"), cv2.COLOR_BGR2GRAY).flatten()
    data["IMAGE"].append(image)

  # Percorre o diretorio de imagens sem mascara e preenche o dict data com as informações de cada imagem
  for file in no_mask:
    data["FILE"].append(f"train{os.sep}maskoff{os.sep}{file}")
    data["LABEL"].append(f"Sem mascara")
    data["TARGET"].append(0)
    image = cv2.cvtColor(cv2.imread(f"images{os.sep}maskoff{os.sep}{file}"), cv2.COLOR_BGR2GRAY).flatten()
    data["IMAGE"].append(image)

  return pandas.DataFrame(data)

# Divide o dataframe para treino e teste
def train_test(data_frame):
  X = list(data_frame["IMAGE"])
  Y = list(data_frame["TARGET"])

  return X, Y

# Calcula a projeção dos dados em um vetor que maximize a variança dos dados e perca a menor quantidade de informação possível e realiza a extração de features das imagens
def pca_model(X_train):
  pca = PCA(n_components=30)
  pca.fit(X_train)

  return pca

# Prever valores de quaisquer novos pontos de dados. O novo ponto recebe um valor baseado em quão próximo ele se parece dos pontos no conjunto de treinamento
def knn(X_train, Y_train):
  warnings.filterwarnings("ignore")

  grid_params = {
    "n_neighbors": [2, 3, 5, 11, 19, 23, 29],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattam", "cosine", "l1", "l2"]
  }

  knn_model = GridSearchCV(KNeighborsClassifier(), grid_params, refit=True)
  knn_model.fit(X_train, Y_train)

  return knn_model

# Define o path xml para pré-treino com o Cascade Classifier, reconhecendo rostos de forma genérica
haar_file = 'haarcascade_frontalface_alt2.xml'
# Define o path do dataset para treinos
dataset = 'dataset'

# Cria uma lista de imagens e um lista com os nomes correspondentes
images = []
labels = []
names = {}
id = 0

# Percorre o diretório de datasets e preenche o array de imagens e nomes com os arquivos correspondentes
for (subdirs, dirs, files) in os.walk(dataset):
  for subdir in dirs:
    names[id] = subdir
    subdir_path = os.path.join(dataset, subdir)
    for file_name in os.listdir(subdir_path):
      path = subdir_path + "/" + file_name
      label = id
      images.append(cv2.imread(path, 0))
      labels.append(int(label))
    id += 1

# Define o tamanho da área de reconhecimento
width = 130
height = 100

# Cria um array numpy para auxiliar nas comparações por conta de seus métodos
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# Treina o modelo com as imagens do dataset
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Importa o XML para auxiliar no reconhecimento
face_cascade = cv2.CascadeClassifier(haar_file)

# Instancia a webcam
webcam = cv2.VideoCapture(0)

# Carrega o dataframe com as imagens para treinamento
data_frame = load_data_frame()

# Divide conjuntos de treino e teste
X_train, y_train = train_test(data_frame)

# Modelo PCA para extração de features da imagem
pca = pca_model(X_train)

# Conjunto de treino com features extraídas
X_train = pca.transform(X_train)

# Treinando modelo classificatório KNN.
knn = knn(X_train, y_train)

# Rotulo para classificação
label = {
  0: "Sem mascara",
  1: "Com mascara"
}

while True:
  # Inicia a camera
  (_, image) = webcam.read()

  # Converte a imagem capturada pela camera em cinza
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Retorna um retangulo com as coordenadas ao redor da face encontrada
  faces = face_cascade.detectMultiScale(gray)

  # Fecha o loop ao apertar Q
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
  # Percorre as informações das faces encontradas
  for (x, y, w, h) in faces:
    face = gray[y:y + h, x:x + w]
    face_resize = cv2.resize(face, (width, height))

    # Retorna a porcentagem de similaridade da face detectada com o dataset
    prediction = model.predict(face_resize)

    classification = ""
    color = (0, 255, 0)

    # Lógica para dizer se está com mascara ou nao
    if face.shape[0] >= 200 and face.shape[1] >= 200:
      #Extrai as features da imagem.
      vector = pca.transform([face_resize.flatten()]) 

      # Tenta identificar se está com máscara ou não.
      pred = knn.predict(vector)[0] 

      # Busca a label conforme a identificação.
      classification = label[pred] 

      # Alterando a cor do retangulo caso esteja sem mascara.
      if pred == 0:
        color = (0,0,255)

    # Mostra um retângulo ao redor do rosto do usuário.
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

     # O valor que calibra o reconhecimento, quanto menor o valor, mais preciso é a leitura da imagem gravada no treinamento.
    if prediction[1] < 120:
      # Coloca o texto acima da área reconhecida
      cv2.putText(image, '% s - %.0f - % s' % (names[prediction[0]], prediction[1], classification), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color)
    else:
      # Se a face não for reconhecida, mostra o desconhecido e detecta se está com máscara ou sem
      cv2.putText(image, 'Desconhecido - % s' % (classification), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color)

  # Exibir a imagem da camera em uma janela.
  cv2.imshow('Leitura Facial', image)
