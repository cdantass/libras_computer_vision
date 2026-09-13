# Reconhecimento de Libras com Visão Computacional

Projeto de reconhecimento de sinais estáticos do alfabeto em Libras (Língua Brasileira de Sinais) usando webcam, MediaPipe Hands e um classificador de Machine Learning.

O sistema detecta a mão pela câmera, extrai os pontos-chave (landmarks) da mão e classifica qual letra do alfabeto está sendo sinalizada em tempo real.

## Como funciona

O projeto segue um pipeline de 3 etapas:

```
dataset de imagens  ->  extração de landmarks  ->  treinamento  ->  reconhecimento em tempo real
   (train/test)         extract_landmarks.py       train_model.py         hand.py
```

Em vez de treinar um modelo diretamente sobre pixels de imagem (o que exigiria muito mais dados e poder computacional), o projeto usa o MediaPipe Hands para extrair 21 pontos da mão (pontas dos dedos, juntas, pulso) de cada imagem. Cada ponto tem coordenadas (x, y, z), totalizando 63 valores numéricos por imagem — esse vetor é o que o classificador aprende a reconhecer.

Essa abordagem é mais leve, roda em CPU comum, e generaliza melhor com datasets menores do que treinar uma CNN do zero em cima das imagens.

## Estrutura do projeto

```
projectlibras/
├── extract_landmarks.py       # Converte imagens em landmarks (gera o CSV)
├── train_model.py             # Treina o classificador Random Forest
├── hand.py                    # Reconhecimento em tempo real via webcam
├── hand_landmarks_dataset.csv # Dataset já processado (landmarks + label)
├── hand_gesture_model.pkl     # Modelo já treinado
├── requirements.txt           # Dependências do projeto
└── readME.md                  # Este arquivo
```

## Dataset

Baseado no Libras Dataset (Kaggle - williansoliveira/libras), organizado em pastas por letra e já dividido em `train/` e `test/`:

```
dataset/
├── train/
│   ├── a/
│   ├── b/
│   └── ...
└── test/
    ├── a/
    ├── b/
    └── ...
```

**Letras cobertas:** A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W, Y (21 classes, 45.601 amostras processadas).

> As letras **H, J, K, X e Z** não estão incluídas porque, em Libras, esses sinais envolvem **movimento** — não é possível representá-los com uma única imagem estática. Reconhecê-los exigiria uma abordagem diferente, analisando uma sequência de frames ao longo do tempo (ex: usando uma rede recorrente como LSTM sobre uma janela de landmarks).

## Requisitos

```bash
pip install -r requirements.txt
```

Dependências principais:
- `mediapipe` — detecção da mão e extração de landmarks
- `opencv-python` — captura de webcam e processamento de imagem
- `scikit-learn` — treinamento do classificador (Random Forest)
- `pandas` / `numpy` — manipulação de dados
- `joblib` — salvar/carregar o modelo treinado
- `matplotlib` / `seaborn` — visualizações (gráficos, matriz de confusão)

## Como usar

### 1. Extrair os landmarks do dataset

Baixe o dataset e extraia numa pasta `dataset/` na raiz do projeto (estrutura `dataset/train/<letra>/` e `dataset/test/<letra>/`). Depois rode:

```bash
python extract_landmarks.py
```

Isso gera o `hand_landmarks_dataset.csv` com os landmarks de todas as imagens processadas.

### 2. Treinar o modelo

```bash
python train_model.py
```

Treina um `RandomForestClassifier` (200 árvores) sobre 80% dos dados, testa nos 20% restantes e imprime a acurácia. O modelo treinado é salvo em `hand_gesture_model.pkl`.

### 3. Rodar o reconhecimento em tempo real

```bash
python hand.py
```

Abre a webcam, desenha o esqueleto da mão detectada e mostra na tela a letra reconhecida. Pressione **ESC** para sair.

> **Nota:** o `hand.py` atual tem um bug na linha do `cv2.flip` (usa a variável `frame` antes dela existir — deveria ser `image`, que é o nome retornado por `cap.read()`). Corrija essa linha antes de rodar, ou o script vai travar com `NameError`.

## Limitações atuais

- Reconhece apenas **sinais estáticos** (uma pose fixa da mão), não sinais com movimento
- Detecta apenas **uma mão por vez** (`max_num_hands=1`)
- Sensível a variações de iluminação e ângulo de câmera não representadas no dataset de treino
- Não há validação de que o sinal está sendo mantido por um tempo mínimo antes de confirmar o reconhecimento (é uma predição por frame, podendo "piscar" entre letras)

## Possíveis evoluções

- Corrigir o bug do `hand.py` e adicionar lógica de "sustentar o sinal por X segundos" antes de confirmar uma letra
- Suporte a sinais dinâmicos (H, J, K, X, Z e palavras completas) usando sequência de frames
- Interface gráfica/web em vez de janela OpenCV pura, com feedback visual de progresso
- Expandir o dataset com gravações próprias para melhorar robustez em diferentes condições de câmera
