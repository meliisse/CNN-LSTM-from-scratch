import numpy as np
from collections import Counter
import random


""" 1) pretraitement:
     - construire le vocabulaire des phrases donner
     - ceci est fait pour pouvoir decoder/encoder de vecteur a mot et vise versa
"""

def build_vocab(sentences, min_freq=1):
    words = []
    for s in sentences:
        words.extend(s.split())
    counts = Counter(words)
    vocab = [w for w, c in counts.items() if c >= min_freq]

    # reserve 0 -> PAD (padding), 1 -> UNK(pour mots inconnus)

    word2idx = {"<PAD>": 0, "<UNK>": 1}

    idx = 2

    """index unique a chaque mot, on saute les deux premieres cases
    (reserver a <UNK>et <PAD>)
    """
    for w in vocab:
        if w not in word2idx:
            word2idx[w] = idx
            idx += 1
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word
#

def texts_to_sequences(sentences, word2idx):
    seqs = []
    for s in sentences:
        seqs.append([word2idx.get(w, word2idx["<UNK>"]) for w in s.split()])
    return seqs

""" Elle transforme un texte brut (ex. des articles médicaux)
en exemples d’entraînement pour apprendre à prédire le mot suivant
"""
def create_examples(seqs, seq_len):
    X, Y = [], []
    for seq in seqs:
        if len(seq) < seq_len + 1:
            continue
        for i in range(len(seq) - seq_len):
            X.append(seq[i:i+seq_len])
            Y.append(seq[i+seq_len])
    return np.array(X, dtype=int), np.array(Y, dtype=int)


"""2) MODEL COMPONENTS (NumPy)
    -softmax:Le softmax transforme un vecteur brut de scores (aussi appelés logits)
        en probabilités qui indiquent la chance de chaque mot d’être le suivant
    -cross_entropy_loss:
"""

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def cross_entropy_loss(probs, y_idx):
    # probs: (vocab_size,), y_idx: int
    return -np.log(probs[y_idx] + 1e-12)

# Conv1D class (filters: num_filters x kernel_size x embed_dim)
class Conv1D:
    def __init__(self, input_length, embed_dim, kernel_size, num_filters):
        self.input_length = input_length
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        # filters small init
        self.filters = np.random.randn(num_filters, kernel_size, embed_dim) * 0.01 #on multiplie par 0.01 pour éviter grands gradients
        self.biases = np.zeros((num_filters,))

    def forward(self, X_embed):
        # X_embed: (seq_len, embed_dim)
        seq_len = X_embed.shape[0]
        out_len = seq_len - self.kernel_size + 1
        out = np.zeros((out_len, self.num_filters))
        # store for backward
        self.X_embed = X_embed.copy()
        self.out_len = out_len
        for f in range(self.num_filters):
            filt = self.filters[f]  # (k, embed_dim)
            for i in range(out_len):
                seg = X_embed[i:i+self.kernel_size]  # (k, embed_dim)
                out[i, f] = np.sum(seg * filt) + self.biases[f]
        self.conv_out = out  # before ReLU
        return out  # caller will apply ReLU / pooling

    def backward(self, dconv):
        # dconv: gradient w.r.t conv_out (after ReLU applied) shape (out_len, num_filters)
        # compute gradients for filters, biases and return gradient w.r.t input embeddings
        dfilters = np.zeros_like(self.filters)
        dbiases = np.zeros_like(self.biases)
        dX = np.zeros_like(self.X_embed)  # (seq_len, embed_dim)
        for f in range(self.num_filters):
            for i in range(self.out_len):
                grad_val = dconv[i, f]  # scalar
                dbiases[f] += grad_val
                seg = self.X_embed[i:i+self.kernel_size]  # (k, embed_dim)
                # filter gradient accumulation
                dfilters[f] += grad_val * seg
                # propagate to input embeddings (each position in the segment)
                dX[i:i+self.kernel_size] += grad_val * self.filters[f]
        # update params handled outside (SGD)
        return dfilters, dbiases, dX

# Fully connected layer
class Dense:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros(output_size)
    #calcule x @ W + b
    def forward(self, x):
        # x: (input_size,)
        self.x = x.copy()
        return np.dot(self.x, self.W) + self.b
    def backward(self, dlogits):
        # dlogits: (output_size,)
        dW = np.outer(self.x, dlogits)  # (input_size, output_size)
        db = dlogits.copy()
        dx = np.dot(self.W, dlogits)  # (input_size,)
        return dW, db, dx

"""3) TRAINING FUNCTION """

def train_cnn(corpus_sentences,
              seq_len = 3,
              embedding_dim = 32,
              kernel_size = 2,
              num_filters = 32,
              epochs = 100,
              lr = 0.1,
              print_every = 10,
              batch_size = 16):
    # Build vocab
    word2idx, idx2word = build_vocab(corpus_sentences)
    vocab_size = len(word2idx)
    print("Vocab size:", vocab_size)
    seqs = texts_to_sequences(corpus_sentences, word2idx)
    X, Y = create_examples(seqs, seq_len)
    if len(X) == 0:
        raise ValueError("No training examples: lower seq_len or provide longer sentences.")
    # initialize parameters
    embeddings = (np.random.randn(vocab_size, embedding_dim) * 0.01)
    conv = Conv1D(input_length=seq_len, embed_dim=embedding_dim, kernel_size=kernel_size, num_filters=num_filters)
    fc = Dense(input_size=num_filters, output_size=vocab_size)

    # training loop (simple SGD, example-level or mini-batch)
    N = len(X)
    for epoch in range(1, epochs+1):
        perm = np.random.permutation(N)
        total_loss = 0.0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = perm[start:end]
            # accumulate gradients over batch
            dEmb_acc = np.zeros_like(embeddings)
            dFilters_acc = np.zeros_like(conv.filters)
            dBias_acc = np.zeros_like(conv.biases)
            dW_acc = np.zeros_like(fc.W)
            db_acc = np.zeros_like(fc.b)

            for idx in batch_idx:
                x_idx = X[idx]  # shape (seq_len,)
                y_idx = int(Y[idx])
                # forward
                X_embed = embeddings[x_idx]  # (seq_len, embed_dim)
                conv_out = conv.forward(X_embed)  # (out_len, num_filters)
                # ReLU
                conv_relu = np.maximum(0, conv_out)
                # Global max pooling across positions -> vector of length num_filters
                max_idxs = np.argmax(conv_relu, axis=0)  # (num_filters,)
                pooled = conv_relu[max_idxs, range(conv_relu.shape[1])]  # (num_filters,)
                # dense
                logits = fc.forward(pooled)  # (vocab_size,)
                probs = softmax(logits)
                loss = cross_entropy_loss(probs, y_idx)
                total_loss += loss

                # backward
                # gradient of logits
                dlogits = probs.copy()
                dlogits[y_idx] -= 1.0  # probs - y_onehot

                # Dense backward
                dW, db, dpooled = fc.backward(dlogits)
                dW_acc += dW
                db_acc += db

                # backprop through max-pool: only position that had the max gets gradient
                dconv_relu = np.zeros_like(conv_relu)  # (out_len, num_filters)
                for f in range(conv.num_filters):
                    pos = max_idxs[f]
                    dconv_relu[pos, f] = dpooled[f]

                # backprop through ReLU
                drelu_mask = (conv.conv_out > 0).astype(float)  # conv.conv_out stored in conv.forward
                dconv = dconv_relu * drelu_mask  # zero where conv_out <= 0

                # conv backward -> gradients for filters, biases, and gradient w.r.t input embeddings
                dfilters, dbiases, dXembed = conv.backward(dconv)
                dFilters_acc += dfilters
                dBias_acc += dbiases

                # accumulate gradients for embeddings (each word in x_idx)
                # dXembed shape: (seq_len, embed_dim)
                for pos_in_seq, word_index in enumerate(x_idx):
                    dEmb_acc[word_index] += dXembed[pos_in_seq]

            # After batch processed, update params (SGD)
            bs = len(batch_idx)
            # embeddings
            embeddings -= (lr / bs) * dEmb_acc
            # conv filters and biases
            conv.filters -= (lr / bs) * dFilters_acc
            conv.biases -= (lr / bs) * dBias_acc
            # dense
            fc.W -= (lr / bs) * dW_acc
            fc.b -= (lr / bs) * db_acc

        if epoch % print_every == 0 or epoch == 1:
            avg_loss = total_loss / N
            print(f"Epoch {epoch}/{epochs} — avg loss: {avg_loss:.4f}")
    # return trained components
    model = {
        "embeddings": embeddings,
        "conv": conv,
        "fc": fc,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "seq_len": seq_len,
        "embedding_dim": embedding_dim
    }
    return model

"""4) PREDICTION / AUTOCOMPLETE"""

def predict_next(model, prefix_words, top_k=5):
    """
    Prédit les k prochains mots les plus probables.

    Args:
        model (dict): Le modèle entraîné
        prefix_words (list): Liste de tokens (strings). Si plus court que seq_len, sera complété avec PAD.
        top_k (int, optional): Nombre de prédictions à retourner. Par défaut 5.

    Returns:
        list: Liste de tuples (mot, probabilité) pour les top_k prédictions
    """
    word2idx = model["word2idx"]
    idx2word = model["idx2word"]
    seq_len = model["seq_len"]
    embeddings = model["embeddings"]
    conv = model["conv"]
    fc = model["fc"]

    # build indices sequence (take last seq_len words)
    seq = [word2idx.get(w, word2idx["<UNK>"]) for w in prefix_words]
    # take last seq_len (if shorter pad left)
    if len(seq) < seq_len:
        seq = [word2idx["<PAD>"]] * (seq_len - len(seq)) + seq
    else:
        seq = seq[-seq_len:]
    X_embed = embeddings[np.array(seq)]  # (seq_len, embed_dim)

    conv_out = conv.forward(X_embed)
    conv_relu = np.maximum(0, conv_out)
    pooled = np.max(conv_relu, axis=0)
    logits = fc.forward(pooled)
    probs = softmax(logits)
    # get top_k
    idxs = np.argsort(probs)[-top_k:][::-1]
    return [(idx2word[i], float(probs[i])) for i in idxs]

# --------------EXAMPLE USAGE-------------


if __name__ == "__main__":
    # Corpus médical détaillé sur le cancer du sein
    corpus = [
        # Descriptions générales
        "breast cancer screening methods include mammography ultrasound and clinical examination",
        "regular mammographic screening reduces mortality rates in breast cancer patients",
        "early detection of breast cancer significantly improves survival rates",

        # Cas Malins (M)
        "malignant breast tumors show irregular shapes and unclear boundaries in mammogram",
        "malignant breast lesions often present with spiculated margins and architectural distortion",
        "malignant cases require immediate aggressive treatment including surgery chemotherapy",
        "malignant breast masses typically show increased vascularity and heterogeneous enhancement",
        "malignant tumors may present with microcalcifications and tissue distortion",
        "malignant breast cancer cells have high proliferation rates and abnormal growth patterns",

        # Cas Normaux (N)
        "normal breast tissue appears symmetrical and uniform in mammographic images",
        "normal mammogram shows regular patterns of fibroglandular and adipose tissue",
        "normal breast examination reveals no suspicious masses or abnormalities",
        "normal breast tissue maintains regular ducts and uniform density",
        "normal screening results indicate absence of suspicious calcifications",

        # Traitements
        "breast cancer treatment options include surgery radiation therapy and chemotherapy",
        "targeted therapy drugs specifically attack breast cancer cells while sparing normal tissue",
        "hormonal therapy effectively treats hormone receptor positive breast cancers",
        "neoadjuvant chemotherapy reduces tumor size before surgical intervention",
        "mastectomy removes entire breast tissue in advanced malignant cases",
        "breast conserving surgery removes tumor while preserving normal breast tissue",
        "radiation therapy targets remaining cancer cells after surgery",

        # Diagnostic et Pronostic
        "triple negative breast cancer requires specialized treatment approaches",
        "her2 positive breast cancers respond well to targeted biological therapies",
        "hormone receptor status determines appropriate treatment strategy",
        "tumor grade and stage influence treatment decisions and prognosis",
        "lymph node involvement indicates potential metastatic spread",

        # Surveillance et Suivi
        "regular follow up imaging monitors treatment response and detects recurrence",
        "annual mammography screening recommended for women over forty years",
        "breast mri provides detailed imaging for high risk patients",
        "genetic testing identifies hereditary breast cancer risk factors",

        # Effets Secondaires et Récupération
        "chemotherapy side effects include fatigue nausea and hair loss",
        "radiation therapy may cause skin changes and breast tenderness",
        "post surgery rehabilitation includes physical therapy and exercise",
        "psychological support essential during breast cancer treatment",

        # Prévention et Facteurs de Risque
        "lifestyle factors influence breast cancer risk and prevention",
        "obesity increases risk of postmenopausal breast cancer",
        "brca gene mutations significantly increase breast cancer risk",
        "regular physical activity reduces breast cancer risk",

        # Innovations et Recherche
        "artificial intelligence improves mammogram interpretation accuracy",
        "personalized medicine tailors breast cancer treatment to genetic profile",
        "immunotherapy shows promise in advanced breast cancer treatment",
        "new targeted therapies continuously improve treatment outcomes"
    ]

    # train (règle les hyperparams ici)
    model = train_cnn(
        corpus_sentences=corpus,
        seq_len=3,
        embedding_dim=50,
        kernel_size=2,
        num_filters=64,
        epochs=200,
        lr=0.5,
        print_every=20,
        batch_size=8
    )

    # test suggestions
    prefixes = [
        ["breast", "cancer", "is"],
        ["cancer", "treatment", "is"],
        ["early", "detection", "of"]
    ]
    for p in prefixes:
        print("\nPrefix:", " ".join(p))
        suggestions = predict_next(model, p, top_k=5)
        for w, prob in suggestions:
            print(f"  {w}  ({prob:.3f})")

class LSTM:
    def __init__(self, input_dim, hidden_dim):
        """
        Initialise une couche LSTM.

        Args:
            input_dim (int): Dimension d'entrée
            hidden_dim (int): Dimension de l'état caché
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # paramètres (poids pour input et hidden)
        self.Wf = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bf = np.zeros((hidden_dim, 1))
        self.Wi = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bi = np.zeros((hidden_dim, 1))
        self.Wc = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bc = np.zeros((hidden_dim, 1))
        self.Wo = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bo = np.zeros((hidden_dim, 1))

    def forward(self, X_seq):
        """
        X_seq: (seq_len, input_dim)
        """
        h_prev = np.zeros((self.hidden_dim, 1))
        c_prev = np.zeros((self.hidden_dim, 1))
        hs = []
        for t in range(X_seq.shape[0]):
            x_t = X_seq[t].reshape(-1, 1)
            concat = np.vstack((h_prev, x_t))
            f_t = self._sigmoid(self.Wf @ concat + self.bf)
            i_t = self._sigmoid(self.Wi @ concat + self.bi)
            c_hat = np.tanh(self.Wc @ concat + self.bc)
            c_t = f_t * c_prev + i_t * c_hat
            o_t = self._sigmoid(self.Wo @ concat + self.bo)
            h_t = o_t * np.tanh(c_t)
            hs.append(h_t)
            h_prev, c_prev = h_t, c_t
        return np.array(hs).squeeze(axis=2), h_t  # all hidden states, last hidden

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, dh_next, dc_next, cache):
        """
        Rétropropagation LSTM

        Args:
            dh_next: Gradient de la loss par rapport à h_t
            dc_next: Gradient de la loss par rapport à c_t
            cache: Cache des valeurs forward pour cette timestep
        """
        x_t, h_prev, c_prev, i_t, f_t, o_t, c_hat, c_t, concat = cache

        # Gradients des portes de sortie
        do = dh_next * np.tanh(c_t)
        do = do * o_t * (1 - o_t)  # dérivée sigmoid

        # Gradient de l'état cellulaire
        dc = dh_next * o_t * (1 - np.tanh(c_t)**2)
        dc = dc + dc_next

        # Gradient de c_hat
        dc_hat = dc * i_t
        dc_hat = dc_hat * (1 - c_hat**2)  # dérivée tanh

        # Gradient de la porte d'entrée
        di = dc * c_hat
        di = di * i_t * (1 - i_t)  # dérivée sigmoid

        # Gradient de la porte d'oubli
        df = dc * c_prev
        df = df * f_t * (1 - f_t)  # dérivée sigmoid

        # Gradients des poids
        dWf = df @ concat.T
        dWi = di @ concat.T
        dWc = dc_hat @ concat.T
        dWo = do @ concat.T

        # Gradients des biais
        dbf = np.sum(df, axis=1, keepdims=True)
        dbi = np.sum(di, axis=1, keepdims=True)
        dbc = np.sum(dc_hat, axis=1, keepdims=True)
        dbo = np.sum(do, axis=1, keepdims=True)

        # Gradient pour la prochaine itération
        dconcat = (self.Wf.T @ df + self.Wi.T @ di +
                  self.Wc.T @ dc_hat + self.Wo.T @ do)
        dh_prev = dconcat[:self.hidden_dim]
        dx = dconcat[self.hidden_dim:]
        dc_prev = f_t * dc

        return dx, dh_prev, dc_prev, (dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo)

def train_cnn_lstm(corpus_sentences, seq_len=3, embedding_dim=50,
                  kernel_size=2, num_filters=32, hidden_dim=64,
                  epochs=100, lr=0.1, print_every=10, validation_split=0.2):
    """
    Entraîne un modèle hybride CNN-LSTM pour la prédiction de texte.

    Args:
        corpus_sentences (list): Liste des phrases d'entraînement
        seq_len (int): Longueur des séquences d'entrée
        embedding_dim (int): Dimension des embeddings
        kernel_size (int): Taille du noyau de convolution
        num_filters (int): Nombre de filtres de convolution
        hidden_dim (int): Dimension de l'état caché LSTM
        epochs (int): Nombre d'époques d'entraînement
        lr (float): Taux d'apprentissage
        print_every (int): Fréquence d'affichage des métriques
        validation_split (float): Proportion des données pour la validation

    Returns:
        dict: Le modèle entraîné avec tous ses composants
    """
    word2idx, idx2word = build_vocab(corpus_sentences)
    vocab_size = len(word2idx)
    seqs = texts_to_sequences(corpus_sentences, word2idx)
    X, Y = create_examples(seqs, seq_len)

    # Split données en train/validation
    n_samples = len(X)
    n_val = int(n_samples * validation_split)
    indices = np.random.permutation(n_samples)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
    conv = Conv1D(seq_len, embedding_dim, kernel_size, num_filters)
    lstm = LSTM(num_filters, hidden_dim)
    fc = Dense(hidden_dim, vocab_size)

    # Historique pour le suivi
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct_preds = 0

        # Training
        for i in range(len(X_train)):
            x_idx = X[i]
            y_idx = int(Y[i])
            X_embed = embeddings[x_idx]

            # ---- CNN ----
            conv_out = conv.forward(X_embed)
            conv_relu = np.maximum(0, conv_out)

            # ---- LSTM ----
            hs, h_last = lstm.forward(conv_relu)

            # ---- Dense + Softmax ----
            logits = fc.forward(h_last.flatten())
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, y_idx)
            total_loss += loss

            # Rétropropagation
            dlogits = probs.copy()
            dlogits[y_idx] -= 1.0

            # Dense backward
            dW_fc, db_fc, dh_last = fc.backward(dlogits)

            # LSTM backward
            dh = np.zeros((conv_relu.shape[0], lstm.hidden_dim))
            dc = np.zeros((lstm.hidden_dim, 1))
            dh[-1] = dh_last.reshape(-1, 1)  # Gradient du dernier état caché

            # Stockage des gradients LSTM
            dWf, dWi, dWc, dWo = np.zeros_like(lstm.Wf), np.zeros_like(lstm.Wi), \
                                np.zeros_like(lstm.Wc), np.zeros_like(lstm.Wo)
            dbf, dbi, dbc, dbo = np.zeros_like(lstm.bf), np.zeros_like(lstm.bi), \
                                np.zeros_like(lstm.bc), np.zeros_like(lstm.bo)

            # Rétropropagation à travers le temps (BPTT)
            for t in reversed(range(conv_relu.shape[0])):
                dx_t, dh_prev, dc_prev, gradients = lstm.backward(dh[t].reshape(-1, 1), dc, cache[t])
                dWf += gradients[0]; dWi += gradients[1]
                dWc += gradients[2]; dWo += gradients[3]
                dbf += gradients[4]; dbi += gradients[5]
                dbc += gradients[6]; dbo += gradients[7]
                dc = dc_prev
                if t > 0:
                    dh[t-1] += dh_prev.flatten()

            # Mise à jour des poids avec SGD
            lr_t = lr / (epoch ** 0.5)  # Learning rate adaptatif

            # Mise à jour LSTM
            lstm.Wf -= lr_t * dWf
            lstm.Wi -= lr_t * dWi
            lstm.Wc -= lr_t * dWc
            lstm.Wo -= lr_t * dWo
            lstm.bf -= lr_t * dbf
            lstm.bi -= lr_t * dbi
            lstm.bc -= lr_t * dbc
            lstm.bo -= lr_t * dbo

            # Mise à jour Dense
            fc.W -= lr_t * dW_fc
            fc.b -= lr_t * db_fc

        # Calcul des métriques d'entraînement
        train_loss = total_loss / len(X_train)
        train_acc = correct_preds / len(X_train)

        # Validation
        val_loss = 0.0
        val_correct = 0
        for i in range(len(X_val)):
            x_idx = X_val[i]
            y_idx = int(Y_val[i])

            # Forward pass sans mise à jour des poids
            X_embed = embeddings[x_idx]
            conv_out = conv.forward(X_embed)
            conv_relu = np.maximum(0, conv_out)
            hs, h_last = lstm.forward(conv_relu)
            logits = fc.forward(h_last.flatten())
            probs = softmax(logits)

            # Métriques de validation
            val_loss += cross_entropy_loss(probs, y_idx)
            if np.argmax(probs) == y_idx:
                val_correct += 1

        val_loss /= len(X_val)
        val_acc = val_correct / len(X_val)

        # Sauvegarde de l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

        if epoch % print_every == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            print("-" * 50)

    model = {
        "embeddings": embeddings,
        "conv": conv,
        "lstm": lstm,
        "fc": fc,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "seq_len": seq_len
    }
    return model
