from keras.models import Sequential
from keras import layers
from keras.layers import Dense
import numpy as np
from six.moves import range

TRAINING_SIZE = 100000
DIGITS = 3
REVERSE = False
MAXLEN = DIGITS + 1 + DIGITS
chars = '0123456789+- '
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 2

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)

# Construct charactor table
ctable = CharacterTable(chars)

def generateData(opTypes):
    questions = []
    expected = []
    seen = set()
    while len(questions) < TRAINING_SIZE:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
        a, b = f(), f()
        if len(opTypes) > 1:
            opType = np.random.choice(list(opTypes))
        else:
            opType = opTypes
        key = tuple(sorted((a, b))) + tuple(opTypes)
        if key in seen:
            continue
        seen.add(key)
        q = '{}{}{}'.format(a,opType,b)
        query = q + ' ' * (MAXLEN - len(q))
        if opType == '+':
            ans = str(a + b)
        else:
            ans = str(a - b)
        ans += ' ' * (DIGITS + 1 - len(ans))
        if REVERSE:
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
    print('Sample Data: ', questions[:5], expected[:5])
    return questions, expected

def vectorization(questions, expected):
    x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(expected), DIGITS + 1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, DIGITS + 1)
    return x, y

def train_test_split(x, y, train_size):
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # train_test_split
    train_x = x[:train_size]
    train_y = y[:train_size]
    test_x = x[train_size:]
    test_y = y[train_size:]

    split_at = len(train_x) - len(train_x) // 10
    (x_train, x_val) = train_x[:split_at], train_x[split_at:]
    (y_train, y_val) = train_y[:split_at], train_y[split_at:]
    
    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)

    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)

    print('Testing Data:')
    print(test_x.shape)
    print(test_y.shape)
    
    return x_train, y_train, x_val, y_val, test_x, test_y

def buildModel():
    print('Build model...')
    model = Sequential()
    model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
    model.add(layers.RepeatVector(DIGITS + 1))
    for i in range(LAYERS):
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(len(chars))))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def trainModel(model, x_train, y_train, x_val, y_val, test_x, test_y):
    print('Model training...')
    acc_record = []
    for iteration in range(100):
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=1,
                  validation_data=(x_val, y_val),
                  verbose=0)
        acc = evaluate(model, test_x, test_y)
        acc_record.append(acc)
        if acc > 0.95:
            break
    return model, acc_record

def evaluate(model, test_x, test_y):
    num_corrects = 0
    prediction = model.predict_classes(test_x)
    for i in range(len(prediction)):
        ans = ctable.decode(test_y[i])
        guess = ctable.decode(prediction[i], calc_argmax=False)
        if ans == guess:
            num_corrects += 1
    return float(num_corrects / len(prediction))

def showsTestingResult(model, test_x, test_y):
    for i in range(10):
        ind = np.random.randint(0, len(test_x))
        rowx, rowy = test_x[np.array([ind])], test_y[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)

def GenerateData(opType, trainSize, digits=3):
    DIGITS = digits
    print('Digits:', DIGITS)
    questions, expected = generateData(opType)
    x, y = vectorization(questions, expected)
    x_train, y_train, x_val, y_val, test_x, test_y = train_test_split(x, y, trainSize)
    return x_train, y_train, x_val, y_val, test_x, test_y

def BuildAndTrainModel(x_train, y_train, x_val, y_val, test_x, test_y):
    model = buildModel()
    model, acc_record = trainModel(model, x_train, y_train, x_val, y_val, test_x, test_y)
    return model, acc_record
