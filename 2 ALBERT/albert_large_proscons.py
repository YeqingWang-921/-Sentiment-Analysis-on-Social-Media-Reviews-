

import json
import numpy as np
import codecs
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizer import SpTokenizer
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, get_all_attributes

locals().update(get_all_attributes(keras.layers))  
set_gelu('tanh') 


maxlen = 32
config_path = 'models/albert_large/albert_config.json'
checkpoint_path = 'models/albert_large/variables/variables'
spm_path = 'models/albert_large/assets/30k-clean.model'


def load_data(filename):
    D = []
    with codecs.open(filename, encoding='utf-8') as f:
        for l in f:
            l=l.strip()
            lg=len(l)
            text=l[:lg-1].strip()
            label=l[-1]
            #print(text)
            #print(label)
            D.append((text, int(label)))
    return D



train_data = load_data('datasets/pros_cons_trainshuffle.data')
valid_data = load_data('datasets/pros_cons_valshuffle.data')
test_data = load_data('datasets/pros_cons_testshuffle.data')


tokenizer = SpTokenizer(spm_path)


class data_generator:
   
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            text, label = self.data[i]
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d



bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    albert=True,
    return_keras_model=False,
)

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(units=2,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()
AdamLR = extend_with_piecewise_linear_lr(Adam)

model.compile(
    loss='sparse_categorical_crossentropy',
   
    optimizer=AdamLR(learning_rate=3e-5,
                     lr_schedule={1000: 1, 2000: 0.1}),
    metrics=['accuracy'],
)


train_generator = data_generator(train_data)
valid_generator = data_generator(valid_data)
test_generator = data_generator(test_data)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(u'val_acc: %05f, best_val_acc: %05f, test_acc: %05f\n' %
              (val_acc, self.best_val_acc, test_acc))


evaluator = Evaluator()
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=50,
                    callbacks=[evaluator])

model.load_weights('best_model.weights')
print(u'final test acc: %05f\n' % (evaluate(test_generator)))
