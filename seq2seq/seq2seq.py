import mxnet as mx
import datetime
import collections

from mxnet import autograd, gluon, nd
from mxnet.gluon import nn, rnn, Block
from mxnet.contrib import text

# PAD用来补长，使得每句一样长
PAD = '<pad>'

# 开始字符
BOS = '<bos>'

# 结束字符
EOS = '<eos>'

epochs = 50
epoch_period = 10
learning_rate = 0.005
max_seq_len = 10

encoder_num_layers = 1
decoder_num_layers = 2

encoder_drop_prod = 0.1
decoder_drop_prod = 0.1

encoder_hidden_dim = 256
decoder_hidden_dim = 256
alignment_dim = 25

ctx = mx.cpu(0)


def read_data(max_seq_len):
    input_tokens = []
    output_tokens = []
    input_seqs = []
    output_seqs = []

    with open('../fra-eng/fra.txt') as f:
        lines = f.readlines()
        for ii, line in enumerate(lines):
            if ii > 100:
                break
            input_seq, output_seq, _ = line.rstrip().split('\t')
            cur_input_tokens = input_seq.split(' ')
            cur_output_tokens = output_seq.split(' ')

            if len(cur_input_tokens) < max_seq_len and len(cur_output_tokens) < max_seq_len:

                input_tokens.extend(cur_input_tokens)
                cur_input_tokens.append(EOS)
                while len(cur_input_tokens) < max_seq_len:
                    cur_input_tokens.append(PAD)
                input_seqs.append(cur_input_tokens)

                output_tokens.extend(cur_output_tokens)
                cur_output_tokens.append(EOS)
                while len(cur_output_tokens) < max_seq_len:
                    cur_output_tokens.append(EOS)
                output_seqs.append(cur_output_tokens)

        en_vocab = text.vocab.Vocabulary(collections.Counter(input_tokens),
                                         reserved_tokens=[PAD, BOS, EOS])
        fr_vocab = text.vocab.Vocabulary(collections.Counter(output_tokens),
                                         reserved_tokens=[PAD, BOS, EOS])
    return en_vocab, fr_vocab, input_seqs, output_seqs


input_vocab, output_vocab, input_seqs, output_seqs = read_data(max_seq_len)
X = nd.zeros((len(input_seqs), max_seq_len), ctx=ctx)
Y = nd.zeros((len(output_seqs), max_seq_len), ctx=ctx)
for i in range(len(input_seqs)):
    X[i] = nd.array(input_vocab.to_indices(input_seqs[i]), ctx=ctx)
    Y[i] = nd.array(output_vocab.to_indices(input_seqs[i]), ctx=ctx)
dataset = gluon.data.ArrayDataset(X, Y)


class Encoder(Block):
    def __init__(self, input_dim, hidden_dim, num_layers, drop_prob):
        super(Encoder, self).__init__()
        with self.name_scope():
            self.embedding = nn.Embedding(input_dim, hidden_dim)
            self.dropout = nn.Dropout(drop_prob)
            self.rnn = rnn.GRU(hidden_dim, num_layers, dropout=drop_prob, input_size=hidden_dim)

    def forward(self, inputs, state):
        emb = self.embedding(inputs).swapaxes(0, 1)
        emb = self.dropout(emb)
        output, state = self.rnn(emb, state)
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


class Decoder(Block):
    def __init__(self, hidden_dim, output_dim, num_layers, max_seq_len, drop_prob,
                 alignment_dim, encoder_hidden_dim):
        super(Decoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_size = hidden_dim
        self.num_layers = num_layers

        with self.name_scope():
            self.embedding = nn.Embedding(output_dim, hidden_dim)
            self.dropout = nn.Dropout(drop_prob)

            # 注意力机制
            self.attention = nn.Sequential()
            with self.attention.name_scope():
                self.attention.add(
                    nn.Dense(units=alignment_dim,
                             in_units=hidden_dim + encoder_hidden_dim,
                             activation='tanh',
                             flatten=False))
                self.attention.add(nn.Dense(1, in_units=alignment_dim, flatten=False))

            self.rnn = rnn.GRU(hidden_dim, num_layers, dropout=drop_prob, input_size=hidden_dim)
            self.out = nn.Dense(output_dim, in_units=hidden_dim)
            self.rnn_concat_input = nn.Dense(hidden_dim, in_units=hidden_dim + encoder_hidden_dim, flatten=False)

    def forward(self, cur_input, state, encoder_outputs):
        # 当rnn为多层时，取最靠近输出层的单层隐含状态
        single_layer_state = [state[0][-1].expand_dims(0)]
        encoder_outputs = encoder_outputs.reshape((self.max_seq_len, 1, self.encoder_hidden_dim))

        # single_layer_state尺寸：[(1, 1, decoder_hidden_dim)]
        # hidden_broadcast尺寸： (max_seq_len, 1, decoder_hidden_dim)
        hidden_broadcast = nd.broadcast_axis(single_layer_state[0], axis=0, size=self.max_seq_len)

        # (max_seq_len, 1, encoder_hidden_dim + decoder_hidden_dim)
        # 将decoder当前时刻的s与encoder所有的h配对组成[s(t-1), h1],[s(t-1), h2]...[s(t-1), h10]
        encoder_outputs_and_hiddens = nd.concat(encoder_outputs, hidden_broadcast, dim=2)

        # (max_seq_len, 1, 1)
        energy = self.attention(encoder_outputs_and_hiddens)

        batch_attention = nd.softmax(energy, axis=0).reshape((1, 1, self.max_seq_len))

        # (1, max_seq_len, encoder_hidden_dim)
        batch_encoder_outputs = encoder_outputs.swapaxes(0, 1)

        # (1, 1, encoder_hidden_dim)
        decoder_context = nd.batch_dot(batch_attention, batch_encoder_outputs)

        # (1, 1, encoder_hidden_dim + decoder_hidden_dim)
        input_and_context = nd.concat(self.embedding(cur_input).reshape(1, 1, self.hidden_size), decoder_context, dim=2)
        # (1, 1, decoder_hidden_dim)
        concat_input = self.rnn_concat_input(input_and_context)
        concat_input = self.dropout(concat_input)

        # 当RNN为多层时，用单层隐含状态初始化各个层的隐含状态
        state = [nd.broadcast_axis(single_layer_state[0], axis=0, size=self.num_layers)]

        output, state = self.rnn(concat_input, state)
        output = self.dropout(output)
        output = self.out(output)

        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


class DecoderInitState(Block):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(DecoderInitState, self).__init__()
        with self.name_scope():
            self.dense = nn.Dense(decoder_hidden_dim, in_units=encoder_hidden_dim, activation='tanh', flatten=False)

    def forward(self, encoder_state):
        return [self.dense(encoder_state)]


def train(encoder, decoder, decoder_init_state, max_seq_len, ctx, eval_en_fr):
    encoder.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    decoder.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    decoder_init_state.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    encoder_optimizer = gluon.Trainer(encoder.collect_params(), 'adam',
                                      {'learning_rate': learning_rate})
    decoder_optimizer = gluon.Trainer(decoder.collect_params(), 'adam',
                                      {'learning_rate': learning_rate})
    decoder_init_state_optimizer = gluon.Trainer(
        decoder_init_state.collect_params(), 'adam',
        {'learning_rate': learning_rate}
    )

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    prev_time = datetime.datetime.now()
    data_iter = gluon.data.DataLoader(dataset, 1, shuffle=True)

    total_loss = 0.0
    for epoch in range(1, epochs + 1):
        for x, y in data_iter:
            with autograd.record():
                loss = nd.array([0], ctx=ctx)
                encoder_state = encoder.begin_state(
                    func=mx.nd.zeros, batch_size=1, ctx=ctx
                )
                encoder_outputs, encoder_state = encoder(x, encoder_state)

                # (max_seq_len, encoder_hidden_dim)
                encoder_outputs = encoder_outputs.flatten()
                decoder_input = nd.array([output_vocab.token_to_idx[BOS]], ctx=ctx)
                decoder_state = decoder_init_state(encoder_state[0])

                for i in range(max_seq_len):
                    decoder_output, decoder_state = decoder(decoder_input, decoder_state, encoder_outputs)
                    decoder_input = nd.array([decoder_output.argmax(axis=1).asscalar()], ctx=ctx)
                    loss = loss + softmax_cross_entropy(decoder_output, y[0][i])
                    if y[0][i].asscalar() == output_vocab.token_to_idx[EOS]:
                        break

            loss.backward()
            encoder_optimizer.step(1)
            decoder_optimizer.step(1)
            decoder_init_state_optimizer.step(1)
            total_loss += loss.asscalar() / max_seq_len

        if epoch % epoch_period == 0 or epoch == 1:
            cur_time = datetime.datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = 'Time %02d:%02d:%02d' % (h, m, s)
            print_loss_avg = total_loss / epoch_period / len(data_iter)
            loss_str = 'Epoch %d, Loss %f, ' % (epoch, print_loss_avg)
            print(loss_str + time_str)
            total_loss = 0.0
            prev_time = cur_time

            translate(encoder, decoder, decoder_init_state, eval_en_fr, ctx, max_seq_len)


def translate(encoder, decoder, decoder_init_state, en_frs, ctx, max_seq_len):
    for en_fr in en_frs:
        print('Input:', en_fr[0])
        input_tokens = en_fr[0].split(' ') + [EOS]

        while len(input_tokens) < max_seq_len:
            input_tokens.append(PAD)
        inputs = nd.array(input_vocab.to_indices(input_tokens), ctx=ctx)
        encoder_state = encoder.begin_state(func=mx.nd.zeros, batch_size=1, ctx=ctx)
        encoder_outputs, encoder_state = encoder(inputs.expand_dims(0), encoder_state)
        encoder_outputs = encoder_outputs.flatten()

        decoder_input = nd.array([output_vocab.token_to_idx[BOS]], ctx=ctx)
        decoder_state = decoder_init_state(encoder_state[0])
        output_tokens = []

        for i in range(max_seq_len):
            decoder_output, decoder_state = decoder(
                decoder_input, decoder_state, encoder_outputs
            )
            pred_i = int(decoder_output.argmax(axis=1).asnumpy())
            if pred_i == output_vocab.token_to_idx[EOS]:
                break
            else:
                output_tokens.append(output_vocab.idx_to_token[pred_i])
            decoder_input = nd.array([pred_i], ctx=ctx)

        print('output:', ' '.join(output_tokens))
        print('expect:', en_fr[1], '\n')


encoder = Encoder(len(input_vocab), encoder_hidden_dim, encoder_num_layers, encoder_drop_prod)
decoder = Decoder(decoder_hidden_dim, len(output_vocab), decoder_num_layers, max_seq_len,
                  decoder_drop_prod, alignment_dim, encoder_hidden_dim)
decoder_init_state = DecoderInitState(encoder_hidden_dim, decoder_hidden_dim)

eval_en_frs = [['she is japanese.', 'elle est japonaise']]
train(encoder, decoder, decoder_init_state, max_seq_len, ctx, eval_en_frs)
