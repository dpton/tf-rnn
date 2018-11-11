import tensorflow as tf
import os
from utils.reprocessing import generator_enqueue
from utils.reprocessing import PAD_ID,GO_ID, EOS_ID, UNK_ID


class Variational_autoencoder_Seq2Seq(object):

    def __init__(self, layers_size=100, VAE_layers_size=100, learning_rate=0.01,
                 vocab_size=1000,
                 init_embedding=None, embedding_size=100,
                 lr_decay=0.9, scope='seq2seq', max_grad_norm=5, queue_capacity=1000,
                 max_length=50, min_length=3, num_samples=1024,
                 type_model='Train', tracking='tracking', dtype=tf.float32):
        self.layers_size = layers_size
        self.learning_rate = learning_rate
        self.VAE_layers_size = VAE_layers_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.lr_decay = lr_decay
        self.scope = scope
        self.embedding_size = embedding_size
        self.min_length = min_length
        self.max_grad_norm = max_grad_norm
        self.init_embedding = init_embedding
        self.tracking = tracking
        self.num_samples = num_samples
        self.dtype = dtype
        self.type_model = type_model
        self.graph = None

    def _cell(self, num_units, keep_prob=1.):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units)
        return cell

    def _clip_gradients(self, grads_and_vars, embedding_norm = 0.1):
        """In addition to standard gradient clipping, also clips embedding
        gradients to a specified value."""
        clipped_gradients = []
        variables = []
        for gradient, variable in grads_and_vars:
            if "embedding" in variable.name:
                tmp = tf.clip_by_norm(
                    gradient.values, embedding_norm)
                gradient = tf.IndexedSlices(tmp, gradient.indices, gradient.dense_shape)
            clipped_gradients.append(gradient)
            variables.append(variable)
        return list(zip(clipped_gradients, variables))

    def _optimizer(self, loss):
        """Create the optimizer node of the graph."""
        self.lr_var = tf.Variable(self.learning_rate, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                          self.max_grad_norm)
#         grads_and_vars = self._clip_gradients(list(zip(grads, tvars)))
        grads_and_vars = zip(grads, tvars)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr_var, epsilon=1e-08)
        _train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=self.global_step)
        return _train_op

    def _sample_posterior(self, x):
        x = tf.concat(x, axis=-1)
        latent_dim = self.VAE_layers_size
        with tf.variable_scope('variantional_autoencoder'):
            #             epsilon = tf.constant(1e-8)
            self.z_mu = tf.layers.dense(
                inputs=x, units=latent_dim, name='z_mu')
            self.z_var = tf.layers.dense(inputs=x, units=latent_dim, activation=tf.nn.softplus,
                                         bias_initializer=tf.constant_initializer(
                                             -2.5),
                                         name='z_log_sigma') + 1e-8

            epsilon = tf.random_normal(shape=tf.shape(self.z_mu))
            z = self.z_mu + tf.sqrt(self.z_var) * epsilon

        self.variable_summaries(z)
        concat = tf.layers.dense(
            inputs=z, units=2 * self.layers_size, name='project_state')
        state = tf.contrib.rnn.LSTMStateTuple(*tf.split(concat, 2, 1))

        kl_div = -0.5 * tf.reduce_sum(1.0 + tf.log(self.z_var) - tf.square(self.z_mu) - self.z_var,
                                      axis=1)
        self.kl_div = tf.reduce_mean(kl_div)

        tf.summary.scalar('KL_divergence_loss', self.kl_div)

        return state

    def _encoder(self, inputs):
        encoder_cell = self._cell(self.layers_size)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, inputs, dtype=self.dtype)

        return encoder_outputs, encoder_state

    def _decoder(self, inputs, state, is_train):
        decoder_cell = self._cell(self.layers_size, self.keep_prob)
        with tf.variable_scope('decoder'):
            self.sequence_length = tf.cast(
                tf.reduce_sum(self.weight, axis=1), tf.int32)

            train_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs, self.sequence_length)
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.embedding,
                tf.fill(
                    [self.batch_size], GO_ID),
                tf.constant(EOS_ID))

            project_layer = layers_core.Dense(self.embedding.get_shape()[
                                              0], name='output_project')
            def _create_decoder(helper):
                return tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=helper,
                    initial_state=state,
                    output_layer=project_layer)  # initial state of decoder
            train_decoder = _create_decoder(train_helper)
            inference_decoder = _create_decoder(inference_helper)

        return tf.cond(is_train,
                       lambda: tf.contrib.seq2seq.dynamic_decode(
                           train_decoder, maximum_iterations=self.max_length),
                       lambda: tf.contrib.seq2seq.dynamic_decode(inference_decoder, maximum_iterations=self.max_length))

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def annealing_schedule(self, t, pivot):
        return tf.nn.sigmoid((t - pivot) / pivot * 10)

    def build_model(self, data_inputs, validate_inputs=None):
        with tf.Graph().as_default() as self.graph:
            # Placeholder
            self.global_step = tf.Variable(0, trainable=False)
            self.batch_size = tf.placeholder(
                tf.int32, shape=(), name='batch_size')
            self.annealing_pivot = tf.placeholder(
                tf.float32, shape=(), name='annealing_pivot')
            self.keep_prob = tf.placeholder(
                tf.float32, shape=(), name='keep_prob')
            self.is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
            self.word_keeping = tf.placeholder(
                dtype=tf.float32, shape=(), name='word_keeping')
            self.l2_epsilon = tf.placeholder(
                dtype=tf.float32, shape=(), name='l2_epsilon')

            self.scale_kl_div = tf.placeholder(
                dtype=tf.float32, shape=(), name='scale_kl_div')

            self.coord = tf.train.Coordinator()
            self.generator_enqueue = generator_enqueue(
                self.coord, data_inputs, validate_inputs)
            encode_queue, decode_queue, weight_queue = self.generator_enqueue.get_queue(
                self.batch_size)

            self.encode_input = tf.placeholder_with_default(
                encode_queue, shape=[None, None], name='encode_input')
            self.decode_input = tf.placeholder_with_default(
                decode_queue, shape=[None, None], name='decode_input')
            self.weight = tf.placeholder_with_default(
                weight_queue, shape=[None, None], name='weigth')

            ###EMBEDDING###
            if self.init_embedding is not None:
                self.embedding = tf.get_variable("embedding",
                                                 initializer=self.init_embedding,
                                                 trainable=True)
            else:
                self.embedding = tf.get_variable("embedding",
                                                 [self.vocab_size, self.embedding_size])

            emb_encode = tf.nn.embedding_lookup(
                self.embedding, self.encode_input)

            # Word dropout

            random_tensor = self.word_keeping + \
                tf.random_uniform(shape=tf.shape(self.decode_input))
            binary_tensor = tf.floor(random_tensor)
            dropped_decode = tf.where(tf.greater(binary_tensor, 0),
                                      self.decode_input, tf.fill(tf.shape(self.decode_input), UNK_ID))
            emb_decode = tf.nn.embedding_lookup(self.embedding, dropped_decode)
            ###

            ###ENCODER###
            encoder_outputs, encoder_state = self._encoder(emb_encode)
            ###

            ###SAMPLING###
            sample = self._sample_posterior(encoder_state)
            ###

            ###DECODER###
            self.outputs, final_state, final_sequence_length = self._decoder(
                emb_decode, sample, self.is_train)
            ###

            # BUILD LOSS
            _logits = self.outputs.rnn_output

            tf.summary.histogram('output', self.outputs.sample_id)
            paddings = [[0, 0], [0, self.max_length -
                                 tf.shape(_logits)[1]], [0, 0]]
            logits = tf.pad(_logits, paddings, "CONSTANT")

            _target = self.decode_input[:, 1:]  # remove the start_token
            paddings = [[0, 0], [0, self.max_length - tf.shape(_target)[1]]]
            target = tf.pad(_target, paddings, "CONSTANT")

            paddings = [
                [0, 0], [0, self.max_length - tf.shape(self.weight)[1]]]
            padded_weight = tf.pad(self.weight, paddings, "CONSTANT")

            self.reconstruct_loss = tf.contrib.seq2seq.sequence_loss(
                logits, target, padded_weight)
            
            tf.summary.scalar('reconstruct_loss', self.reconstruct_loss)

            annealing_weight = self.annealing_schedule(
                tf.cast(self.global_step, tf.float32), self.annealing_pivot)
            # TOTAL LOSS
            self.total_loss = self.reconstruct_loss + annealing_weight * self.kl_div
            tf.summary.scalar('total_loss', self.total_loss)

            self.op = self._optimizer(self.total_loss)

            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.tracking)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        return self

    def save(self, sess, model_dir=""):
        self.saver.save(sess, os.path.join(
            model_dir, 'dynamic_attention_decoder'), global_step=self.global_step)

    def load(self, sess, model_dir=""):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('success')
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('fail')

    def validate(self, sess, inputs, batch_size=1, feed_dict={}):
        losses = 0
        for i in range(len(inputs) // batch_size):
            def pad(x):
                max_len = len(max(x, key=len))
                x = map(lambda y: (max_len - len(x)) * [0], x)
                return x
            batch_inputs = inputs[i * batch_size: (i + 1) * batch_size]
            weight_input = map(lambda x: [1.] * len(x), batch_inputs)
            decode_input = map(lambda x: [GO_ID] + x + [EOS_ID], batch_inputs)
            feed_dict[self.encode_input] = pad(batch_inputs)
            feed_dict[self.decode_input] = pad(decode_input)
            feed_dict[self.weight] = pad(weight_input)

            loss, = sess.run([self.total_loss], feed_dict=feed_dict)
            losses += loss
        return losses

    def train(self, sess):
        thread = threading.Thread(target=self.generator_enqueue.run, args=(
            sess, self.max_length, self.min_length))
        thread.daemon = True
        thread.start()
        self.coord.register_thread(thread)

    def stop_train(self, sess):
        self.coord.request_stop()
