import tensorflow as tf
from utils import create_queue

def linear(args, output_size, bias, bias_start=0.0):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable(
                "weights", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = tf.get_variable(
                    "biases", [output_size],
                    dtype=dtype,
                    initializer=tf.constant_initializer(bias_start, dtype=dtype))
        return tf.nn.bias_add(res, biases)
def _extract_beam_search(embedding, beam_size, num_symbols, embedding_size,  output_projection=None,
                                                            update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.
    Args:
        embedding: embedding tensor for symbols.
        output_projection: None or a pair (W, B). If provided, each fed previous
            output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.
    Returns:
        A loop function.
    """
    def loop_function(prev, i, log_beam_probs, beam_path, beam_symbols):
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(
                    prev, output_projection[0], output_projection[1])
        probs = tf.log(tf.nn.softmax(prev))
        global temp
        if i > 1:
            probs = tf.reshape(probs + log_beam_probs[-1], [-1, beam_size * num_symbols])
        best_probs, indices = tf.nn.top_k(probs, beam_size)
        indices = tf.stop_gradient(tf.reshape(indices, [-1]))
        best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))
        
        symbols = indices % num_symbols # Which word in vocabulary.
        beam_parent = indices // num_symbols # Which hypothesis it came from.


        beam_symbols.append(symbols)
        beam_path.append(beam_parent)
        log_beam_probs.append(best_probs)

        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, symbols)
        emb_prev = tf.reshape(emb_prev,[beam_size,embedding_size])
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev
    return loop_function
def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.
    Args:
        embedding: embedding tensor for symbols.
        output_projection: None or a pair (W, B). If provided, each fed previous
            output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.
    Returns:
        A loop function.
    """
    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(
                    prev, output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.nn.tf.stop_gradient(emb_prev)
        return emb_prev
    return loop_function
def attention_decoder(decoder_inputs,
                    initial_state,
                    attention_states,
                    cell,
                    output_size=None,
                    num_heads=1,
                    loop_function=None,
                    dtype=None,
                    scope=None,
                    initial_state_attention=False,
                    beam_size = 1,
                    is_beam_search = False):
    """RNN decoder with attention for the sequence-to-sequence model.
    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1412.7449 (see below for
    details). It is recommended for complex sequence-to-sequence tasks.
    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        output_size: Size of the output vectors; if None, we use cell.output_size.
        num_heads: Number of attention heads that read from attention_states.
        loop_function: If not None, this function will be applied to i-th output
            in order to generate i+1-th input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol). This can be used for decoding,
            but also for training to emulate http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states -- useful when we wish to resume decoding from  a previously
            stored decoder state and attention states.
    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors of
                shape [batch_size x output_size]. These represent the generated outputs.
                Output i is computed from input i (which is either the i-th element
                of decoder_inputs or loop_function(output {i-1}, i)) as follows.
                First, we run the cell on a combination of the input and previous
                attention masks:
                    cell_output, new_state = cell(linear(input, prev_attn), prev_state).
                Then, we calculate new attention masks:
                    new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
                and then we calculate the output:
                    output = linear(cell_output, new_attn).
            state: The state of each decoder cell the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
    Raises:
        ValueError: when num_heads is not positive, there are no inputs, shapes
            of attention_states are not set, or input size cannot be inferred
            from the input.
    """
    if decoder_inputs is None:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s"
                                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with tf.variable_scope(
            scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = tf.shape(decoder_inputs)[0]   # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = tf.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = tf.reshape(
                attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size    # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = tf.get_variable("AttnW_%d" % a,
                                            [1, 1, attn_size, attention_vec_size])
            hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                    tf.get_variable("AttnV_%d" % a, [attention_vec_size]))
        state = initial_state
        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []    # Results of attention reads will be stored here.
            attn_vec = []
            if isinstance(query, list):    # If the query is a tuple, flatten it.
                query_list = [q for q in query]
                for q in query_list:    # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                
                query = tf.concat(query_list,1 )
            for a in xrange(num_heads):
                with tf.variable_scope("Attention_%d" % a):
                    y = tf.reshape(query, [-1, 1, 1, attention_vec_size])
                    
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(
                            v[a] * tf.tanh(hidden_features[a] + y), [2, 3])
                    a = tf.nn.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(
                            tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                            [1, 2])
                    ds.append(tf.reshape(d, [-1, attn_size]))
                    attn_vec.append(a)
            return ds, attn_vec

        outputs = []
        attn_vecs = []
        prev = None
        log_beam_probs, beam_path, beam_symbols = [], [], []
        batch_attn_size = tf.stack([batch_size, attn_size])
        attns = [tf.zeros(batch_attn_size, dtype=dtype)
                         for _ in xrange(num_heads)]
        for a in attns:    # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
#         if initial_state_attention:
#             attns, attn_vec = attention(initial_state)
        if loop_function is not None and is_beam_search:
            tf.assert_equal(batch_size, 1)
        for i, inp in enumerate(tf.unstack(decoder_inputs, axis = 1)):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    if is_beam_search: 
                        inp = loop_function(prev, i, log_beam_probs, beam_path, beam_symbols)
                    else:
                        inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            # Run the RNN.
            cell_output, state = cell(inp, state)          
            if i == 0 and initial_state_attention:
                with tf.variable_scope(tf.get_variable_scope(),
                                       reuse=True):
                    attns, attn_vec = attention(attns)
            else:
                attns, attn_vec = attention(attns)
            with tf.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, cell.output_size, True)
                with tf.variable_scope("Projection"):
                    output = linear(output, output_size, True)
            if i == 0 and loop_function is not None and is_beam_search:
                states =[]
#                 for kk in range(beam_size):
#                     states.append(state)
#                 state = tf.reshape(tf.concat(0, states), [-1, state_size])
                for s in state: 
                    if isinstance(cell.state_size, tuple):
                        s = tf.tile(s, [1, beam_size, 1])
                        s = tf.contrib.rnn.LSTMStateTuple(*tf.unpack(s))
                    else:
                        s = tf.tile(s, [beam_size, 1])
                    states.append(s)
                state = states
            if loop_function is not None:
                prev = output
            outputs.append(output)
            attn_vecs.append(attn_vec)
    if is_beam_search and loop_function is not None:
        return outputs, state, attn_vecs, tf.reshape(tf.concat(beam_path, 0),[beam_size, -1]), tf.reshape(tf.concat(beam_symbols,0),[beam_size,-1])
    else:
        return outputs, state, attn_vecs


def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False,
                                initial_embedding = None,
                                is_beam_search = False,
                                beam_size = 1):
    """RNN decoder with embedding and attention and a pure-decoding option.
    Args:
        decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: rnn_cell.RNNCell defining the cell function.
        num_symbols: Integer, how many symbols come into the embedding.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_size: Size of the output vectors; if None, use output_size.
        output_projection: None or a pair (W, B) of output projection weights and
            biases; W has shape [output_size x num_symbols] and B has shape
            [num_symbols]; if provided and feed_previous=True, each fed previous
            output will first be multiplied by W and added B.
        feed_previous: Boolean; if True, only the first of decoder_inputs will be
            used (the "GO" symbol), and all other decoder inputs will be generated by:
                next = embedding_lookup(embedding, argmax(previous_output)),
            In effect, this implements a greedy decoder. It can also be used
            during training to emulate http://arxiv.org/abs/1506.03099.
            If False, decoder_inputs are used as given (the standard decoder case).
        update_embedding_for_previous: Boolean; if False and feed_previous=True,
            only the embedding for the first symbol of decoder_inputs (the "GO"
            symbol) will be updated by back propagation. Embeddings for the symbols
            generated from the decoder itself remain unchanged. This parameter has
            no effect if feed_previous=False.
        dtype: The dtype to use for the RNN initial states (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
            "embedding_attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states -- useful when we wish to resume decoding from a previously
            stored decoder state and attention states.
    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
            state: The state of each decoder cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
    Raises:
        ValueError: When output_projection has the wrong shape.
    """
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = tf.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])
        proj_weights = tf.convert_to_tensor(output_projection[0], dtype=dtype)
        proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])

    with tf.variable_scope(
            scope or "embedding_attention_decoder", dtype=dtype) as scope:
        if initial_embedding is not None:
            embedding = tf.get_variable("embedding",
                                        initializer = initial_embedding,
                                        trainable=False)
        else:
            embedding = tf.get_variable("embedding",
                                        [num_symbols, embedding_size])
        loop_function = None
        if feed_previous:
            if not is_beam_search:
                loop_function = _extract_argmax_and_embed(
                    embedding, output_projection,
                    update_embedding_for_previous)
            else:
                loop_function = _extract_beam_search(
                    embedding, beam_size, num_symbols,embedding_size, output_projection,
                    update_embedding_for_previous)
        emb_inp = tf.nn.embedding_lookup(embedding, decoder_inputs)
        return attention_decoder(
                emb_inp,
                initial_state,
                attention_states,
                cell,
                output_size=output_size,
                num_heads=num_heads,
                loop_function=loop_function,
                initial_state_attention=initial_state_attention,
                is_beam_search = is_beam_search,
                beam_size = beam_size)


def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                encode_embedding_size,
                                decode_embedding_size,
#                                 encode_sequence_length=None,
#                                 decode_sequence_length=None,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False,
                                initial_embedding_encode = None,
                                initial_embedding_decode = None,
                                is_beam_search = False,
                                beam_size = 1):
    """Embedding sequence-to-sequence model with attention.
    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. It keeps the outputs of this
    RNN at every step to use for attention later. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs attention decoder, initialized with the last
    encoder state, on embedded decoder_inputs and attending to encoder outputs.
    Warning: when output_projection is None, the size of the attention vectors
    and variables will be made proportional to num_decoder_symbols, can be large.
    Args:
        encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        num_encoder_symbols: Integer; number of symbols on the encoder side.
        num_decoder_symbols: Integer; number of symbols on the decoder side.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_projection: None or a pair (W, B) of output projection weights and
            biases; W has shape [output_size x num_decoder_symbols] and B has
            shape [num_decoder_symbols]; if provided and feed_previous=True, each
            fed previous output will first be multiplied by W and added B.
        feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
            of decoder_inputs will be used (the "GO" symbol), and all other decoder
            inputs will be taken from previous outputs (as in embedding_rnn_decoder).
            If False, decoder_inputs are used as given (the standard decoder case).
        dtype: The dtype of the initial RNN state (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
            "embedding_attention_seq2seq".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states.
    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x num_decoder_symbols] containing the generated
                outputs.
            state: The state of each decoder cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with tf.variable_scope(
            scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
        # Encoder.
        if initial_embedding_encode is not None:
            embedding = tf.get_variable("embedding_encoder",
                                       initializer = initial_embedding_encode,
                                       trainable=False)
        else:
            embedding = tf.get_variable("embedding_encoder",
                                        [num_encoder_symbols, encode_embedding_size])

        emb_encoder = tf.nn.embedding_lookup(embedding, encoder_inputs)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell, emb_encoder,dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        attention_states =  encoder_outputs
        # Decoder.
        output_size = None
        if output_projection is None:
            output_size = num_decoder_symbols

        if isinstance(feed_previous, bool):
            return embedding_attention_decoder(
                decoder_inputs,
                encoder_state,
                attention_states,
                cell,
                num_decoder_symbols,
                decode_embedding_size,
                num_heads=num_heads,
                output_size=output_size,
                output_projection=output_projection,
                feed_previous=feed_previous,
                initial_state_attention=initial_state_attention,
                initial_embedding=initial_embedding_decode,
                is_beam_search = is_beam_search,
                beam_size = beam_size)
import numpy as np
import pickle
import random
import time
class Seq2Seq_model(object):
    def __init__(self, layers_size = [100], learning_rate = 0.01, 
                 vocab_en_size = 1000, vocab_de_size = 1000,
                 batch_size = 32, dropout = 1.0, embedding_en_size = 100, embedding_de_size = 100,
                 init_encode = None, init_decode = None,
                 lr_decay = 0.9, scope = 'seq2seq_a', max_grad_norm = 2,capacity = 1000,
                 encode_sequence_length = 25, decode_sequence_length = 25, num_sampled = 2048,
                 checkpoint_dir = 'model/', config_path = None, dtype = tf.float32):
        if config_path is None:
            self.layers_size = layers_size
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.vocab_en_size = vocab_en_size
            self.vocab_de_size = vocab_de_size
            self.lr_decay = lr_decay
            self.dropout = dropout
            self.scope = scope
            self.embedding_en_size = embedding_en_size
            self.embedding_de_size = embedding_de_size
            self.max_grad_norm = max_grad_norm
            self.capacity = capacity
            self.checkpoint_dir = checkpoint_dir
            self.init_encode = init_encode
            self.init_decode = init_decode
            self.encode_sequence_length = encode_sequence_length
            self.decode_sequence_length = decode_sequence_length
            self.num_sampled = num_sampled
            self.dtype = dtype
            self.config = None
            self.graph = None
        else:
            with open(config_path, 'rb') as f:
                init_attr = pickle.load(f)
            for key in init_attr:
                setattr(self, key, init_attr[key])
    def set_config(self, config_path):
        self.config_path = config_path
        with open(config_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
    def _inference(self, encoder_inputs, decoder_inputs, forward_only, is_beam_search, beam_size):
        lstm_cells = []
        self.keep_prob = tf.placeholder(tf.float32)
        for layer_size in self.layers_size:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(layer_size)
            lstm_cells.append(tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=self.keep_prob))
        cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
        
        ##TODO
        self.output_projection = None
        self.softmax_loss_function = None
            
        
        out = embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols = self.vocab_en_size,
                                num_decoder_symbols = self.vocab_de_size,
                                encode_embedding_size = self.embedding_en_size,
                                decode_embedding_size = self.embedding_de_size,
                                num_heads=1,
                                output_projection=self.output_projection,
                                feed_previous=forward_only,
                                dtype=self.dtype,
                                scope=None,
                                initial_state_attention=False,
                                initial_embedding_encode = self.init_encode,
                                initial_embedding_decode = self.init_decode,
                                is_beam_search = is_beam_search,
                                beam_size = beam_size)
        return out
        
        
    def _build_loss(self, outputs, targets, weights):
        self.real_output = tf.stack(outputs[:-1], axis = 1)
        print (self.real_output, targets, weights)
        return tf.contrib.seq2seq.sequence_loss(self.real_output,
                                           targets, 
                                           weights,
                                          softmax_loss_function = self.softmax_loss_function)
    def _create_optimizer_node(self, loss):
        """Create the optimizer node of the graph."""
        self.lr_var = tf.Variable(self.learning_rate, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                          self.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_var)
        _train_op = optimizer.apply_gradients(zip(grads, tvars))
        return _train_op
    def build(self, sess, encode_data = None, decode_data = None, is_beam_search = False,
                 beam_size = 5, forward_only = False):
            
        self.coord = tf.train.Coordinator()
        self.queue, self.enqueue_threads, self.basic_enqueue = create_queue(sess, 
                                                                    self.coord,
                                                                    encode_data,
                                                                    decode_data,
                                                                    self.capacity)
        self.encode_inputs, self.decode_inputs, self.target_weights = self.queue.dequeue_many(self.batch_size)
        self.closing_queue = self.queue.close(cancel_pending_enqueues = True)
        #shift target by one
        self.targets = self.decode_inputs[:, 1:]
        if is_beam_search and forward_only:
            self.outputs,_, self.states ,self.beam_path,_ = self._inference(self.encode_inputs,
                                                        self.decode_inputs,
                                                        forward_only,
                                                        is_beam_search,
                                                        beam_size)
        else:
            self.outputs, _, self.states = self._inference(self.encode_inputs,
                                                        self.decode_inputs,
                                                        forward_only,
                                                        is_beam_search,
                                                        beam_size)
        if not is_beam_search:
            self.loss = self._build_loss(self.outputs, self.targets, self.target_weights)
            self.train_op = self._create_optimizer_node(self.loss)
        self.saver = tf.train.Saver()
    def start_run(self):
        for i in range(len(self.enqueue_threads)):
            self.enqueue_threads[i].start()
    def end_run(self):
        self.sess.run(self.closing_queue)
        self.coord.request_stop()
