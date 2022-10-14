import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

def custom_w_init(kernel_size):
    zeros = np.zeros(shape=(kernel_size, 1, 1))
    one_i = int((kernel_size-1)/2)
    zeros[one_i] = 1
    return tf.constant_initializer(zeros)

class Conv1SplitConcat(tf.keras.layers.Layer):
    """Custom adaptive convolutional layer v.1.
       Initialized with simple 000010000 kernels that evolve during training.
       Each kernel length is tailored to specific propaganda category mean span length in training dataset"""
    def __init__(self, kernel_lenghts: list, sentence_length: int, trainable=True):
        super(Conv1SplitConcat, self).__init__()
        self.kernel_lengths = kernel_lenghts
        self.sentence_length = sentence_length
        self.trainable = trainable

    def build(self, input_shape):
        assert input_shape[-1] == len(self.kernel_lengths)
        self.custom_layers = self.create_custom_layers()

    def create_custom_layers(self):
        layers = [tf.keras.layers.Conv1D(trainable=self.trainable, filters=1, kernel_size=i,
                                         input_shape=(self.sentence_length, 1),
                                         padding='same', kernel_initializer=custom_w_init(i),
                                         ) for i in self.kernel_lengths]
        return layers

    def call(self, inputs, **kwargs):
        outs = []
        split_input = tf.split(inputs, num_or_size_splits=len(self.kernel_lengths), axis=-1)
        for inpt, lyr in zip(split_input, self.custom_layers):
            outs.append(lyr(inpt))
        return tf.concat(outs, axis=-1)



class AdaptiveGaussianConvLayer(tf.keras.layers.Layer):
    """Custom adaptive convolutional layer v.2.
       Parameters of gaussian distribution of kernels are passed during layer
       initialization and behave as trainable parameters in the model"""
    def __init__(self):
        super(AdaptiveGaussianConvLayer, self).__init__()
        self.X = None
    
    def build(self, input_shape):
        assert input_shape is not None
        val_shape = input_shape[0]
        sigma_shape = input_shape[1]
        mu_shape = input_shape[2]
        assert len(val_shape) == 3
        assert len(sigma_shape) == 3
        assert len(mu_shape) == 3
        assert val_shape[1] == sigma_shape[1]
        assert val_shape[1] == mu_shape[1]
        assert sigma_shape[2] == 1
        assert mu_shape[2] == 1
        N = val_shape[1]
        X = []
        for i in range(N):
            X.append(np.array(range(N)) - i)
        self.X = tf.constant(np.array(X).astype(np.float32))
    
    def call(self, inputs):
        V = tf.transpose(inputs[0], (0, 2, 1))
        sigmas = inputs[1]
        mus = inputs[2]
        W = tf.math.exp(tf.math.square((self.X - mus) / sigmas) * (-1 / 2))
        output = tf.matmul(V, W)
        return tf.transpose(output, (0, 2, 1))




class BertPTC_slaner():
    def __init__(self, float_type, num_labels: int, max_seq_length: int, hub_path, kernels: list, arch_params):
        super(BertPTC_slaner, self).__init__()
        self.arch_params = arch_params
        
        self.input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        self.input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        self.input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
        self.encoder = hub.KerasLayer(hub_path, trainable=True)
        
        # Custom parts
        self.dropout = tf.keras.layers.Dropout(0.2, noise_shape=(None, max_seq_length, 1))
        self.intermediate_classifier = tf.keras.layers.Dense(num_labels, dtype=float_type, activation='relu')
        if self.arch_params['score_input']:
            self.concat = tf.keras.layers.Concatenate(axis=-1)
        if self.arch_params['random_init']:
            initializer = tf.keras.initializers.Constant(1.0)
            self.sigma_generator = tf.keras.layers.Dense(1, dtype=float_type, activation='linear',
                                                         bias_initializer=initializer,
                                                         trainable=self.arch_params['with_sigma'])
            self.mu_generator = tf.keras.layers.Dense(1, dtype=float_type, activation='linear',
                                                      trainable=self.arch_params['with_mu'])
        else:
            initializer = tf.keras.initializers.Constant(1.0)
            self.sigma_generator = tf.keras.layers.Dense(1, dtype=float_type, kernel_initializer='zeros',
                                                         bias_initializer=initializer, activation='linear',
                                                         trainable=self.arch_params['with_sigma'])
            self.mu_generator = tf.keras.layers.Dense(1, dtype=float_type, kernel_initializer='zeros',
                                                      bias_initializer='zeros', activation='linear',
                                                      trainable=self.arch_params['with_mu'])
        self.adaptive_conv = AdaptiveGaussianConvLayer()
        if self.arch_params['category_conv']:
            self.category_conv = Conv1SplitConcat(kernel_lenghts=kernels, sentence_length=max_seq_length)
        self.adder = tf.keras.layers.Add()
        self.softmax_layer = tf.keras.layers.Softmax()
        if self.arch_params['presoftmax']:
            self.normaliser = tf.keras.layers.Lambda(lambda x: tf.linalg.normalize(x, ord=1, axis=-1)[0])
    
    def getSimpleModel(self):
        # Obtain input from BERT
        encoder_inputs = dict(
            input_word_ids=self.input_word_ids,
            input_mask=self.input_mask,
            input_type_ids=self.input_type_ids,
        )
        bert_outputs = self.encoder(encoder_inputs)
        bert_sequence_output = bert_outputs["sequence_output"]  # [batch_size, seq_length, 768].
        # Dropout words
        if self.arch_params['dropout']:
            bert_sequence_output = self.dropout(bert_sequence_output)
        # Compute category scores
        scores = self.intermediate_classifier(bert_sequence_output)  # [batch_size, seq_length, num_labels]
        # Obtain convolution parameters
        if self.arch_params['bert_input'] and self.arch_params['score_input']:
            conv_param_input = self.concat([bert_sequence_output, scores])
        elif self.arch_params['bert_input']:
            conv_param_input = bert_sequence_output
        elif self.arch_params['score_input']:
            conv_param_input = scores
        sigmas = self.sigma_generator(conv_param_input)
        mus = self.mu_generator(conv_param_input)
        # Prepare scores for convolution
        if self.arch_params['presoftmax']:
            convoluted_scores = self.softmax_layer(scores)
        else:
            convoluted_scores = scores
        # Compute convolution
        if not self.arch_params['reversed']:
            convoluted_scores = self.adaptive_conv([convoluted_scores, sigmas, mus])
            if self.arch_params['category_conv']:
                convoluted_scores = self.category_conv(convoluted_scores)
        else:
            if self.arch_params['category_conv']:
                convoluted_scores = self.category_conv(convoluted_scores)
            convoluted_scores = self.adaptive_conv([convoluted_scores, sigmas, mus])
        if self.arch_params['residual_connection']:
            convoluted_scores = self.adder([convoluted_scores, scores])
        # Normalise convolution output
        if self.arch_params['presoftmax']:
            output = self.normaliser(convoluted_scores)
        else:
            output = self.softmax_layer(convoluted_scores)
        model = tf.keras.Model(inputs=[self.input_word_ids, self.input_mask, self.input_type_ids], outputs=output)
        return model


class ParseSpreadingLayer(tf.keras.layers.Layer):
    def __init__(self,trainable=True):
        super(ParseSpreadingLayer, self).__init__()
        self.trainable=trainable
    
    def build(self, input_shape):
        assert input_shape is not None
        emb_shape, parse_shape = input_shape
        assert len(emb_shape) == 3
        assert len(parse_shape) == 3
        assert emb_shape[1] == parse_shape[1]
        self.scaling_lookup = self.add_weight(
            name='Lookup table',
            shape=(emb_shape[1] + 1),
            trainable=self.trainable,
            initializer=tf.keras.initializers.Ones)
    
    def call(self, inputs):
        word_embeddings, parse_mask = inputs
        parse_mask = tf.cast(parse_mask, word_embeddings.dtype)
        parse_mask_t = tf.transpose(parse_mask, (0, 2, 1))
        span_embeddings = tf.matmul(parse_mask_t, word_embeddings)
        weight = tf.math.reduce_sum(parse_mask_t, axis=2)
        weight = tf.reshape(weight, (-1, weight.shape[1], 1))
        span_embeddings = tf.math.divide_no_nan(span_embeddings, weight)
        span_embeddings_t = tf.transpose(span_embeddings, (0, 2, 1))
        multiplier = tf.gather(self.scaling_lookup, tf.cast(weight, tf.int32))
        parse_mask_t_scaled = tf.multiply(parse_mask_t, multiplier)
        new_word_embeddings = tf.transpose(tf.matmul(span_embeddings_t, parse_mask_t_scaled),
                                           (0, 2, 1)) + word_embeddings
        weight2 = tf.math.reduce_sum(parse_mask_t_scaled, axis=1) + 1.0
        weight2 = tf.reshape(weight2, (-1, weight2.shape[1], 1))
        result = tf.math.divide_no_nan(new_word_embeddings, weight2)
        return result


class BertPTC_parse():
    def __init__(self, float_type, num_labels: int, max_seq_length: int, hub_path, max_k):
        super(BertPTC_parse, self).__init__()
        self.input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        self.input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        self.input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
        self.input_parse_mask = tf.keras.layers.Input(shape=(max_seq_length, max_k), dtype=tf.int32,
                                                      name='input_parse_mask')
        self.encoder = hub.KerasLayer(hub_path, trainable=True)
        
        # Custom parts
        self.dropout = tf.keras.layers.Dropout(0.2, noise_shape=(None, max_seq_length, 1))
        self.parse_spreading = ParseSpreadingLayer(trainable=False)
        self.intermediate_classifier = tf.keras.layers.Dense(num_labels, dtype=float_type, activation='relu')
        self.softmax_layer = tf.keras.layers.Softmax()
    
    def getSimpleModel(self):
        # Obtain input from BERT
        encoder_inputs = dict(
            input_word_ids=self.input_word_ids,
            input_mask=self.input_mask,
            input_type_ids=self.input_type_ids,
        )
        bert_outputs = self.encoder(encoder_inputs)
        bert_sequence_output = bert_outputs["sequence_output"]  # [batch_size, seq_length, 768].
        bert_sequence_output = self.dropout(bert_sequence_output)
        # Parse spreading
        new_sequence_output = self.parse_spreading([bert_sequence_output, self.input_parse_mask])
        # Compute category scores
        scores = self.intermediate_classifier(new_sequence_output)  # [batch_size, seq_length, num_labels]
        # Normalise output
        output = self.softmax_layer(scores)
        model = tf.keras.Model(
            inputs=[self.input_word_ids, self.input_mask, self.input_type_ids, self.input_parse_mask], outputs=output)
        return model
