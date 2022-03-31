import tensorflow_hub as hub
import tensorflow as tf

class BertPTC_sl():


    def __init__(self, float_type, num_labels, max_seq_length, hub_path):
        super(BertPTC_sl, self).__init__()
        self.input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        self.input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        self.input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
        self.encoder = hub.KerasLayer(hub_path, trainable=True)

        # Custom parts
        self.intermediate_classifier = tf.keras.layers.Dense(num_labels,
                                                             dtype=float_type, activation='relu')
        self.softmax_layer = tf.keras.layers.Softmax()
    #
    def getSingleLabelModel(self):
        encoder_inputs = dict(
            input_word_ids=self.input_word_ids,
            input_mask=self.input_mask,
            input_type_ids=self.input_type_ids,
        )
        outputs = self.encoder(encoder_inputs)
        sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
        intermediate_output = self.intermediate_classifier(sequence_output)
        output = self.softmax_layer(intermediate_output)
        model = tf.keras.Model(inputs=[self.input_word_ids, self.input_mask, self.input_type_ids], outputs=output)
        return model