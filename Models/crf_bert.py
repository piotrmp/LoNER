import tensorflow_hub as hub
import tensorflow as tf
from tensorflow_addons.layers import CRF


class BertPTC_CRF():
    def __init__(self, float_type, num_labels, max_seq_length, hub_path):
        super(BertPTC_CRF, self).__init__()
        self.input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        self.input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        self.input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
        self.encoder = hub.KerasLayer(hub_path, trainable=True)
        self.classifier = tf.keras.layers.Dense(num_labels, name='output', dtype=float_type)
        self.CRF = CRF(num_labels, use_kernel=False)

    def getModelCRFLayer(self):

        encoder_inputs = dict(
            input_word_ids=self.input_word_ids,
            input_mask=self.input_mask,
            input_type_ids=self.input_type_ids,
        )
        outputs = self.encoder(encoder_inputs)
        sequence_output = outputs["sequence_output"]
        dense_output = self.classifier(sequence_output)
        output = self.CRF(dense_output, self.input_mask > 0)

        model = tf.keras.Model(inputs=[self.input_word_ids, self.input_mask, self.input_type_ids], outputs=output)
        return model

