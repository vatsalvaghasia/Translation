from django.http import HttpResponse
from django.shortcuts import render
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.layers import Input, LSTM, Dense


def home(request):
    return render(request, "home.html")


def result(request):
    msg = request.GET['msg']
    ans = deocded_output(msg)
    return render(request, "home.html", {'ans': ans})


cv = CountVectorizer(binary=True, tokenizer=lambda txt: txt.split(), stop_words=None, analyzer='char')

datafile = pickle.load(open("training_data.pkl", "rb"))
input_characters = datafile['input_characters']
target_characters = datafile['target_characters']
max_input_length = datafile['max_input_length']
max_target_length = datafile['max_target_length']
num_en_chars = datafile['num_en_chars']
num_dec_chars = datafile['num_dec_chars']

# Inference model
# load the model
model = models.load_model("s2s")
# construct encoder model from the output of second layer
# discard the encoder output and store only states.
enc_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
# add input object and state from the layer.
en_model = Model(model.input[0], [state_h_enc, state_c_enc])

# create Input object for hidden and cell state for decoder
# shape of layer with hidden or latent dimension
dec_state_input_h = Input(shape=(256,), name="input_3")
dec_state_input_c = Input(shape=(256,), name="input_4")
dec_states_inputs = [dec_state_input_h, dec_state_input_c]

# add input from the encoder output and initialize with
# states.
dec_lstm = model.layers[3]
dec_outputs, state_h_dec, state_c_dec = dec_lstm(
    model.input[1], initial_state=dec_states_inputs
)
dec_states = [state_h_dec, state_c_dec]
dec_dense = model.layers[4]
dec_outputs = dec_dense(dec_outputs)
# create Model with the input of decoder state input and encoder input
# and decoder output with the decoder states.
dec_model = Model(
    [model.input[1]] + dec_states_inputs, [dec_outputs] + dec_states
)


def decode_sequence(input_seq):
    # create dict object to get character from the index.
    reverse_target_char_index = dict(enumerate(target_characters))
    # get the states from the user input sequence
    states_value = en_model.predict(input_seq)

    # fit target characters and
    # initialize every first character to be 1 which is '\t'.
    # Generate empty target sequence of length 1.
    co = cv.fit(target_characters)
    target_seq = np.array([co.transform(list("\t")).toarray().tolist()], dtype="float32")

    # if the iteration reaches the end of text than it will be stop the it
    stop_condition = False
    # append every predicted character in decoded sentence
    decoded_sentence = ""
    while not stop_condition:
        # get predicted output and discard hidden and cell state.
        output_chars, h, c = dec_model.predict([target_seq] + states_value)

        # get the index and from dictionary get character from it.
        char_index = np.argmax(output_chars[0, -1, :])
        text_char = reverse_target_char_index[char_index]
        decoded_sentence += text_char

        # Exit condition: either hit max length
        # or find stop character.
        if text_char == "\n" or len(decoded_sentence) > max_target_length:
            stop_condition = True
        # update target sequence to the current character index.
        target_seq = np.zeros((1, 1, num_dec_chars))
        target_seq[0, 0, char_index] = 1.0
        states_value = [h, c]
    # return the decoded sentence
    return decoded_sentence


def bagofcharacters(input_t):
    en_in_data = []
    pad_en = [1] + [0] * (len(input_characters) - 1)

    cv_inp = cv.fit(input_characters)
    en_in_data.append(cv_inp.transform(list(input_t)).toarray().tolist())

    if len(input_t) < max_input_length:
        for _ in range(max_input_length - len(input_t)):
            en_in_data[0].append(pad_en)

    return np.array(en_in_data, dtype="float32")


def deocded_output(msg):
    en_in_data = bagofcharacters(msg.lower() + ".")
    return decode_sequence(en_in_data)


def my_msg(msg):
    if not msg:
        return
