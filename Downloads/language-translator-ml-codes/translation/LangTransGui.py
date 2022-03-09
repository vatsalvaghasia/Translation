import tkinter
from tkinter import messagebox
from tkinter import *
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.layers import Input, LSTM, Dense
import speech_recognition as sr
import pyttsx3

BG_GRAY = "#ABB2B9"
BG_COLOR = "#000"
TEXT_COLOR = "#FFF"
FONT = "Melvetica 14"
FONT_BOLD = "Melvetica 13 bold"

cv = CountVectorizer(binary=True, tokenizer=lambda txt: txt.split(), stop_words=None, analyzer='char')


class LangTRans:
    def __init__(self):
        # initialize tkinter window and load the file
        self.window = Tk()
        self.main_window()
        self.datafile()

    def datafile(self):
        # get all datas from datafile and load the model.
        datafile = pickle.load(open("training_data.pkl", "rb"))
        self.input_characters = datafile['input_characters']
        self.target_characters = datafile['target_characters']
        self.max_input_length = datafile['max_input_length']
        self.max_target_length = datafile['max_target_length']
        self.num_en_chars = datafile['num_en_chars']
        self.num_dec_chars = datafile['num_dec_chars']
        self.loadmodel()

    # runwindow
    def run(self):
        self.window.mainloop()

    def main_window(self):
        # add title to window and configure it
        self.window.title("Language Translator")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=520, height=520, bg=BG_COLOR)

        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text="Translate to french Language!",
                           font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)
        line = Label(self.window, width=450, bg=BG_COLOR)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # create text widget where input and output will be displayed
        self.text_widget = Text(self.window, width=20, height=2, bg="#fff", fg="#000", font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # create scrollbar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # create bottom label where text widget will placed
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        # this is for user to put english text
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.65, relheight=0.06, rely=0.008, relx=0.008)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self.on_enter)
        # send button which will call on_enter function to send the text
        send_button2 = Button(bottom_label, text="Voice\n Input", font=FONT_BOLD, width=2, bg="Red",
                              command=lambda: self.voice_input(None))
        send_button2.place(relx=0.66, rely=0.008, relheight=0.06, relwidth=0.1325)

        # send button which will call on_enter function to send the text
        send_button = Button(bottom_label, text="Only Text", font=FONT_BOLD, width=8, bg="Red",
                             command=lambda: self.on_enter(None))
        send_button.place(relx=0.80, rely=0.008, relheight=0.03, relwidth=0.20)
        # send button which will call on_enter function to send the text
        send_button1 = Button(bottom_label, text="Voice", font=FONT_BOLD, width=2, bg="Red",
                              command=lambda: self.on_enter_voice(None))
        send_button1.place(relx=0.80, rely=0.04, relheight=0.027, relwidth=0.20)

    def loadmodel(self):
        # Inference model
        # load the model
        model = models.load_model("s2s")
        # construct encoder model from the output of second layer
        # discard the encoder output and store only states.
        enc_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
        # add input object and state from the layer.
        self.en_model = Model(model.input[0], [state_h_enc, state_c_enc])

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
        self.dec_model = Model(
            [model.input[1]] + dec_states_inputs, [dec_outputs] + dec_states
        )

    def decode_sequence(self, input_seq):
        reverse_target_char_index = dict(enumerate(self.target_characters))
        states_value = self.en_model.predict(input_seq)
        co = cv.fit(self.target_characters)
        target_seq = np.array([co.transform(list("\t")).toarray().tolist()], dtype="float32")

        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_chars, h, c = self.dec_model.predict([target_seq] + states_value)
            char_index = np.argmax(output_chars[0, -1, :])
            text_char = reverse_target_char_index[char_index]
            decoded_sentence += text_char
            if text_char == "\n" or len(decoded_sentence) > self.max_target_length:
                stop_condition = True
            target_seq = np.zeros((1, 1, self.num_dec_chars))
            target_seq[0, 0, char_index] = 1.0
            states_value = [h, c]
        return decoded_sentence

    def on_enter(self, event):
        # get user query and bot response
        msg = self.msg_entry.get()
        self.my_msg(msg, "English")
        self.decoded_output(msg, "French")

    def on_enter_voice(self, event):
        # get user query and bot response
        msg = self.msg_entry.get()
        self.my_msg(msg, "English")
        self.decoded_output_voice(msg, "French with vocal output")

    def voice_input(self, event):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source, phrase_time_limit=2)
            try:
                msg = r.recognize_google(audio)
                self.msg_entry.insert(0, msg)

            except:
                tkinter.messagebox.showerror(title="Error", message="Sorry could not recognize what you said.")

    def bagofcharacters(self, input_t):
        cv = CountVectorizer(binary=True, tokenizer=lambda txt: txt.split(), stop_words=None, analyzer='char')
        en_in_data = [];
        pad_en = [1] + [0] * (len(self.input_characters) - 1)

        cv_inp = cv.fit(self.input_characters)
        en_in_data.append(cv_inp.transform(list(input_t)).toarray().tolist())

        if len(input_t) < self.max_input_length:
            for _ in range(self.max_input_length - len(input_t)):
                en_in_data[0].append(pad_en)

        return np.array(en_in_data, dtype="float32")

    def decoded_output(self, msg, sender):
        self.text_widget.configure(state=NORMAL)
        en_in_data = self.bagofcharacters(msg.lower() + ".")
        self.text_widget.insert(END, str(sender) + " : " + self.decode_sequence(en_in_data)
                                + "\n\n")
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)

    def decoded_output_voice(self, msg, sender):
        self.text_widget.configure(state=NORMAL)
        en_in_data = self.bagofcharacters(msg.lower() + ".")
        self.text_widget.insert(END, str(sender) + " : " + self.decode_sequence(en_in_data)
                                + "\n\n")
        engine = pyttsx3.init()
        engine.setProperty("rate", 135)
        engine.say(self.decode_sequence(en_in_data))
        engine.runAndWait()
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)

    def my_msg(self, msg, sender):
        if not msg:
            return
        self.msg_entry.delete(0, END)
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, str(sender) + " : " + str(msg) + "\n")
        self.text_widget.configure(state=DISABLED)


# run the file
if __name__ == "__main__":
    LT = LangTRans()
    LT.run()
