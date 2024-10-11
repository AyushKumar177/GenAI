import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import webbrowser
import datetime


class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        self.setup_gui()
        self.load_chatbot_data()
        self.conversation_history = []

    def setup_gui(self):
        self.master.title("Customer Chatbot")
        self.master.geometry("500x600")
        self.master.configure(bg="#f0f0f0")

        style = ttk.Style()
        style.theme_use("clam")

        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_history = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, width=60, height=25, font=("Arial", 10)
        )
        self.chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_history.config(state=tk.DISABLED)

        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)

        self.user_input = ttk.Entry(input_frame, width=50, font=("Arial", 10))
        self.user_input.pack(side=tk.LEFT, padx=(0, 5), expand=True, fill=tk.X)
        self.user_input.bind("<Return>", lambda event: self.send_message())

        self.send_button = ttk.Button(
            input_frame, text="Send", command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.clear_button = ttk.Button(
            button_frame, text="Clear Chat", command=self.clear_chat
        )
        self.clear_button.pack(side=tk.LEFT, padx=(0, 5))

        self.save_button = ttk.Button(
            button_frame, text="Save Chat", command=self.save_chat
        )
        self.save_button.pack(side=tk.LEFT)

        self.help_button = ttk.Button(button_frame, text="Help", command=self.show_help)
        self.help_button.pack(side=tk.RIGHT)

    def load_chatbot_data(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open("data\intents.json").read())
        self.words = pickle.load(open("data\words.pkl", "rb"))
        self.classes = pickle.load(open("data\classes.pkl", "rb"))
        self.model = load_model("bot_model.h5")

    def send_message(self):
        user_message = self.user_input.get()
        self.user_input.delete(0, tk.END)
        if user_message:
            self.update_chat_history(f"You: {user_message}")
            bot_response = self.get_bot_response(user_message)
            self.update_chat_history(f"Bot: {bot_response}")
            self.conversation_history.append((user_message, bot_response))

    def update_chat_history(self, message):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, message + "\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def get_bot_response(self, user_message):
        if user_message.lower() in ["exit", "quit", "bye"]:
            return "Goodbye! Have a great day!"
        elif user_message.lower().startswith("search "):
            query = user_message[7:]
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"I've opened a web search for '{query}'."
        elif user_message.lower() == "time":
            return (
                f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
            )
        else:
            ints = self.predict_class(user_message)
            return self.get_response(ints)

    def clean_up_sentence(self, sentence):
        return [
            self.lemmatizer.lemmatize(word.lower())
            for word in nltk.word_tokenize(sentence)
        ]

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [1 if word in sentence_words else 0 for word in self.words]
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [
            {"intent": self.classes[r[0]], "probability": str(r[1])} for r in results
        ]

    def get_response(self, intents_list):
        if not intents_list:
            return "I'm not sure how to respond to that. Can you please rephrase your question?"
        tag = intents_list[0]["intent"]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "I'm sorry, I don't have a specific response for that. Can you try asking something else?"

    def clear_chat(self):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state=tk.DISABLED)
        self.conversation_history.clear()

    def save_chat(self):
        filename = (
            f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(filename, "w") as f:
            for user_msg, bot_msg in self.conversation_history:
                f.write(f"You: {user_msg}\n")
                f.write(f"Bot: {bot_msg}\n\n")
        messagebox.showinfo("Chat Saved", f"Chat history has been saved to {filename}")

    def show_help(self):
        help_text = """
        Welcome to the Enhanced NLP Chatbot!

        Special Commands:
        - Type 'exit', 'quit', or 'bye' to end the conversation.
        - Type 'search <query>' to open a web search.
        - Type 'time' to get the current time.

        Features:
        - Clear Chat: Clears the current conversation.
        - Save Chat: Saves the conversation history to a file.
        - Help: Shows this help message.

        Enjoy chatting!
        """
        messagebox.showinfo("Chatbot Help", help_text)


if __name__ == "__main__":
    root = tk.Tk()
    ChatbotGUI(root)
    root.mainloop()