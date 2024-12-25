import os
import nltk
import ssl
import streamlit as st
import random
import json
import csv
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure SSL and download NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Hardcoded intents (remove the file loading for simplicity)
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "How are you", "What's up"],
        "responses": ["Hi there!", "Hello!", "I'm fine, thank you!", "Going good! How about you?"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "Goodbye", "See you later"],
        "responses": ["Goodbye!", "See you later!", "Have a great day!"]
    }
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Function to chat with the chatbot
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Streamlit application
def main():
    st.title("Chatbot Using NLP")
    st.sidebar.title("Menu")

    # Create a sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Chat with the Bot")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # User input
        user_input = st.text_input("You:", key="user_input")
        if user_input:
            # Get chatbot response
            response = chatbot(user_input)
            st.session_state['chat_history'].append({"user": user_input, "bot": response})

            # Display the chat history
            for chat in st.session_state['chat_history']:
                st.write(f"You: {chat['user']}")
                st.write(f"Bot: {chat['bot']}")

            # Save to CSV
            if not os.path.exists("chat_log.csv"):
                with open("chat_log.csv", "w", newline="", encoding="utf-8") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp"])
            with open("chat_log.csv", "a", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow([user_input, response, timestamp])

    elif choice == "Conversation History":
        st.subheader("Conversation History")
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header
                for row in csv_reader:
                    st.write(f"You: {row[0]}")
                    st.write(f"Bot: {row[1]}")
                    st.write(f"Timestamp: {row[2]}")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.subheader("About")
        st.write("This is a simple chatbot application created using NLP and machine learning.")

if __name__ == "__main__":
    main()
