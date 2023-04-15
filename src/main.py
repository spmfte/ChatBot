from model import ChatBotModel

def main():
    chatbot = ChatBotModel()
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        response = chatbot.generate_response(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()

