import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def generate_response(user_input, past_chat_history=None, max_length=256):
    # Encode the user input and concatenate it with past chat history, if it exists
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Prepare the bot's input IDs: either concatenate with history or just use the new input
    bot_input_ids = torch.cat([past_chat_history, new_user_input_ids], dim=-1) if past_chat_history is not None else new_user_input_ids

    # Compute attention masks (1 for real tokens, 0 for padding)
    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)  # assuming bot_input_ids is a tensor

    # Generate a response from the model with attention masks
    chat_output = model.generate(bot_input_ids, attention_mask=attention_mask, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    
    return chat_output[:, bot_input_ids.shape[-1]:]


def chat():
    # Initialize chat history (None at the beginning)
    chat_history = None
    print("Chatbot: Hello! Ask me anything or type 'quit' to end.")

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Generate the response
        chat_output = generate_response(user_input, chat_history)
        chat_history = chat_output  # Update chat history with the new response

        # Decode and print the response
        response_text = tokenizer.decode(chat_output[0], skip_special_tokens=True)
        print("Chatbot:", response_text)

# Start chatting
chat()
