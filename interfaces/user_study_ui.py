import gradio as gr
from core.user_study_chat_engine import init_chat_history, reset_chat_history, generate_chat_response
from core.user_study_chat_engine import stream_chat_messages
from user_study_config import STREAM_THOUGHTS
from core.user_study_ui_config import title, description, example1, example2

user_info = {"username": ""}


def login(username):
    if username:
        user_info["username"] = username

        init_chat_history(user_info)
        return {
            login_section: gr.Column(visible=False),
            chatbot_section: gr.Column(visible=True)
        }
    return None, None


def leave_session():
    reset_chat_history(user_info)
    user_info["username"] = ""
    return {
        login_section: gr.Column(visible=True),
        chatbot_section: gr.Column(visible=False),
        username_input: gr.Textbox(value=""),
        chat.chatbot_value: []
    }


def response(message, history):
    if STREAM_THOUGHTS:
        yield from stream_chat_messages(message, user_info)
    else:
        yield generate_chat_response(message, user_info)

with gr.Blocks() as demo:
    with gr.Column(visible=True) as login_section:
        gr.Markdown("## Enter your details to start the chatbot")
        username_input = gr.Textbox(
            placeholder="Enter your username", label="Username")
        submit_button = gr.Button("Submit")

    with gr.Column(visible=False) as chatbot_section:
        chat = gr.ChatInterface(
            fn=response,
            type='messages',
            title=title,
            description=description,
            examples=["Hello", example1, example2],
        )

        gr.Row(scale=2)
        with gr.Row():
            gr.Column(scale=2)
            gr.Column()
            leave_button = gr.Button("Leave", variant='stop')

    submit_button.click(login, [username_input], [
                        login_section, chatbot_section])
    leave_button.click(leave_session, inputs=[], outputs=[
                       login_section, chatbot_section, username_input, chat.chatbot_value])
