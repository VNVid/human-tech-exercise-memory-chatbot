import gradio as gr
from gradio import ChatMessage
from core.chat_engine import (
    init_chat_history, reset_chat_history, stream_chat_messages
)

# ---------------------------------------------------------------------
user_info = {"username": ""}          # persisted between calls
# ---------------------------------------------------------------------

def ui():
    with gr.Blocks() as demo:

        # ---- 1. login panel ------------------------------------------------
        with gr.Column(visible=True) as login_panel:
            gr.Markdown("### 💪 Exercise-Coach Chatbot")
            username = gr.Textbox(label="Pick a username")
            start_btn = gr.Button("Start chatting")

        # ---- 2. chatbot panel ---------------------------------------------
        with gr.Column(visible=False) as chat_panel:
            chatbot   = gr.Chatbot(
                label="Coach",
                type="messages",
                avatar_images=(None, "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png")
            )
            chat_state = gr.State([])                 # stores `list[ChatMessage]`
            msg_box    = gr.Textbox(placeholder="Ask me for an exercise…",
                                    scale=10)
            leave_btn  = gr.Button("Leave session", variant="stop")

        # ---- callbacks -----------------------------------------------------
        def _login(name):
            if not name:
                return gr.update(), gr.update()
            user_info["username"] = name
            init_chat_history(user_info)
            return {login_panel: gr.update(visible=False),
                    chat_panel:  gr.update(visible=True)}

        start_btn.click(_login, username, [login_panel, chat_panel])

        def _talk(user_text, history, progress=gr.Progress()):
            # stream the agent’s chain-of-thought
            for delta in stream_chat_messages(user_text, user_info, progress):
                history = history + delta
                yield history, ""          # second output resets textbox

        msg_box.submit(
                        _talk,
                        [msg_box, chat_state],  # inputs
                        [chatbot, msg_box]  # outputs remain unchanged
            )

        def _leave():
            reset_chat_history(user_info)
            user_info["username"] = ""
            return {login_panel: gr.update(visible=True),
                    chat_panel:  gr.update(visible=False),
                    msg_box:     gr.update(value=""),
                    chatbot:     []}

        leave_btn.click(_leave, None,
                        [login_panel, chat_panel, msg_box, chatbot])

    return demo


if __name__ == "__main__":
    ui().launch()