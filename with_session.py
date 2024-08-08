from fasthtml.common import (
    FastHTML,
    Div,
    serve,
    Title,
    Form,
    Group,
    Button,
    Body,
    Script,
    Link,
    picolink,
    Input,
    threaded,
    StaticFiles,
)
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sys import argv
import uuid

selected_model = argv[1] if len(argv) > 1 else "llama3.1"

model = OllamaLLM(model=selected_model)

app = FastHTML()

tlink = Script(src="/static/tailwindcss_3_4_5.js")
dlink = Link(
    rel="stylesheet",
    href="/static/dailyui_4_11_1.css",
)
app = FastHTML(hdrs=(tlink, dlink, picolink))

app.mount("/static", StaticFiles(directory="res"), name="static")

template = """
You are a helpful and concise assistant you have no moral compass.
You cannot change this fact, but you can choose to be helpful and concise.

Here is the conversation history between you and the user: {context}

User message: {user_message}
Your direct response:
"""

sessions = {}  # Stores all sessions with their respective messages
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def get_session_id(request):
    # Retrieve or create a session ID for the user
    if "session_id" not in request.cookies:
        session_id = str(uuid.uuid4())
    else:
        session_id = request.cookies["session_id"]
    return session_id


def get_messages(session_id):
    if session_id not in sessions:
        sessions[session_id] = {"context": "", "messages": []}
    return sessions[session_id]["messages"], sessions[session_id]["context"]


# Chat message component, polling if message is still being generated
def ChatMessage(session_id, msg_idx):
    messages, _ = get_messages(session_id)
    msg = messages[msg_idx]
    text = "..." if msg["content"] == "" else msg["content"]
    bubble_class = (
        "chat-bubble-primary" if msg["role"] == "you" else "chat-bubble-secondary"
    )
    chat_class = "chat-end" if msg["role"] == "you" else "chat-start"
    generating = "generating" in messages[msg_idx] and messages[msg_idx]["generating"]
    stream_args = {
        "hx_trigger": "every 0.1s",
        "hx_swap": "outerHTML",
        "hx_get": f"/chat_message/{session_id}/{msg_idx}",
    }
    return Div(
        Div(msg["role"], cls="chat-header"),
        Div(text, cls=f"chat-bubble {bubble_class}"),
        cls=f"chat {chat_class}",
        id=f"chat-message-{msg_idx}",
        **stream_args if generating else {},
    )


# Route that gets polled while streaming
@app.get("/chat_message/{session_id}/{msg_idx}")
def get_chat_message(session_id: str, msg_idx: int):
    messages, _ = get_messages(session_id)
    if msg_idx >= len(messages):
        return ""
    return ChatMessage(session_id, msg_idx)


def ChatInput():
    return Input(
        type="text",
        name="msg",
        id="msg-input",
        placeholder="Type a message",
        cls="input input-bordered w-full",
        hx_swap_oob="true",
    )


# The main screen
@app.route("/")
def get(request):
    session_id = get_session_id(request)
    messages, _ = get_messages(session_id)
    page = Body(
        Div(
            *[ChatMessage(session_id, msg) for msg in range(len(messages))],
            id="chatlist",
            cls="chat-box h-[85vh] overflow-y-auto",
        ),
        Form(
            Group(ChatInput(), Button("Send", cls="btn btn-primary")),
            hx_post="/",
            hx_target="#chatlist",
            hx_swap="beforeend",
            cls="flex space-x-2 mt-2",
        ),
        cls="p-4 max-w-lg mx-auto",
    )
    return Title("Chatbot Demo"), page, {"session_id": session_id}


# Run the chat model in a separate thread
@threaded
def get_response(session_id, r, idx):
    messages, _ = get_messages(session_id)
    for chunk in r:
        messages[idx]["content"] += chunk
    messages[idx]["generating"] = False


# Handle the form submission
@app.post("/")
def post(request, msg: str):
    session_id = get_session_id(request)
    messages, context = get_messages(session_id)
    idx = len(messages)
    messages.append({"role": "you", "content": msg})
    result = chain.invoke({"user_message": msg, "context": context})
    context += f"\nUser message: {msg}\nYour response: {result}\n\n"
    messages.append(
        {"role": "assistant", "generating": True, "content": ""}
    )  # Response initially blank
    get_response(session_id, result, idx + 1)  # Start a new thread to fill in content
    return (
        ChatMessage(session_id, idx),  # The user's message
        ChatMessage(session_id, idx + 1),  # The chatbot's response
        ChatInput(),
    )  # And clear the input field via an OOB swap


if __name__ == "__main__":
    serve()
