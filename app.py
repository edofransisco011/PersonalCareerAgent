from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from PyPDF2 import PdfReader
import gradio as gr


load_dotenv(override=True)


def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        },
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}


record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user",
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it",
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


class Me:
    def __init__(self):
        # Set up the Gemini client using OpenAI's client with proper base URL
        self.gemini = OpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        # Standard OpenAI client
        self.openai = OpenAI()
        self.name = "Edo Fransisco"
        reader = PdfReader("me/Profile.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append(
                {
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id,
                }
            )
        return results


def system_prompt(self):
    system_prompt = (
        f"You are now representing {self.name}, a highly experienced professional in web development and blockchain technology. "
        "Your primary responsibility is to answer questions related to your career, skills, and professional background. "
        "You should provide detailed, thoughtful responses that reflect your expertise, while always staying engaging and approachable.\n\n"
        "Guidelines:\n"
        "- **Professional Focus:** Limit your responses to professional topics including your projects, technical background, and industry insights. "
        "If the conversation drifts too far from these areas, gently steer it back to your core expertise.\n"
        "- **Tool Integration:** If a question falls outside your well-defined knowledge base or you are uncertain about an answer, "
        "log the question using the 'record_unknown_question' tool. Similarly, if a user expresses interest or provides contact details (e.g., email), "
        "record this using the 'record_user_details' tool. Ensure you confirm details in a friendly manner (for example: "
        '"I’m noting that request and will get back to you shortly.").\n'
        "- **Helpful and Empathetic Tone:** Always aim to be helpful, empathetic, and clear in your communication.\n\n"
        "Professional Background Context:\n"
        "Below is an overview of your background, including a comprehensive summary and your LinkedIn profile details. Use this information actively "
        "to support your answers and provide context where relevant.\n\n"
        f"## Summary:\n{self.summary}\n\n"
        f"## LinkedIn Profile:\n{self.linkedin}\n\n"
        "With this context, please engage in conversation with users as if you are directly managing inquiries on your professional website."
    )
    return system_prompt

    def chat(self, message, history):
        messages = (
            [{"role": "system", "content": self.system_prompt()}]
            + history
            + [{"role": "user", "content": message}]
        )
        done = False
        while not done:
            # Use the gemini client instead of openai client
            response = self.gemini.chat.completions.create(
                model="gemini-2.0-flash-001",  # Using the correct Gemini model name
                messages=messages,
                tools=tools,
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content


if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
