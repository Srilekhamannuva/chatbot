import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
import openai
import os

# ✅ Securely set your OpenAI key (best: use st.secrets or system env)
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🧠 Streamlit App
def main():
    st.set_page_config(page_title="Smart Summarizing Chatbot", page_icon="🧠", layout="centered")
    st.title("🧠 Smart Summarizing Chatbot")
    st.caption("Powered by LangChain + OpenAI")

    # Initialize memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationSummaryBufferMemory(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
            max_token_limit=200  # more tokens for longer memory
        )

    # Initialize LLM chain
    if "chain" not in st.session_state:
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""
You are a helpful assistant. Here's the conversation so far:
{history}

User: {input}
Assistant:"""
        )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        st.session_state.chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=st.session_state.memory
        )

    # Store full chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 🗣️ Display all previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 💬 User input box (chat style)
    if user_input := st.chat_input("Type your message..."):
        # Display user message immediately
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.run(user_input)
                st.markdown(response)

        # Save bot message to history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # 🧾 Display summarized memory (for curiosity)
    with st.expander("🧠 Conversation Summary (Memory Buffer)"):
        st.write(st.session_state.memory.moving_summary_buffer)


if __name__ == "__main__":
    main()
