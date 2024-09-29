import asyncio
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
from youtube_transcript_api.formatters import TextFormatter
from langchain_google_genai import ChatGoogleGenerativeAI
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import streamlit as st
import aiohttp
import re
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

async def get_title(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.title.text
            print(f"Title: {title}")
            return title

async def fetch_and_format_transcript(url):
    match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        return "No video ID found."

    video_id = match.group(1)
    formatter = TextFormatter()

    try:
        transcript = await asyncio.to_thread(YouTubeTranscriptApi.get_transcript, video_id)
        formatted_transcript = formatter.format_transcript(transcript)
        return formatted_transcript
    except NoTranscriptFound:
        try:
            transcript_list = await asyncio.to_thread(YouTubeTranscriptApi.list_transcripts, video_id)
            transcript = transcript_list.find_transcript(['hi'])
            translated_transcript = transcript.translate('en')
            transcript = await asyncio.to_thread(translated_transcript.fetch)
            formatted_transcript = formatter.format_transcript(transcript)
            return formatted_transcript
        except (NoTranscriptFound, TranscriptsDisabled):
            return "Subtitles have not been provided for this video yet."
    except TranscriptsDisabled:
        return "Subtitles have been disabled for this video."

async def ai(url):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=google_api_key
    )
    
    transcript = await fetch_and_format_transcript(url)
    messages = [
        (
            "system",
            "You're a helpful assistant. You will be given the transcripts of a YouTube Video. You have to provide a summary of the video",
        ),
        ("human", f"{transcript}"),
    ]
    ai_msg = await asyncio.to_thread(llm.invoke, messages)
    return ai_msg.content

def main():
    st.title("ClipWise🎥 - An AI tool for YouTube")
    st.write("Enter the YouTube video link below:")

    yt_link = st.text_input("YouTube Link:")

    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button("Summarize"):
        st.session_state.summary = None
        st.session_state.messages = []
        if yt_link:
            with st.spinner("Generating summary... Please wait."):
                st.session_state.summary = asyncio.run(ai(yt_link))

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=google_api_key
    )

    memory = ConversationBufferMemory()

    conversation = ConversationChain(
        llm=llm,
        verbose=False,
        memory=memory
    )

    # Initialize conversation context
    if yt_link:
        transcript = asyncio.run(fetch_and_format_transcript(yt_link))
        memory.save_context(
            {"input": f"You are a helpful AI Assistant. You are given a Youtube video transcript. You must answer any questions, regardless if its answer is in the video or not. Following is the video Script {transcript}"},
            {"output": ""}
        )

    # Sidebar for chat messages
    with st.sidebar:
        st.title("Ask me any questions you have about the video")
        messages = st.container()
        
        for message in st.session_state.messages:
            messages.chat_message(message['role']).write(message['content'])

        if prompt := st.chat_input("Say something"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            messages.chat_message("user").write(prompt)
            
            with st.spinner("Generating..."):
                response = conversation.predict(input=prompt)

            st.session_state.messages.append({"role": "assistant", "content": response})
            messages.chat_message("assistant").write(response)

    if st.session_state.summary:
        st.subheader("Video Summary:")
        st.write(st.session_state.summary)

if __name__ == "__main__":
    main()
