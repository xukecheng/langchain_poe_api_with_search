import os
import re
import asyncio
import logging
import requests

import jieba
import tiktoken
import azure.functions as func
from typing import List, Type, Any, Dict, AsyncIterator, Literal, Union, cast
from langdetect import detect
from fastapi_poe import PoeBot, make_app, run
from fastapi_poe.types import ProtocolMessage
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import LLMResult
from langchain.callbacks.base import AsyncCallbackHandler
from googleapiclient.discovery import build


# Define constants or configurations
BROWSERLESS_URL = os.environ.get("BROWSERLESS_URL")
BROWSERLESS_TOKEN = os.environ.get("BROWSERLESS_TOKEN")

GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

MODE = "gpt-3.5-turbo"

API_KEY = os.environ.get("API_KEY")


# Define Pydantic models
class BrowseUrlInput(BaseModel):
    """Inputs for browse_url"""

    url: str = Field(description="Url to browse")


class GoogleSearchInput(BaseModel):
    """Inputs for google_search_snippets"""

    query: str = Field(description="String to search")


# Define LangChain tool
class BrowseUrlTool(BaseTool):
    name = "browse_url"
    description = "Useful when you want to scrape a website for data"
    args_schema: Type[BaseModel] = BrowseUrlInput

    def _run(self, url: str):
        response = self._browse_and_cutoff_url(url)
        return response

    async def _arun(self, url: str):
        response = await asyncio.to_thread(self._browse_and_cutoff_url, url)
        return response

    def _browse_and_cutoff_url(self, url: str) -> str:
        response = self._fetch_url_response(url)
        text = response.json()["data"][0]["results"][0]["text"].replace("\n", "")
        return self._cutoff_text_by_words(text, 1000)

    def _fetch_url_response(self, url: str):
        return requests.post(
            f"{BROWSERLESS_URL}/scrape?token={BROWSERLESS_TOKEN}",
            json={
                "url": url,
                "elements": [{"selector": "body"}],
            },
        )

    def _cutoff_text_by_words(self, text: str, cutoff_words: int) -> str:
        clean_text = re.sub(r"<.*?>|\n", "", text)
        detected_lang = detect(clean_text)
        if detected_lang == "zh-cn":
            words = jieba.lcut(clean_text)
        else:
            words = re.findall(r"\b\w+\b", clean_text)
        words = [word for word in words if word.strip()]
        cutoff_array = words[:cutoff_words]
        while self._count_tokens(" ".join(cutoff_array)) > 2000:
            cutoff_array = cutoff_array[:-5]
        return " ".join(cutoff_array)

    def _count_tokens(self, s):
        enc = tiktoken.encoding_for_model(MODE)
        return len(enc.encode(s))


class GoogleSearchTool(BaseTool):
    name = "google_search_snippets"
    description = "Search Google for recent results."
    args_schema: Type[BaseModel] = GoogleSearchInput

    def _get_metadata_result(self, result):
        metadata_result = {
            "title": result["title"],
            "link": result["link"],
        }
        if "snippet" in result:
            metadata_result["snippet"] = result["snippet"]
        return metadata_result

    def _search(self, query):
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        request = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=3)
        response = request.execute()
        return response.get("items", [])

    def _run(self, query: str):
        results = self._search(query)
        if not results:
            return [{"Result": "No good Google Search Result was found"}]
        metadata_results = [self._get_metadata_result(result) for result in results]
        return metadata_results

    async def _arun(self, query: str):
        results = await asyncio.to_thread(self._search, query)
        if not results:
            return [{"Result": "No good Google Search Result was found"}]
        metadata_results = [self._get_metadata_result(result) for result in results]
        return metadata_results


# Define LangChain tools
tools = [
    GoogleSearchTool(),
    BrowseUrlTool(),
]


# Define function to convert Poe messages to ConversationBufferMemory
def convert_poe_messages(
    poe_messages: List[ProtocolMessage],
) -> ConversationBufferMemory:
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    for poe_message in poe_messages:
        if poe_message.role == "user":
            memory.chat_memory.add_user_message(poe_message.content)
            # TODO: add attachments logic
            # if len(poe_message.attachments) != 0:
            #     for attachment in poe_message.attachments:
            #         url = attachment.url
            #         content_type = attachment.content_type
            #         name = attachment.name
        elif poe_message.role == "assistant":
            memory.chat_memory.add_ai_message(poe_message.content)
        else:
            memory.chat_memory.add_user_message(poe_message.content)
    return memory


class AsyncIteratorCallbackHandler(AsyncCallbackHandler):
    """Callback handler that returns an async iterator."""

    queue: asyncio.Queue[str]

    done: asyncio.Event

    @property
    def always_verbose(self) -> bool:
        return True

    def __init__(self) -> None:
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # If two calls are made in a row, this resets the state
        self.done.clear()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token is not None and token != "":
            self.queue.put_nowait(token)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print(response)
        if (
            response.generations[0][0].text is not None
            and response.generations[0][0].text != ""
        ):
            self.done.set()

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self.done.set()

    # TODO implement the other methods

    async def aiter(self) -> AsyncIterator[str]:
        while not self.queue.empty() or not self.done.is_set():
            # Wait for the next token in the queue,
            # but stop waiting if the done event is set
            done, other = await asyncio.wait(
                [
                    # NOTE: If you add other tasks here, update the code below,
                    # which assumes each set has exactly one task each
                    asyncio.ensure_future(self.queue.get()),
                    asyncio.ensure_future(self.done.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the other task
            if other:
                other.pop().cancel()

            # Extract the value of the first completed task
            token_or_done = cast(Union[str, Literal[True]], done.pop().result())

            # If the extracted value is the boolean True, the done event was set
            if token_or_done is True:
                break

            # Otherwise, the extracted value is a token, which we yield
            yield token_or_done


# Define EchoBot
class EchoBot(PoeBot):
    async def get_response(self, query):
        callback_handler = AsyncIteratorCallbackHandler()
        # callback_manager = AsyncCallbackManager(
        #     [callback_handler, StreamingStdOutCallbackHandler()]
        # )

        mode = ChatOpenAI(
            temperature=0,
            model=MODE,
            streaming=True,
            callbacks=[callback_handler],
        )

        memory = convert_poe_messages(query.query)

        agent = initialize_agent(
            tools,
            llm=mode,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs={
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            },
            memory=memory,
        )

        logging.info(query.query)

        last_message = query.query[-1].content
        run = asyncio.create_task(agent.arun(last_message))

        # Yield the tokens as they come in
        async for token in callback_handler.aiter():
            yield self.text_event(token)
        # Await the chain run
        await run

        # yield self.text_event(agent.run(last_message))


# Run the EchoBot
app = make_app(EchoBot(), api_key=API_KEY)

# run(EchoBot(), api_key=API_KEY)
