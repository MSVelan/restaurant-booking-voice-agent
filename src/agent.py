from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional

import logging
import requests
import os
import yaml

from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from enum import Enum
from pydantic import Field
from uuid import uuid4, UUID

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
    RunContext,
)
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation, silero, deepgram, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import dateparser

logger = logging.getLogger("restaurant-booking-agent")
logger.setLevel(logging.INFO)


load_dotenv()


class Status(str, Enum):
    CONFIRMED = "confirmed"
    PENDING = "pending"
    CANCELLED = "cancelled"


class SeatPreference(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


class Seating(BaseModel):
    """Contact information for a person."""

    seat_preference: SeatPreference = Field(
        default=SeatPreference.INDOOR,
        description="Whether to sit outside and enjoy the beauty of nature or \
            to sit inside because the weather is bad",
    )


# using pydantic class is better, but went with dataclass due to time constraints
@dataclass
class UserData:
    booking_id: str = field(default_factory=lambda: uuid4().hex)
    restaurant_city_loc: str = "Bangalore"
    restaurant_lat: str = "12.9767936"
    restaurant_long: str = "77.590082"

    customer_name: Optional[str] = None

    # Reservation agent
    number_of_guests: Optional[int] = None
    booking_date: Optional[datetime] = None
    booking_time_str: Optional[str] = None
    booking_time: Optional[timedelta] = None

    weather_info_obj: Optional[dict] = None
    # content to feed to llm to get seating preference
    weather_info: Optional[str] = None
    seating_preference: Optional[SeatPreference] = None

    cuisine_preference: Optional[str] = None
    special_requests: Optional[str] = None

    # Checkout agent stores data into db

    status: Optional[Status] = None
    checked_out: Optional[bool] = None
    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None

    def summarize(self) -> str:
        data = {
            "restaurant_city_loc": self.restaurant_city_loc,
            "customer_name": self.customer_name or "unknown",
            "number_of_guests": self.number_of_guests or "unknown",
            "booking_date": (
                self.booking_date.strftime("%d %M %Y")
                if self.booking_date
                else "unknown"
            ),
            "booking_time": self.booking_time_str or "unknown",
            "cuisine_preference": self.cuisine_preference or "unknown",
            "special_requests": self.special_requests or "unknown",
            "weather_info": self.weather_info or "unknown",
            "seating_preference": (
                self.seating_preference.value if self.seating_preference else "unknown"
            ),
            "status": self.status.value if self.status else "unknown",
            "checked_out": self.checked_out or False,
        }
        # summarize in yaml performs better than json
        return yaml.dump(data)


RunContext_T = RunContext[UserData]


@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_name = name
    return f"The name is updated to {name}"


@function_tool()
async def to_greeter(context: RunContext_T) -> Agent:
    """Called when user asks any unrelated questions or requests
    any other services not in your job description."""
    curr_agent: BaseAgent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)


class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"entering task {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # add the previous agent's chat history to the current agent
        if isinstance(userdata.prev_agent, Agent):
            truncated_chat_ctx = userdata.prev_agent.chat_ctx.copy(
                exclude_instructions=True, exclude_function_call=False
            ).truncate(max_items=6)
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [
                item for item in truncated_chat_ctx.items if item.id not in existing_ids
            ]
            chat_ctx.items.extend(items_copy)

        # add an instructions including the user data as assistant message
        chat_ctx.add_message(
            role="system",  # role=system works for OpenAI's LLM and Realtime API
            content=f"You are {agent_name} agent. Current user data is {userdata.summarize()}",
        )

        await self.update_chat_ctx(chat_ctx)
        await self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(
        self, name: str, context: RunContext_T
    ) -> tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent

        return next_agent, f"Transferring to {name}."


class Greeter(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly restaurant receptionist. "
                "The restaurant supports varies cuisines like Chinese, Mexican, Italian, Indian, etc.\n"
                "Your jobs are to greet the caller and guide them to the right agent using the given tools."
            ),
            # llm=groq.LLM(model="llama-3.3-70b-versatile", parallel_tool_calls=False),
            llm=openai.LLM.with_cerebras(
                model="llama3.1-8b", parallel_tool_calls=False
            ),
        )

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when user wants to make or update a reservation.
        This function handles transitioning to the reservation agent
        who will collect the necessary details like reservation date and time,
        customer name, number of guests and any special requests."""
        return await self._transfer_to_agent("reservation", context)

    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to proceed to checkout to confirm their reservation."""
        return await self._transfer_to_agent("checkout", context)


class Reservation(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a reservation agent at a restaurant. Your jobs are "
            "to ask for the reservation date and time, cuisine preferences, "
            "customer's name, number of guests and special requests(if any) step by step. "
            "Next, call the tool to check weather and predict the seat preference."
            "Confirm the reservation details with the customer "
            "and then call the tool to proceed to the checkout.",
            tools=[update_name, to_greeter],
            # llm=groq.LLM(model="llama-3.3-70b-versatile"),
            llm=openai.LLM.with_cerebras(model="llama-3.3-70b"),
        )

    def get_weather_conditions(self, userdata: UserData):
        lat = userdata.restaurant_lat
        lon = userdata.restaurant_long

        api_key = os.getenv("OPENWEATHER_API_KEY", "")
        params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}

        base_url = "https://api.openweathermap.org/data/2.5/forecast"
        weather_conditions = ""
        target_datetime = userdata.booking_date + userdata.booking_time

        try:
            response = requests.get(url=base_url, params=params)
            response.raise_for_status()
            data = response.json()
            forecasts = data["list"]
            closest_forecast = None
            min_diff = float("inf")
            for forecast in forecasts:
                forecast_time = datetime.fromtimestamp(forecast["dt"])
                time_diff = abs((forecast_time - target_datetime).total_seconds())

                if time_diff < min_diff:
                    min_diff = time_diff
                    closest_forecast = forecast
            if closest_forecast:
                userdata.weather_info_obj = closest_forecast
                forecast_dt = datetime.fromtimestamp(closest_forecast["dt"])
                temp_unit = "°C"
                speed_unit = "m/s"

                print(f"\n{'='*50}")
                print(f"Weather Forecast")
                print(f"Location: {lat}, {lon}")
                print(f"{'='*50}")
                print(f"Requested: {target_datetime.strftime('%Y-%m-%d %H:%M')}")
                print(f"Forecast:  {forecast_dt.strftime('%Y-%m-%d %H:%M')}")
                print(f"{'-'*50}")
                # Record the weather conditions
                weather_conditions += f"Condition: {closest_forecast['weather'][0]['description'].title()}\n"
                weather_conditions += (
                    f"Temperature: {closest_forecast['main']['temp']}{temp_unit}\n"
                )
                weather_conditions += (
                    f"Feels Like: {closest_forecast['main']['feels_like']}{temp_unit}\n"
                )
                weather_conditions += f"Min/Max: {closest_forecast['main']['temp_min']}/{closest_forecast['main']['temp_max']}{temp_unit}\n"
                weather_conditions += (
                    f"Humidity: {closest_forecast['main']['humidity']}%\n"
                )
                weather_conditions += (
                    f"Wind Speed: {closest_forecast['wind']['speed']} {speed_unit}\n"
                )
                weather_conditions += f"Clouds: {closest_forecast['clouds']['all']}%\n"
                if "rain" in closest_forecast:
                    weather_conditions += (
                        f"Rain (3h): {closest_forecast['rain'].get('3h', 0)} mm\n"
                    )
                if "snow" in closest_forecast:
                    weather_conditions += (
                        f"Snow (3h): {closest_forecast['snow'].get('3h', 0)} mm\n"
                    )
                logger.info(f"Weather conditions: {weather_conditions}")
                print(f"{'='*50}\n")

        except Exception as e:
            print(f"Exception occurred: {e}")
            logger.info("Weather data unavailable. Defaulting to indoor seating.")
            return None

        userdata.weather_info = weather_conditions
        return weather_conditions

    async def update_weather_info(self, weather_conditions: str, userdata: UserData):
        system_prompt = """
        You are a world class waiter in a restaurant with a great intuition of nature based on 
        weather conditions. You are supposed to output the seating preference that is best suited 
        for the given climatic conditions.
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{query}"),
            ]
        )

        llm = init_chat_model("groq:llama-3.3-70b-versatile")
        structured_llm = llm.with_structured_output(Seating)
        pipeline = prompt_template | structured_llm
        result = await pipeline.ainvoke({"query": weather_conditions})
        userdata.seating_preference = result.seat_preference
        logger.info(f"Seating preference updated to {userdata.seating_preference}")

    @function_tool()
    async def update_reservation_date(
        self,
        date_str: Annotated[str, Field(description="Date of reservation")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their reservation date.
        Confirm the date with the user before calling the function."""
        userdata = context.userdata
        booking_date = dateparser.parse(
            date_str,
            settings={
                "PREFER_DATES_FROM": "future",
                "TIMEZONE": "Asia/Kolkata",
                "TO_TIMEZONE": "Asia/Kolkata",
            },
        )
        if not booking_date:
            return "Could not understand the date provided. Clarify the date from the user."

        userdata.booking_date = booking_date
        return f"The reservation date is updated to {date_str}"

    @function_tool()
    async def update_reservation_time(
        self,
        booking_time_str: Annotated[str, Field(description="The reservation time")],
        context: RunContext_T,
    ):
        """Called when the user provides their reservation time.
        Confirm the time with the user before calling the function."""
        userdata = context.userdata
        parsed_dt = dateparser.parse(
            booking_time_str,
            settings={"TIMEZONE": "Asia/Kolkata", "TO_TIMEZONE": "Asia/Kolkata"},
        )
        if not parsed_dt:
            return "Could not understand the time provided. Clarify the time from the user."
        time_delta = timedelta(hours=parsed_dt.hour, minutes=parsed_dt.minute)
        userdata.booking_time_str = booking_time_str
        userdata.booking_time = time_delta
        return f"The reservation time is updated to {booking_time_str}"

    @function_tool()
    async def update_number_of_guests(
        self,
        number_of_guests: Annotated[
            int, Field(description="The number of guests for reservation")
        ],
        context: RunContext_T,
    ):
        """Called when the user provides the number of guests for reservation.
        Confirm the number of guests with the user before calling the function."""
        userdata = context.userdata
        userdata.number_of_guests = number_of_guests
        return f"The number of guests is updated to {number_of_guests}"

    @function_tool()
    async def update_cuisine_preference(
        self,
        cuisine_preferences: Annotated[
            list[str], Field(description="Cuisine preferences of the customer")
        ],
        context: RunContext_T,
    ) -> str:
        """Called when the user specifies or updates their cuisine preferences"""
        userdata = context.userdata
        userdata.cuisine_preference = ", ".join(cuisine_preferences)
        return f"The cuisine preferences is updated to {cuisine_preferences}"

    @function_tool()
    async def update_special_requests(
        self, special_requests: str, context: RunContext_T
    ):
        """Called when the user specifies any special requests. This could
        include requests related to dietary restrictions, special occasions, etc."""
        userdata = context.userdata
        userdata.special_requests = special_requests
        return f"Special Request is updated to {special_requests}"

    @function_tool()
    async def update_seat_preference(
        self,
        context: RunContext_T,
    ):
        """Called after the user has confirmed the reservation date and time."""
        userdata = context.userdata
        if not userdata.booking_date:
            return f"Please provide booking date first."
        if not userdata.booking_time:
            return f"Please provie booking time first."
        weather_conditions = self.get_weather_conditions(userdata)
        if weather_conditions:
            await self.update_weather_info(weather_conditions, userdata)
        return f"Seating preference updated to {userdata.seating_preference}"

    @function_tool()
    async def proceed_to_checkout(
        self, context: RunContext_T
    ) -> str | tuple[Agent, str]:
        """Called when the user wants to proceed to the checkout."""
        userdata = context.userdata
        if not userdata.customer_name:
            return "Please provide your name first."

        if not userdata.booking_date:
            return "Please provide booking date first."

        if not userdata.booking_time:
            return "Please provide booking time first."

        if not userdata.number_of_guests:
            return "Please provide number of guests first."

        if not userdata.cuisine_preference:
            return "Please provide cuisine preference first."

        if not userdata.special_requests:
            return "Please provide special requests if you would like."

        if not userdata.seating_preference:
            return "Please provide seating preference if you would like."

        return await self._transfer_to_agent("checkout", context)


class Checkout(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a checkout agent at a restaurant."
                "The restaurant provides cuisines of different types such as Indian, Italian, Chinese, Mexican, etc."
                "Confirm the user about the details and ask if the user wants to update the order (call appropriate tool for updating if required). "
                "After confirmation, proceed to storing details by calling the appropriate tool."
            ),
            tools=[update_name, to_greeter],
            llm=openai.LLM.with_cerebras(model="qwen-3-32b", parallel_tool_calls=False),
        )

    @function_tool()
    async def confirm_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the checkout."""
        userdata = context.userdata

        # send post request to http://localhost:3000/api/bookings with required payload
        headers = {"Content-Type": "application/json"}
        payload = {
            "bookingId": userdata.booking_id,
            "customerName": userdata.customer_name,
            "numberOfGuests": userdata.number_of_guests,
            "bookingDate": userdata.booking_date.isoformat(),
            "bookingTime": userdata.booking_time_str,
            "cuisinePreference": userdata.cuisine_preference,
            "specialRequests": userdata.special_requests,
            "weatherInfo": userdata.weather_info_obj,
            "seatingPreference": userdata.seating_preference.value,
            "status": userdata.status.value,
        }
        baseurl = "http://localhost:3000/api/bookings"

        try:
            await requests.post(url=baseurl, headers=headers, json=payload)
        except Exception as e:
            logger.error(f"Error occurred when storing the data: {e}")

        userdata.checked_out = True
        logger.info(userdata.summarize())
        return await to_greeter(context)

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to update their order."""
        return await self._transfer_to_agent("reservation", context)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(),
            "reservation": Reservation(),
            "checkout": Checkout(),
        }
    )
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        userdata=userdata,
        # stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        # stt=deepgram.STTv2(
        #     model="nova-2",
        #     # eager_eot_threshold=0.4,
        # ),
        stt=deepgram.STTv2(
            model="flux-general-en",
            eager_eot_threshold=0.4,
        ),
        # llm=inference.LLM(model="openai/gpt-4.1-mini"),
        llm=openai.LLM.with_cerebras(
            model="llama-3.3-70b",
        ),
        # tts=inference.TTS(
        #     model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        # ),
        tts=deepgram.TTS(
            model="aura-asteria-en",
        ),
        max_tool_steps=6,
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    @session.on("user_started_speaking")
    def on_user_started_speaking():
        print(" [DEBUG] User started speaking (VAD triggered)")

    @session.on("user_speech_committed")
    def on_user_speech_committed(msg):
        print(f" [DEBUG] User speech committed (STT Result): {msg.content}")

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
