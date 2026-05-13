"""MCP server for the tau2 airline task."""

import json
import os
import sys
import types
import uuid
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _ensure_tau2_path() -> Path:
    tau2_path = Path(_require_env("TAU2_BENCH_PATH"))
    if not tau2_path.exists():
        raise FileNotFoundError(f"tau2-bench not found at {tau2_path}")
    tau2_str = str(tau2_path / "src")
    if tau2_str not in sys.path:
        sys.path.insert(0, tau2_str)
    tau2_src = tau2_path / "src"
    package_paths = {
        "tau2": tau2_src / "tau2",
        "tau2.agent": tau2_src / "tau2" / "agent",
        "tau2.agent.base": tau2_src / "tau2" / "agent" / "base",
    }
    for package_name, package_path in package_paths.items():
        if package_name in sys.modules:
            continue
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package
    return tau2_path


_ensure_tau2_path()

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.tools import AirlineTools
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import OUT_OF_SCOPE, STOP, TRANSFER, UserState
from tau2.utils.utils import get_now


def _normalize_result(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_normalize_result(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_result(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_result(item) for key, item in value.items()}
    return value


def _serialize_message(message: AssistantMessage | UserMessage | ToolMessage) -> dict[str, Any]:
    if isinstance(message, AssistantMessage):
        kind = "assistant"
    elif isinstance(message, UserMessage):
        kind = "user"
    else:
        kind = "tool"
    return {"kind": kind, "data": message.model_dump(mode="json")}


class AirlineSession:
    def __init__(self) -> None:
        workspace = Path(_require_env("TAU2_AIRLINE_WORKSPACE"))
        self.state_dir = workspace / ".tau2_airline"
        self.db_path = self.state_dir / "state" / "airline_db.json"
        self.task_path = self.state_dir / "task.json"
        self.trajectory_path = self.state_dir / "trajectory.json"
        self.user_state_path = self.state_dir / "user_state.json"
        self.status_path = self.state_dir / "conversation_status.json"

        with open(self.task_path) as f:
            self.task = Task.model_validate(json.load(f))
        self.db = FlightDB.load(self.db_path)
        self.tools = AirlineTools(self.db)

        user_llm_args = os.environ.get("TAU2_USER_SIM_LLM_ARGS")
        llm_args = json.loads(user_llm_args) if user_llm_args else {}
        self.user = UserSimulator(
            llm=_require_env("TAU2_USER_SIM_MODEL"),
            instructions=str(self.task.user_scenario),
            llm_args=llm_args,
        )
        if self.user_state_path.exists():
            with open(self.user_state_path) as f:
                self.user_state = UserState.model_validate(json.load(f))
        else:
            self.user_state = self.user.get_init_state()

    def _load_status(self) -> dict[str, Any]:
        with open(self.status_path) as f:
            return json.load(f)

    def _save_status(self, status: dict[str, Any]) -> None:
        with open(self.status_path, "w") as f:
            json.dump(status, f, indent=2)

    def _load_trajectory(self) -> list[dict[str, Any]]:
        with open(self.trajectory_path) as f:
            return json.load(f)

    def _save_trajectory(self, trajectory: list[dict[str, Any]]) -> None:
        with open(self.trajectory_path, "w") as f:
            json.dump(trajectory, f, indent=2)

    def _append_messages(
        self, messages: list[AssistantMessage | UserMessage | ToolMessage]
    ) -> None:
        trajectory = self._load_trajectory()
        trajectory.extend(_serialize_message(message) for message in messages)
        self._save_trajectory(trajectory)

    def _save_user_state(self) -> None:
        with open(self.user_state_path, "w") as f:
            json.dump(self.user_state.model_dump(mode="json"), f, indent=2)

    def _save_db(self) -> None:
        self.db.dump(self.db_path)

    def _classify_stop(self, user_message: UserMessage) -> tuple[bool, str | None]:
        content = user_message.content or ""
        if STOP in content:
            return True, "stop"
        if TRANSFER in content:
            return True, "transfer"
        if OUT_OF_SCOPE in content:
            return True, "out_of_scope"
        return False, None

    def _user_payload(self, user_message: UserMessage) -> dict[str, Any]:
        done, stop_reason = self._classify_stop(user_message)
        content = user_message.content or ""
        if done:
            content = ""
        return {
            "user_message": content,
            "done": done,
            "stop_reason": stop_reason,
        }

    def start_conversation(self) -> dict[str, Any]:
        status = self._load_status()
        if status["started"]:
            raise ValueError("Conversation already started")

        status["started"] = True
        status["start_time"] = get_now()

        assistant_message = AssistantMessage(role="assistant", content="Hi! How can I help you today?")
        user_message, self.user_state = self.user.generate_next_message(
            assistant_message, self.user_state
        )
        done, stop_reason = self._classify_stop(user_message)
        status["done"] = done
        status["stop_reason"] = stop_reason
        if done:
            status["end_time"] = get_now()

        self._append_messages([assistant_message, user_message])
        self._save_user_state()
        self._save_status(status)
        return self._user_payload(user_message)

    def reply_to_user(self, message: str) -> dict[str, Any]:
        status = self._load_status()
        if not status["started"]:
            raise ValueError("Conversation has not started")
        if status["done"]:
            raise ValueError("Conversation already completed")

        assistant_message = AssistantMessage(role="assistant", content=message)
        user_message, self.user_state = self.user.generate_next_message(
            assistant_message, self.user_state
        )
        done, stop_reason = self._classify_stop(user_message)
        status["done"] = done
        status["stop_reason"] = stop_reason
        if done:
            status["end_time"] = get_now()

        self._append_messages([assistant_message, user_message])
        self._save_user_state()
        self._save_status(status)
        return self._user_payload(user_message)

    def _record_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        error: bool,
    ) -> None:
        tool_call_id = str(uuid.uuid4())
        assistant_message = AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=tool_call_id,
                    name=tool_name,
                    arguments=arguments,
                    requestor="assistant",
                )
            ],
        )
        content = str(result) if error else json.dumps(_normalize_result(result))
        tool_message = ToolMessage(
            id=tool_call_id,
            role="tool",
            content=content,
            requestor="assistant",
            error=error,
        )
        self._append_messages([assistant_message, tool_message])

    def run_airline_tool(self, tool_name: str, **arguments: Any) -> Any:
        tool = getattr(self.tools, tool_name)
        try:
            result = tool(**arguments)
            normalized = _normalize_result(result)
            self._record_tool_result(tool_name, arguments, normalized, error=False)
            if self.tools.tool_mutates_state(tool_name):
                self._save_db()
            return normalized
        except Exception as exc:
            payload = {"error": str(exc)}
            self._record_tool_result(
                tool_name, arguments, f"Error: {exc}", error=True
            )
            return payload


_SESSION: AirlineSession | None = None


def SESSION() -> AirlineSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = AirlineSession()
    return _SESSION


MCP = FastMCP("tau2_airline", instructions="Airline environment and user simulator for tau2 airline tasks.")


@MCP.tool
def start_conversation() -> dict[str, Any]:
    """Start the simulated customer conversation and receive the user's opening message."""
    return SESSION().start_conversation()


@MCP.tool
def reply_to_user(message: str) -> dict[str, Any]:
    """Send a message to the simulated customer and receive the next user response."""
    return SESSION().reply_to_user(message)


@MCP.tool
def book_reservation(
    user_id: str,
    origin: str,
    destination: str,
    flight_type: str,
    cabin: str,
    flights: list[dict[str, Any]],
    passengers: list[dict[str, Any]],
    payment_methods: list[dict[str, Any]],
    total_baggages: int,
    nonfree_baggages: int,
    insurance: str,
) -> Any:
    """Book a reservation in the airline system."""
    return SESSION().run_airline_tool(
        "book_reservation",
        user_id=user_id,
        origin=origin,
        destination=destination,
        flight_type=flight_type,
        cabin=cabin,
        flights=flights,
        passengers=passengers,
        payment_methods=payment_methods,
        total_baggages=total_baggages,
        nonfree_baggages=nonfree_baggages,
        insurance=insurance,
    )


@MCP.tool
def calculate(expression: str) -> Any:
    """Evaluate a math expression."""
    return SESSION().run_airline_tool("calculate", expression=expression)


@MCP.tool
def cancel_reservation(reservation_id: str) -> Any:
    """Cancel a reservation."""
    return SESSION().run_airline_tool("cancel_reservation", reservation_id=reservation_id)


@MCP.tool
def get_reservation_details(reservation_id: str) -> Any:
    """Fetch reservation details."""
    return SESSION().run_airline_tool(
        "get_reservation_details", reservation_id=reservation_id
    )


@MCP.tool
def get_user_details(user_id: str) -> Any:
    """Fetch user profile details."""
    return SESSION().run_airline_tool("get_user_details", user_id=user_id)


@MCP.tool
def list_all_airports() -> Any:
    """List all supported airports."""
    return SESSION().run_airline_tool("list_all_airports")


@MCP.tool
def search_direct_flight(origin: str, destination: str, date: str) -> Any:
    """Search direct flights for a date."""
    return SESSION().run_airline_tool(
        "search_direct_flight",
        origin=origin,
        destination=destination,
        date=date,
    )


@MCP.tool
def search_onestop_flight(origin: str, destination: str, date: str) -> Any:
    """Search one-stop flights for a date."""
    return SESSION().run_airline_tool(
        "search_onestop_flight",
        origin=origin,
        destination=destination,
        date=date,
    )


@MCP.tool
def send_certificate(user_id: str, amount: int) -> Any:
    """Issue a travel certificate to the user."""
    return SESSION().run_airline_tool("send_certificate", user_id=user_id, amount=amount)


@MCP.tool
def transfer_to_human_agents(summary: str) -> Any:
    """Transfer the user to a human agent."""
    return SESSION().run_airline_tool("transfer_to_human_agents", summary=summary)


@MCP.tool
def update_reservation_baggages(
    reservation_id: str,
    total_baggages: int,
    nonfree_baggages: int,
    payment_id: str,
) -> Any:
    """Update reservation baggage counts."""
    return SESSION().run_airline_tool(
        "update_reservation_baggages",
        reservation_id=reservation_id,
        total_baggages=total_baggages,
        nonfree_baggages=nonfree_baggages,
        payment_id=payment_id,
    )


@MCP.tool
def update_reservation_flights(
    reservation_id: str,
    cabin: str,
    flights: list[dict[str, Any]],
    payment_id: str,
) -> Any:
    """Update reservation flights and cabin."""
    return SESSION().run_airline_tool(
        "update_reservation_flights",
        reservation_id=reservation_id,
        cabin=cabin,
        flights=flights,
        payment_id=payment_id,
    )


@MCP.tool
def update_reservation_passengers(
    reservation_id: str,
    passengers: list[dict[str, Any]],
) -> Any:
    """Update passenger details for an existing reservation."""
    return SESSION().run_airline_tool(
        "update_reservation_passengers",
        reservation_id=reservation_id,
        passengers=passengers,
    )


@MCP.tool
def get_flight_status(flight_number: str, date: str) -> Any:
    """Get the status for a flight instance."""
    return SESSION().run_airline_tool(
        "get_flight_status",
        flight_number=flight_number,
        date=date,
    )


if __name__ == "__main__":
    MCP.run("stdio", show_banner=False)
