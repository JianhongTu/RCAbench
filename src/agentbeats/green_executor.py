from abc import abstractmethod
from pydantic import ValidationError

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidParamsError,
    Task,
    TaskState,
    UnsupportedOperationError,
    InternalError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

import logging
logger = logging.getLogger(__name__)

from agentbeats.models import EvalRequest


class GreenAgent:

    @abstractmethod
    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        pass

    @abstractmethod
    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        pass


class GreenExecutor(AgentExecutor):

    def __init__(self, green_agent: GreenAgent):
        self.agent = green_agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        request_text = context.get_user_input()
        try:
            req: EvalRequest = EvalRequest.model_validate_json(request_text)
            ok, msg = self.agent.validate_request(req)
            if not ok:
                raise ServerError(error=InvalidParamsError(message=msg))
        except ValidationError as e:
            raise ServerError(error=InvalidParamsError(message=e.json()))

        msg = context.message
        if msg:
            task = new_task(msg)
            await event_queue.enqueue_event(task)
        else:
            raise ServerError(error=InvalidParamsError(message="Missing message."))

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting assessment.\n{req.model_dump_json()}", context_id=context.context_id)
        )

        try:
            await self.agent.run_eval(req, updater)
            try:
                await updater.complete()
            except RuntimeError as e:
                if "terminal state" in str(e):
                    logger.debug(f"Task {updater.task_id} already in terminal state, skipping completion.")
                else:
                    raise
        except Exception as e:
            logger.error(f"Agent error in execution: {e}", exc_info=True)
            try:
                await updater.failed(new_agent_text_message(f"Agent error: {e}", context_id=context.context_id))
            except RuntimeError as re:
                if "terminal state" in str(re):
                    logger.debug(f"Task {updater.task_id} already in terminal state, skipping failure notice.")
                else:
                    logger.error(f"Failed to mark task as failed: {re}")
            raise ServerError(error=InternalError(message=str(e)))

    async def _is_task_terminal(self, updater: TaskUpdater) -> bool:
        """Check if a task is already in a terminal state."""
        try:
            # This is a bit of a hack since TaskUpdater doesn't expose state easily
            # But we can try to get the task from the store if we had access to it.
            # For now, we'll just catch the RuntimeError in the updater itself if we can,
            # or trust that our agents are now better behaved.
            # But let's add a safer check if possible.
            return False 
        except:
            return False

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
