from __future__ import annotations

from verl.experimental.star_ppo.workflows.base import WorkflowRunner


class BuiltinWorkflowRunner(WorkflowRunner):
    """Backward-compatible built-in workflow dispatcher."""

    async def run_batch(self, batch, epoch):
        workflow_name = str(self.config.star.get("workflow", {}).get("name", "single_agent"))
        if workflow_name == "single_agent":
            return await self.trainer._run_single_agent_workflow(batch, epoch)
        raise ValueError(
            f"Unknown built-in workflow name: {workflow_name}. "
            "For complex workflows, configure star.workflow.runner.path/name."
        )
