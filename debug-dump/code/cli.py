import typer
from triad.services.base import IStatusChecker
from triad.services.google_quota import GoogleQuotaChecker
from triad.models.status import Status
from triad.exceptions import TriadError

app = typer.Typer(
    name="triad",
    help="A simple status checking tool.",
    add_completion=False,
)

def _get_status_checker() -> IStatusChecker:
    """
    Dependency provider for the status checker service.
    """
    return GoogleQuotaChecker()

def _format_status(status: Status) -> str:
    """Formats the status object for display."""
    return f"{status.icon} {status.source} ({status.details}) now"

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    Checks and displays the status of a configured service.
    """
    if ctx.invoked_subcommand is not None:
        return

    try:
        status_checker = _get_status_checker()
        status = status_checker.get_status()
        formatted_status = _format_status(status)
        typer.echo(formatted_status)
    except TriadError as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1)

def run():
    """Script entry point for the 'triad' command."""
    app()