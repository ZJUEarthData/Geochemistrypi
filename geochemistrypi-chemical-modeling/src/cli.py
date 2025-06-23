from typing import Tuple
import typer
from .fsolve import solve_equations as fsolve_solve
from .root import equations, jacobi_matrix
from scipy.optimize import root

app = typer.Typer(help="A command line tool to compare fsolve and root methods for solving equations.")

@app.command()
def fsolve(
    phi_ref: float = typer.Option(0.5, "--phi-ref", "-p", help="Initial guess for φ_ref"),
    beta_sple: float = typer.Option(0.5, "--beta-sple", "-b", help="Initial guess for β_sple"),
    beta_mix: float = typer.Option(2.0, "--beta-mix", "-m", help="Initial guess for β_mix")
) -> None:
    """
    Solve the system of equations using scipy.optimize.fsolve method.
    
    Args:
        phi_ref (float): Initial guess for φ_ref parameter
        beta_sple (float): Initial guess for β_sple parameter
        beta_mix (float): Initial guess for β_mix parameter
    """
    initial_guess = (phi_ref, beta_sple, beta_mix)
    solution = fsolve_solve(initial_guess)
    typer.echo(f"fsolve Solution: {solution}")


@app.command()
def root_method(
    phi_ref: float = typer.Option(0.5, "--phi-ref", "-p", help="Initial guess for φ_ref"),
    beta_sple: float = typer.Option(0.5, "--beta-sple", "-b", help="Initial guess for β_sple"),
    beta_mix: float = typer.Option(2.0, "--beta-mix", "-m", help="Initial guess for β_mix")
) -> None:
    """
    Solve the system of equations using scipy.optimize.root method.
    
    Args:
        phi_ref (float): Initial guess for φ_ref parameter
        beta_sple (float): Initial guess for β_sple parameter
        beta_mix (float): Initial guess for β_mix parameter
    """
    initial_guess = (phi_ref, beta_sple, beta_mix)
    solution = root(equations, initial_guess, jac=jacobi_matrix, method='hybr')
    typer.echo(f"root Solution: {solution.x}")


@app.command()
def compare(
    phi_ref: float = typer.Option(0.5, "--phi-ref", "-p", help="Initial guess for φ_ref"),
    beta_sple: float = typer.Option(0.5, "--beta-sple", "-b", help="Initial guess for β_sple"),
    beta_mix: float = typer.Option(2.0, "--beta-mix", "-m", help="Initial guess for β_mix")
) -> None:
    """
    Compare solutions from both fsolve and root methods.
    
    Args:
        phi_ref (float): Initial guess for φ_ref parameter
        beta_sple (float): Initial guess for β_sple parameter
        beta_mix (float): Initial guess for β_mix parameter
    """
    initial_guess = (phi_ref, beta_sple, beta_mix)
    
    # Get solutions from both methods
    fsolve_solution = fsolve_solve(initial_guess)
    root_solution = root(equations, initial_guess, jac=jacobi_matrix, method='hybr')
    
    # Print results
    typer.echo("Comparison of solutions:")
    typer.echo(f"fsolve Solution: {fsolve_solution}")
    typer.echo(f"root Solution: {root_solution.x}")


if __name__ == "__main__":
    app()