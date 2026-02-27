"""
User Guides Generator for Project HyperTensor.

This module provides tools for creating and generating user guides,
tutorials, and documentation content for the HyperTensor framework.

Features:
    - Structured guide sections
    - Code examples with validation
    - Markdown and RST output
    - Template-based generation
    - Interactive tutorial generation
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto


class DifficultyLevel(Enum):
    """Tutorial difficulty level."""

    BEGINNER = auto()
    INTERMEDIATE = auto()
    ADVANCED = auto()
    EXPERT = auto()


class GuideType(Enum):
    """Type of user guide."""

    QUICKSTART = auto()
    TUTORIAL = auto()
    HOWTO = auto()
    REFERENCE = auto()
    EXPLANATION = auto()


@dataclass
class CodeExample:
    """A runnable code example.

    Attributes:
        code: The Python code to execute.
        description: What this example demonstrates.
        expected_output: Expected output (for validation).
        language: Programming language (default: python).
        requires: List of required imports/dependencies.
        title: Optional title for the example.
    """

    code: str
    description: str = ""
    expected_output: str | None = None
    language: str = "python"
    requires: list[str] = field(default_factory=list)
    title: str | None = None

    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = []

        if self.title:
            lines.append(f"### {self.title}")
            lines.append("")

        if self.description:
            lines.append(self.description)
            lines.append("")

        lines.append(f"```{self.language}")
        lines.append(self.code.strip())
        lines.append("```")

        if self.expected_output:
            lines.append("")
            lines.append("**Output:**")
            lines.append("```")
            lines.append(self.expected_output.strip())
            lines.append("```")

        return "\n".join(lines)

    def to_rst(self) -> str:
        """Convert to reStructuredText format."""
        lines = []

        if self.title:
            lines.append(self.title)
            lines.append("~" * len(self.title))
            lines.append("")

        if self.description:
            lines.append(self.description)
            lines.append("")

        lines.append(f".. code-block:: {self.language}")
        lines.append("")
        for line in self.code.strip().split("\n"):
            lines.append(f"    {line}")

        if self.expected_output:
            lines.append("")
            lines.append("**Output:**")
            lines.append("")
            lines.append(".. code-block:: none")
            lines.append("")
            for line in self.expected_output.strip().split("\n"):
                lines.append(f"    {line}")

        return "\n".join(lines)


@dataclass
class GuideSection:
    """A section within a user guide.

    Attributes:
        title: Section title.
        content: Section text content.
        level: Heading level (1-6).
        examples: List of code examples.
        subsections: Nested subsections.
        notes: List of note/tip/warning blocks.
        images: List of image paths.
    """

    title: str
    content: str = ""
    level: int = 2
    examples: list[CodeExample] = field(default_factory=list)
    subsections: list[GuideSection] = field(default_factory=list)
    notes: list[dict[str, str]] = field(default_factory=list)
    images: list[dict[str, str]] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = []

        # Title
        prefix = "#" * self.level
        lines.append(f"{prefix} {self.title}")
        lines.append("")

        # Content
        if self.content:
            lines.append(self.content)
            lines.append("")

        # Notes
        for note in self.notes:
            note_type = note.get("type", "note").upper()
            note_content = note.get("content", "")
            lines.append(f"> **{note_type}:** {note_content}")
            lines.append("")

        # Images
        for image in self.images:
            alt = image.get("alt", "Image")
            path = image.get("path", "")
            caption = image.get("caption", "")
            lines.append(f"![{alt}]({path})")
            if caption:
                lines.append(f"*{caption}*")
            lines.append("")

        # Examples
        for example in self.examples:
            lines.append(example.to_markdown())
            lines.append("")

        # Subsections
        for subsection in self.subsections:
            lines.append(subsection.to_markdown())
            lines.append("")

        return "\n".join(lines)

    def to_rst(self) -> str:
        """Convert to reStructuredText format."""
        underlines = ["=", "-", "~", "^", '"', "'"]
        underline_char = underlines[min(self.level - 1, len(underlines) - 1)]

        lines = []

        # Title
        lines.append(self.title)
        lines.append(underline_char * len(self.title))
        lines.append("")

        # Content
        if self.content:
            lines.append(self.content)
            lines.append("")

        # Notes
        for note in self.notes:
            note_type = note.get("type", "note")
            note_content = note.get("content", "")
            lines.append(f".. {note_type}::")
            lines.append("")
            for line in note_content.split("\n"):
                lines.append(f"    {line}")
            lines.append("")

        # Images
        for image in self.images:
            path = image.get("path", "")
            caption = image.get("caption", "")
            lines.append(f".. image:: {path}")
            if caption:
                lines.append(f"    :alt: {caption}")
            lines.append("")

        # Examples
        for example in self.examples:
            lines.append(example.to_rst())
            lines.append("")

        # Subsections
        for subsection in self.subsections:
            lines.append(subsection.to_rst())
            lines.append("")

        return "\n".join(lines)


@dataclass
class Tutorial:
    """A complete tutorial or user guide.

    Attributes:
        title: Tutorial title.
        description: Brief description/abstract.
        difficulty: Difficulty level.
        guide_type: Type of guide.
        prerequisites: List of prerequisites.
        objectives: Learning objectives.
        sections: List of guide sections.
        authors: List of author names.
        version: Documentation version.
        estimated_time: Estimated completion time.
    """

    title: str
    description: str = ""
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    guide_type: GuideType = GuideType.TUTORIAL
    prerequisites: list[str] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    sections: list[GuideSection] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    estimated_time: str | None = None

    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = []

        # Title
        lines.append(f"# {self.title}")
        lines.append("")

        # Metadata
        lines.append(f"**Difficulty:** {self.difficulty.name.capitalize()}")
        if self.estimated_time:
            lines.append(f"**Estimated Time:** {self.estimated_time}")
        lines.append(f"**Version:** {self.version}")
        lines.append("")

        # Description
        if self.description:
            lines.append(self.description)
            lines.append("")

        # Prerequisites
        if self.prerequisites:
            lines.append("## Prerequisites")
            lines.append("")
            for prereq in self.prerequisites:
                lines.append(f"- {prereq}")
            lines.append("")

        # Objectives
        if self.objectives:
            lines.append("## Learning Objectives")
            lines.append("")
            for obj in self.objectives:
                lines.append(f"- {obj}")
            lines.append("")

        # Table of Contents
        if self.sections:
            lines.append("## Table of Contents")
            lines.append("")
            for i, section in enumerate(self.sections, 1):
                anchor = section.title.lower().replace(" ", "-")
                lines.append(f"{i}. [{section.title}](#{anchor})")
            lines.append("")

        # Sections
        for section in self.sections:
            lines.append(section.to_markdown())
            lines.append("")

        # Authors
        if self.authors:
            lines.append("---")
            lines.append("")
            lines.append(f"*Authors: {', '.join(self.authors)}*")

        return "\n".join(lines)

    def to_rst(self) -> str:
        """Convert to reStructuredText format."""
        lines = []

        # Title
        lines.append(self.title)
        lines.append("=" * len(self.title))
        lines.append("")

        # Metadata
        lines.append(f":Difficulty: {self.difficulty.name.capitalize()}")
        if self.estimated_time:
            lines.append(f":Estimated Time: {self.estimated_time}")
        lines.append(f":Version: {self.version}")
        lines.append("")

        # Description
        if self.description:
            lines.append(self.description)
            lines.append("")

        # TOC
        lines.append(".. contents:: Table of Contents")
        lines.append("    :depth: 2")
        lines.append("")

        # Sections
        for section in self.sections:
            lines.append(section.to_rst())
            lines.append("")

        return "\n".join(lines)


class GuideBuilder:
    """Builder class for constructing tutorials and guides.

    This class provides a fluent interface for building tutorials
    with sections, examples, and metadata.

    Example:
        >>> builder = GuideBuilder("My Tutorial")
        >>> builder.set_difficulty(DifficultyLevel.BEGINNER)
        >>> builder.add_section("Introduction", "Welcome...")
        >>> tutorial = builder.build()
    """

    def __init__(self, title: str):
        """Initialize the builder.

        Args:
            title: The tutorial title.
        """
        self.tutorial = Tutorial(title=title)
        self._current_section: GuideSection | None = None

    def set_description(self, description: str) -> GuideBuilder:
        """Set the tutorial description.

        Args:
            description: Brief description/abstract.

        Returns:
            Self for chaining.
        """
        self.tutorial.description = description
        return self

    def set_difficulty(self, difficulty: DifficultyLevel) -> GuideBuilder:
        """Set the difficulty level.

        Args:
            difficulty: Difficulty level.

        Returns:
            Self for chaining.
        """
        self.tutorial.difficulty = difficulty
        return self

    def set_guide_type(self, guide_type: GuideType) -> GuideBuilder:
        """Set the guide type.

        Args:
            guide_type: Type of guide.

        Returns:
            Self for chaining.
        """
        self.tutorial.guide_type = guide_type
        return self

    def set_estimated_time(self, time: str) -> GuideBuilder:
        """Set the estimated completion time.

        Args:
            time: Estimated time (e.g., "30 minutes").

        Returns:
            Self for chaining.
        """
        self.tutorial.estimated_time = time
        return self

    def add_prerequisite(self, prereq: str) -> GuideBuilder:
        """Add a prerequisite.

        Args:
            prereq: Prerequisite description.

        Returns:
            Self for chaining.
        """
        self.tutorial.prerequisites.append(prereq)
        return self

    def add_objective(self, objective: str) -> GuideBuilder:
        """Add a learning objective.

        Args:
            objective: Learning objective.

        Returns:
            Self for chaining.
        """
        self.tutorial.objectives.append(objective)
        return self

    def add_author(self, author: str) -> GuideBuilder:
        """Add an author.

        Args:
            author: Author name.

        Returns:
            Self for chaining.
        """
        self.tutorial.authors.append(author)
        return self

    def add_section(
        self,
        title: str,
        content: str = "",
        level: int = 2,
    ) -> GuideBuilder:
        """Add a new section.

        Args:
            title: Section title.
            content: Section content.
            level: Heading level.

        Returns:
            Self for chaining.
        """
        section = GuideSection(title=title, content=content, level=level)
        self.tutorial.sections.append(section)
        self._current_section = section
        return self

    def add_subsection(
        self,
        title: str,
        content: str = "",
    ) -> GuideBuilder:
        """Add a subsection to the current section.

        Args:
            title: Subsection title.
            content: Subsection content.

        Returns:
            Self for chaining.
        """
        if self._current_section is None:
            raise ValueError("No current section. Add a section first.")

        subsection = GuideSection(
            title=title,
            content=content,
            level=self._current_section.level + 1,
        )
        self._current_section.subsections.append(subsection)
        return self

    def add_example(
        self,
        code: str,
        description: str = "",
        expected_output: str | None = None,
        title: str | None = None,
    ) -> GuideBuilder:
        """Add a code example to the current section.

        Args:
            code: The code to demonstrate.
            description: Description of the example.
            expected_output: Expected output.
            title: Optional title.

        Returns:
            Self for chaining.
        """
        if self._current_section is None:
            raise ValueError("No current section. Add a section first.")

        example = CodeExample(
            code=textwrap.dedent(code).strip(),
            description=description,
            expected_output=expected_output,
            title=title,
        )
        self._current_section.examples.append(example)
        return self

    def add_note(self, content: str, note_type: str = "note") -> GuideBuilder:
        """Add a note to the current section.

        Args:
            content: Note content.
            note_type: Type (note, tip, warning, danger).

        Returns:
            Self for chaining.
        """
        if self._current_section is None:
            raise ValueError("No current section. Add a section first.")

        self._current_section.notes.append(
            {
                "type": note_type,
                "content": content,
            }
        )
        return self

    def add_image(
        self,
        path: str,
        alt: str = "Image",
        caption: str = "",
    ) -> GuideBuilder:
        """Add an image to the current section.

        Args:
            path: Image file path.
            alt: Alt text.
            caption: Image caption.

        Returns:
            Self for chaining.
        """
        if self._current_section is None:
            raise ValueError("No current section. Add a section first.")

        self._current_section.images.append(
            {
                "path": path,
                "alt": alt,
                "caption": caption,
            }
        )
        return self

    def build(self) -> Tutorial:
        """Build and return the tutorial.

        Returns:
            The constructed Tutorial.
        """
        return self.tutorial


def create_getting_started() -> Tutorial:
    """Create the Getting Started guide for HyperTensor.

    Returns:
        Complete Getting Started tutorial.
    """
    builder = GuideBuilder("Getting Started with HyperTensor")

    builder.set_description(
        "This guide will help you get up and running with Project HyperTensor, "
        "a quantum-inspired tensor network framework for hypersonic CFD."
    )
    builder.set_difficulty(DifficultyLevel.BEGINNER)
    builder.set_guide_type(GuideType.QUICKSTART)
    builder.set_estimated_time("30 minutes")

    builder.add_prerequisite("Python 3.9 or higher")
    builder.add_prerequisite("Basic knowledge of linear algebra")
    builder.add_prerequisite("Familiarity with NumPy/PyTorch")

    builder.add_objective("Install HyperTensor and verify the installation")
    builder.add_objective("Create your first tensor network simulation")
    builder.add_objective("Run a simple CFD benchmark")
    builder.add_objective("Understand the core concepts")

    # Installation section
    builder.add_section(
        "Installation",
        """
HyperTensor can be installed via pip or from source. We recommend using a virtual 
environment to avoid dependency conflicts.
""",
    )

    builder.add_example(
        """
# Create and activate a virtual environment
python -m venv hypertensor-env
source hypertensor-env/bin/activate  # Linux/Mac
# or: hypertensor-env\\Scripts\\activate  # Windows

# Install HyperTensor
pip install tensornet
""",
        description="Installing via pip:",
    )

    builder.add_example(
        """
# Clone the repository
git clone https://github.com/tigantic/HyperTensor.git
cd HyperTensor

# Install in development mode
pip install -e .
""",
        description="Installing from source:",
    )

    builder.add_note(
        "For GPU acceleration, ensure you have CUDA installed and use "
        "`pip install tensornet[gpu]`.",
        "tip",
    )

    # Quick start section
    builder.add_section(
        "Quick Start",
        """
Let's run a simple example to verify everything is working correctly.
""",
    )

    builder.add_example(
        """
import torch
from tensornet import MPS, heisenberg_mpo, run_dmrg

# Create a Heisenberg spin chain Hamiltonian
L = 10  # 10 sites
H = heisenberg_mpo(L, Jx=1.0, Jy=1.0, Jz=1.0)

# Run DMRG to find the ground state
result = run_dmrg(H, chi_max=32)

print(f"Ground state energy: {result.energy:.6f}")
print(f"Energy per site: {result.energy/L:.6f}")
""",
        description="Finding the ground state of a quantum spin chain:",
        expected_output="Ground state energy: -4.258035\\nEnergy per site: -0.425804",
    )

    # Core concepts section
    builder.add_section(
        "Core Concepts",
        """
HyperTensor is built on tensor network methods, which provide efficient 
representations of high-dimensional data. Here are the key concepts:
""",
    )

    builder.add_subsection(
        "Matrix Product States (MPS)",
        """
An MPS is a factorized representation of a high-dimensional tensor as a 
chain of smaller tensors. This enables exponential compression of data 
that satisfies an "area law" of correlations.
""",
    )

    builder.add_subsection(
        "Matrix Product Operators (MPO)",
        """
An MPO extends the MPS concept to operators (matrices). Physical 
Hamiltonians like the Heisenberg model can be exactly represented as 
MPOs with small bond dimension.
""",
    )

    builder.add_subsection(
        "DMRG Algorithm",
        """
The Density Matrix Renormalization Group (DMRG) is a variational algorithm 
for finding ground states of 1D quantum systems. It iteratively optimizes 
each tensor in the MPS while keeping others fixed.
""",
    )

    # CFD section
    builder.add_section(
        "Your First CFD Simulation",
        """
HyperTensor extends tensor networks to computational fluid dynamics. 
Let's run the classic Sod shock tube benchmark.
""",
    )

    builder.add_example(
        """
from tensornet.cfd import Euler1D, sod_shock_tube_ic

# Create solver with 200 grid points
solver = Euler1D(nx=200, domain=(0.0, 1.0))

# Initialize with Sod shock tube conditions
state = sod_shock_tube_ic(solver.nx, solver.x, gamma=1.4)

# Run simulation
t_final = 0.2
state = solver.solve(state, t_final, cfl=0.5)

# Get primitive variables
rho, u, p = solver.to_primitive(state)
print(f"Max density: {rho.max():.3f}")
print(f"Max velocity: {u.max():.3f}")
""",
        description="Running the Sod shock tube benchmark:",
        expected_output="Max density: 1.000\\nMax velocity: 0.927",
    )

    # Next steps section
    builder.add_section(
        "Next Steps",
        """
Now that you have HyperTensor working, here are some suggested next steps:

1. **Explore the tutorials**: Check out our in-depth tutorials for tensor 
   networks, CFD, and advanced features.

2. **Read the API reference**: Familiarize yourself with the full API 
   documentation.

3. **Join the community**: Connect with other users on GitHub Discussions.

4. **Contribute**: We welcome contributions! See CONTRIBUTING.md for guidelines.
""",
    )

    builder.add_author("HyperTensor Team")

    return builder.build()


def create_cfd_tutorial() -> Tutorial:
    """Create the CFD tutorial for HyperTensor.

    Returns:
        Complete CFD tutorial.
    """
    builder = GuideBuilder("Computational Fluid Dynamics with HyperTensor")

    builder.set_description(
        "Learn how to use HyperTensor for hypersonic computational fluid dynamics, "
        "from basic Euler equations to reactive Navier-Stokes simulations."
    )
    builder.set_difficulty(DifficultyLevel.INTERMEDIATE)
    builder.set_guide_type(GuideType.TUTORIAL)
    builder.set_estimated_time("2 hours")

    builder.add_prerequisite("Completed the Getting Started guide")
    builder.add_prerequisite("Basic understanding of fluid dynamics")
    builder.add_prerequisite("Familiarity with conservation laws")

    builder.add_objective("Understand the Euler and Navier-Stokes equations")
    builder.add_objective("Set up 1D, 2D, and 3D CFD simulations")
    builder.add_objective("Apply proper boundary conditions")
    builder.add_objective("Validate simulations against analytical solutions")
    builder.add_objective("Use QTT compression for efficient simulations")

    # Introduction
    builder.add_section(
        "Introduction",
        """
Computational Fluid Dynamics (CFD) solves the governing equations of fluid 
motion numerically. For hypersonic flows (Mach > 5), we encounter unique 
challenges including strong shocks, high temperatures, and chemical reactions.

HyperTensor uses tensor network compression to efficiently represent flow 
fields, exploiting the "area law" scaling of correlations in turbulent flows.
""",
    )

    # Euler equations section
    builder.add_section(
        "1D Euler Equations",
        """
The Euler equations describe inviscid, compressible flow. In 1D, the 
conservation form is:

$$\\frac{\\partial U}{\\partial t} + \\frac{\\partial F}{\\partial x} = 0$$

where $U = [\\rho, \\rho u, E]^T$ are conserved variables and 
$F = [\\rho u, \\rho u^2 + p, u(E + p)]^T$ are the fluxes.
""",
    )

    builder.add_example(
        """
from tensornet.cfd import Euler1D, EulerState
import torch

# Create 1D Euler solver
solver = Euler1D(
    nx=400,               # Grid points
    domain=(0.0, 1.0),    # Domain [0, 1]
    flux_scheme='hllc',   # HLLC Riemann solver
    limiter='van_leer',   # Van Leer TVD limiter
)

# Create initial condition (shock tube)
rho = torch.where(solver.x < 0.5, torch.ones_like(solver.x), 0.125 * torch.ones_like(solver.x))
u = torch.zeros_like(solver.x)
p = torch.where(solver.x < 0.5, torch.ones_like(solver.x), 0.1 * torch.ones_like(solver.x))

state = EulerState.from_primitive(rho, u, p, gamma=1.4)

# Evolve to t = 0.2
final_state = solver.solve(state, t_final=0.2, cfl=0.8)
""",
        description="Setting up a 1D Euler simulation:",
    )

    # 2D Euler section
    builder.add_section(
        "2D Euler Equations (Hypersonic Wedge)",
        """
For 2D flows, we use Strang splitting to decompose the problem into 
alternating 1D sweeps. This enables efficient computation of oblique 
shocks and complex flow patterns.
""",
    )

    builder.add_example(
        """
from tensornet.cfd import Euler2D, supersonic_wedge_ic
from tensornet.cfd.geometry import WedgeGeometry

# Create 2D solver
solver = Euler2D(
    nx=200, ny=100,
    domain=((0.0, 2.0), (0.0, 1.0)),
    flux_scheme='hllc',
)

# Define wedge geometry (15 degree half-angle)
wedge = WedgeGeometry(
    theta=15.0,  # degrees
    x_start=0.5,
    length=1.0,
)

# Initialize Mach 5 flow
state = supersonic_wedge_ic(
    solver, 
    mach=5.0, 
    gamma=1.4,
    wedge=wedge,
)

# Run simulation
final_state = solver.solve(state, t_final=1.0, cfl=0.5)

# Compute shock angle
beta = wedge.compute_shock_angle(mach=5.0, gamma=1.4)
print(f"Oblique shock angle: {beta:.2f} degrees")
""",
        description="Simulating Mach 5 flow over a wedge:",
    )

    builder.add_note(
        "The oblique shock angle can be compared to the analytical θ-β-M "
        "relationship for validation.",
        "tip",
    )

    # Navier-Stokes section
    builder.add_section(
        "Navier-Stokes with Viscous Effects",
        """
For high-fidelity simulations, we include viscous effects through the 
Navier-Stokes equations. This adds diffusion terms for momentum and energy.
""",
    )

    builder.add_example(
        """
from tensornet.cfd import NavierStokes2D, NavierStokes2DConfig
from tensornet.cfd.viscous import sutherland_viscosity

# Configure NS solver
config = NavierStokes2DConfig(
    nx=300, ny=150,
    domain=((0.0, 3.0), (0.0, 0.5)),
    Re=1e5,              # Reynolds number
    Pr=0.72,             # Prandtl number
    use_sutherland=True, # Temperature-dependent viscosity
)

solver = NavierStokes2D(config)

# Initialize flat plate boundary layer
state = solver.flat_plate_ic(mach=2.0, T_wall=300.0)

# Run with implicit time stepping for stability
final_state = solver.solve(
    state, 
    t_final=0.5, 
    cfl=2.0,  # Can use larger CFL with implicit
    implicit=True,
)
""",
        description="Setting up a viscous flat plate simulation:",
    )

    # QTT compression section
    builder.add_section(
        "QTT Compression",
        """
Quantized Tensor Train (QTT) compression enables logarithmic scaling of 
storage and computation. This is the key innovation of HyperTensor.
""",
    )

    builder.add_example(
        """
from tensornet.cfd.qtt import euler_to_qtt, qtt_to_euler, compression_analysis

# Compress Euler state to QTT format
qtt_state, info = euler_to_qtt(
    final_state,
    chi_max=64,        # Maximum bond dimension
    normalize=False,   # Preserve field amplitudes
)

print(f"Compression ratio: {info.compression_ratio:.2f}x")
print(f"Relative error: {info.relative_error:.2e}")

# Reconstruct for analysis
reconstructed = qtt_to_euler(qtt_state, solver.nx, solver.ny)

# Analyze compression vs accuracy trade-off
analysis = compression_analysis(final_state, chi_values=[16, 32, 64, 128])
""",
        description="Compressing flow fields with QTT:",
    )

    builder.add_author("HyperTensor Team")

    return builder.build()


def create_tensor_network_primer() -> Tutorial:
    """Create the Tensor Network primer for HyperTensor.

    Returns:
        Complete Tensor Network primer tutorial.
    """
    builder = GuideBuilder("Tensor Networks: A Primer")

    builder.set_description(
        "An introduction to tensor networks for scientists and engineers, "
        "covering MPS, MPO, DMRG, TEBD, and their application to physics."
    )
    builder.set_difficulty(DifficultyLevel.INTERMEDIATE)
    builder.set_guide_type(GuideType.EXPLANATION)
    builder.set_estimated_time("3 hours")

    builder.add_prerequisite("Linear algebra fundamentals")
    builder.add_prerequisite("Basic quantum mechanics (helpful but not required)")
    builder.add_prerequisite("Python programming experience")

    builder.add_objective("Understand tensor network notation and diagrams")
    builder.add_objective("Implement MPS operations from scratch")
    builder.add_objective("Apply DMRG to find ground states")
    builder.add_objective("Use TEBD for time evolution")
    builder.add_objective("Recognize the area law and its implications")

    # What are tensors
    builder.add_section(
        "What are Tensors?",
        """
A **tensor** is a multi-dimensional array of numbers. The number of indices 
(dimensions) is called the **order** or **rank** of the tensor:

- Order 0: Scalar (single number)
- Order 1: Vector (1D array)
- Order 2: Matrix (2D array)  
- Order N: N-dimensional array

In quantum many-body physics, the state of L qubits is described by a 
tensor with $2^L$ components—exponentially large in system size!
""",
    )

    builder.add_example(
        """
import torch

# Scalar (0th order tensor)
scalar = torch.tensor(3.14)
print(f"Scalar shape: {scalar.shape}")

# Vector (1st order tensor)
vector = torch.randn(5)
print(f"Vector shape: {vector.shape}")

# Matrix (2nd order tensor)
matrix = torch.randn(3, 4)
print(f"Matrix shape: {matrix.shape}")

# 4th order tensor
tensor_4 = torch.randn(2, 3, 4, 5)
print(f"4th order tensor shape: {tensor_4.shape}")
""",
        description="Creating tensors of different orders:",
    )

    # Tensor contractions
    builder.add_section(
        "Tensor Contractions",
        """
**Contraction** is the fundamental operation on tensors—it generalizes 
matrix multiplication. When we contract two tensors, we sum over shared indices.

For matrices A_{ij} and B_{jk}, matrix multiplication is:
$$C_{ik} = \\sum_j A_{ij} B_{jk}$$

This contracts over index j. The result has indices i and k.
""",
    )

    builder.add_example(
        """
import torch

# Two matrices
A = torch.randn(3, 4)  # shape (3, 4)
B = torch.randn(4, 5)  # shape (4, 5)

# Matrix multiplication = contraction over middle index
C = A @ B  # shape (3, 5)

# Generalized contraction with einsum
C_einsum = torch.einsum('ij,jk->ik', A, B)
assert torch.allclose(C, C_einsum)

# More complex contraction
T1 = torch.randn(2, 3, 4)  # shape (2, 3, 4)
T2 = torch.randn(4, 5, 3)  # shape (4, 5, 3)

# Contract indices 1 (dim 3) of T1 with index 2 (dim 3) of T2
# and index 2 (dim 4) of T1 with index 0 (dim 4) of T2
result = torch.einsum('ijk,kli->jl', T1, T2)  # shape (3, 5)
""",
        description="Tensor contractions with einsum:",
    )

    # MPS section
    builder.add_section(
        "Matrix Product States (MPS)",
        """
A Matrix Product State represents a high-dimensional tensor as a product 
of smaller matrices (actually, 3rd-order tensors). For a state |ψ⟩ of L sites:

$$|\\psi\\rangle = \\sum_{s_1...s_L} A^{[1]}_{s_1} A^{[2]}_{s_2} \\cdots A^{[L]}_{s_L} |s_1...s_L\\rangle$$

Each $A^{[i]}_{s_i}$ is a $\\chi_{i-1} \\times \\chi_i$ matrix, where $\\chi$ 
is the **bond dimension**. The total storage is $O(L \\chi^2 d)$ instead 
of $O(d^L)$!
""",
    )

    builder.add_example(
        """
from tensornet import MPS

# Create a random MPS with 10 sites, local dimension 2, bond dimension 16
mps = MPS.random(L=10, d=2, chi=16)

print(f"Number of sites: {mps.L}")
print(f"Local dimension: {mps.d}")
print(f"Bond dimensions: {mps.bond_dims}")

# Compute the norm
norm = mps.norm()
print(f"Norm: {norm:.6f}")

# Compute entanglement entropy at the middle bond
S = mps.entanglement_entropy(site=5)
print(f"Entanglement entropy at bond 5: {S:.4f}")
""",
        description="Creating and inspecting an MPS:",
    )

    # DMRG section
    builder.add_section(
        "DMRG Algorithm",
        """
The **Density Matrix Renormalization Group** (DMRG) is a variational 
algorithm for finding ground states. It works by:

1. Sweeping left-to-right and right-to-left through the chain
2. At each bond, optimizing the local tensors while keeping others fixed
3. Using SVD to maintain the MPS structure and truncate small singular values

DMRG converges exponentially fast for gapped 1D systems and achieves 
machine-precision accuracy for many quantum models.
""",
    )

    builder.add_example(
        """
from tensornet import heisenberg_mpo, run_dmrg

# Create Heisenberg Hamiltonian
L = 20
H = heisenberg_mpo(L, Jx=1.0, Jy=1.0, Jz=1.0)

# Run DMRG
result = run_dmrg(
    H,
    chi_max=64,         # Maximum bond dimension
    n_sweeps=10,        # Number of sweeps
    cutoff=1e-10,       # SVD truncation threshold
    verbose=True,
)

print(f"Ground state energy: {result.energy:.10f}")
print(f"Final bond dimension: {max(result.mps.bond_dims)}")
print(f"Truncation error: {result.truncation_error:.2e}")
""",
        description="Finding the ground state with DMRG:",
    )

    # Area law section
    builder.add_section(
        "The Area Law",
        """
The **area law** states that for ground states of gapped local Hamiltonians, 
the entanglement entropy scales with the boundary area, not the volume:

$$S(A) \\sim |\\partial A|$$

This is why tensor networks work—they efficiently represent states with 
limited entanglement. The bond dimension χ needed to represent a state 
accurately is related to the entanglement by $\\chi \\sim e^S$.

For turbulent CFD, recent work suggests similar area-law scaling, making 
tensor network compression viable for fluid dynamics!
""",
    )

    builder.add_author("HyperTensor Team")

    return builder.build()


def create_deployment_guide() -> Tutorial:
    """Create the Deployment guide for HyperTensor.

    Returns:
        Complete Deployment tutorial.
    """
    builder = GuideBuilder("Deploying HyperTensor to Embedded Systems")

    builder.set_description(
        "Learn how to deploy HyperTensor models to embedded systems like "
        "NVIDIA Jetson for real-time hypersonic vehicle guidance."
    )
    builder.set_difficulty(DifficultyLevel.ADVANCED)
    builder.set_guide_type(GuideType.HOWTO)
    builder.set_estimated_time("2 hours")

    builder.add_prerequisite("Completed CFD and Tensor Network tutorials")
    builder.add_prerequisite("Access to NVIDIA Jetson device or similar")
    builder.add_prerequisite("Basic understanding of ONNX and TensorRT")

    builder.add_objective("Export models to ONNX format")
    builder.add_objective("Optimize for TensorRT inference")
    builder.add_objective("Deploy to Jetson AGX Orin")
    builder.add_objective("Achieve real-time performance targets")

    # Export section
    builder.add_section(
        "Exporting to ONNX",
        """
The first step in deployment is exporting your trained model or surrogate 
to ONNX (Open Neural Network Exchange) format. This provides a portable 
representation that can be optimized for various hardware.
""",
    )

    builder.add_example(
        """
from tensornet.infra.deployment import (
    TensorRTExporter,
    ExportConfig,
    Precision,
    OptimizationLevel,
)

# Configure export
config = ExportConfig(
    precision=Precision.FP16,           # Use FP16 for speed
    optimization_level=OptimizationLevel.O2,
    input_shape=(1, 4, 64, 64),         # Batch, channels, H, W
    dynamic_batch=True,
)

# Create exporter
exporter = TensorRTExporter(config)

# Export surrogate model
result = exporter.export_to_onnx(
    model=surrogate_model,
    output_path="surrogate.onnx",
    input_names=["state"],
    output_names=["prediction"],
)

print(f"Exported to: {result.path}")
print(f"Model size: {result.size_mb:.2f} MB")
""",
        description="Exporting a surrogate model to ONNX:",
    )

    # TensorRT section
    builder.add_section(
        "Optimizing with TensorRT",
        """
NVIDIA TensorRT provides high-performance inference optimization. It fuses 
layers, optimizes memory access patterns, and leverages Tensor Cores for 
dramatic speedups.
""",
    )

    builder.add_example(
        """
# Build TensorRT engine from ONNX
engine_result = exporter.optimize_for_tensorrt(
    onnx_path="surrogate.onnx",
    output_path="surrogate.trt",
    workspace_mb=2048,      # GPU memory for optimization
    use_fp16=True,
    use_int8=False,         # Would need calibration data
)

# Validate against reference outputs
validation = exporter.validate_exported_model(
    reference_model=surrogate_model,
    trt_engine=engine_result.engine,
    test_inputs=test_data,
    tolerance=1e-3,
)

print(f"Max error: {validation.max_error:.2e}")
print(f"Passed: {validation.passed}")
""",
        description="Building a TensorRT engine:",
    )

    # Jetson section
    builder.add_section(
        "Deploying to Jetson",
        """
The NVIDIA Jetson AGX Orin is our target embedded platform for hypersonic 
vehicle guidance. It provides 275 TOPS of AI performance in a compact, 
power-efficient package.
""",
    )

    builder.add_example(
        """
from tensornet.infra.deployment import (
    EmbeddedRuntime,
    JetsonConfig,
    PowerMode,
    ThermalMonitor,
)

# Configure for Jetson AGX Orin
jetson_config = JetsonConfig(
    power_mode=PowerMode.MAXN,     # Maximum performance
    gpu_freq_mhz=1300,
    dla_enable=True,               # Use Deep Learning Accelerators
    memory_pool_mb=4096,
)

# Create runtime
runtime = EmbeddedRuntime(jetson_config)

# Load TensorRT engine
runtime.load_engine("surrogate.trt")

# Run inference with timing
with ThermalMonitor() as thermal:
    for i in range(1000):
        result, latency = runtime.infer_with_timing(input_tensor)
        
    avg_latency = sum(runtime.latencies) / len(runtime.latencies)
    print(f"Average latency: {avg_latency*1000:.2f} ms")
    print(f"Max temperature: {thermal.max_temp:.1f}°C")
""",
        description="Running inference on Jetson:",
    )

    builder.add_note(
        "For missile-compatible SWaP (Size, Weight, and Power), the 15W or 30W "
        "power modes may be more appropriate than MAXN.",
        "warning",
    )

    # Real-time section
    builder.add_section(
        "Meeting Real-Time Requirements",
        """
Hypersonic guidance requires update rates of 100+ Hz (< 10ms latency). 
Here are strategies to achieve this:

1. **Use FP16 precision**: 2x speedup with minimal accuracy loss
2. **Batch predictions**: Process multiple trajectory points together
3. **Pre-allocate memory**: Avoid dynamic allocation during inference
4. **Fuse pre/post processing**: Include normalization in the model
5. **Use DLA cores**: Offload convolutions to dedicated accelerators
""",
    )

    builder.add_author("HyperTensor Team")

    return builder.build()
