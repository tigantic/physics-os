"""
Verification and Validation (V&V) Module for Project HyperTensor.

Provides formal V&V framework including:
- Verification: Solving the equations right (code correctness)
- Validation: Solving the right equations (physical accuracy)
- Uncertainty quantification for validation
- V&V plan management and reporting

These tools support rigorous scientific software quality assurance.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from abc import ABC, abstractmethod
import torch
import numpy as np
from pathlib import Path
import json
import time


class VVLevel(Enum):
    """Level of V&V rigor."""
    BASIC = auto()      # Simple checks
    STANDARD = auto()   # Standard testing
    RIGOROUS = auto()   # Comprehensive testing
    CERTIFICATION = auto()  # Certification-level


class VVCategory(Enum):
    """Category of V&V activity."""
    CODE_VERIFICATION = auto()  # Is code correct?
    SOLUTION_VERIFICATION = auto()  # Are solutions accurate?
    VALIDATION = auto()  # Does model match reality?
    UNCERTAINTY = auto()  # What are the uncertainties?


@dataclass
class VVTest:
    """
    Definition of a V&V test.
    
    Attributes:
        name: Test name
        category: V&V category
        level: Rigor level
        description: What this test verifies/validates
        executor: Function that runs the test
        acceptance_criteria: Criteria for passing
        priority: Test priority (1=highest)
    """
    name: str
    category: VVCategory
    level: VVLevel
    description: str
    executor: Callable[[], Dict[str, Any]]
    acceptance_criteria: Dict[str, float] = field(default_factory=dict)
    priority: int = 2
    dependencies: List[str] = field(default_factory=list)
    
    def run(self) -> 'VVTestResult':
        """
        Execute the V&V test.
        
        Returns:
            VVTestResult with outcomes
        """
        start_time = time.time()
        
        try:
            outputs = self.executor()
            elapsed = time.time() - start_time
            
            # Check acceptance criteria
            passed = True
            criteria_results = {}
            
            for criterion, threshold in self.acceptance_criteria.items():
                if criterion in outputs:
                    value = outputs[criterion]
                    criterion_passed = value <= threshold
                    criteria_results[criterion] = {
                        'value': value,
                        'threshold': threshold,
                        'passed': criterion_passed,
                    }
                    if not criterion_passed:
                        passed = False
            
            return VVTestResult(
                test_name=self.name,
                category=self.category,
                passed=passed,
                duration=elapsed,
                outputs=outputs,
                criteria_results=criteria_results,
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return VVTestResult(
                test_name=self.name,
                category=self.category,
                passed=False,
                duration=elapsed,
                error=str(e),
            )


@dataclass
class VVTestResult:
    """
    Result of a V&V test execution.
    
    Attributes:
        test_name: Name of the test
        category: V&V category
        passed: Whether test passed
        duration: Execution time in seconds
        outputs: Test outputs/metrics
        criteria_results: Results per criterion
        error: Error message if failed
    """
    test_name: str
    category: VVCategory
    passed: bool
    duration: float
    outputs: Dict[str, Any] = field(default_factory=dict)
    criteria_results: Dict[str, Dict] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'category': self.category.name,
            'passed': self.passed,
            'duration': self.duration,
            'outputs': {k: float(v) if isinstance(v, (float, np.floating)) else v 
                       for k, v in self.outputs.items()},
            'criteria_results': self.criteria_results,
            'error': self.error,
        }


@dataclass
class VVPlan:
    """
    V&V Plan containing all tests to execute.
    
    Organizes tests by category and level, with dependency tracking.
    """
    name: str
    version: str = "1.0"
    description: str = ""
    tests: List[VVTest] = field(default_factory=list)
    results: List[VVTestResult] = field(default_factory=list)
    
    def add_test(self, test: VVTest):
        """Add a test to the plan."""
        self.tests.append(test)
    
    def get_tests_by_category(self, category: VVCategory) -> List[VVTest]:
        """Get all tests in a category."""
        return [t for t in self.tests if t.category == category]
    
    def get_tests_by_level(self, level: VVLevel) -> List[VVTest]:
        """Get all tests at a given level."""
        return [t for t in self.tests if t.level == level]
    
    def _resolve_order(self) -> List[VVTest]:
        """Resolve test execution order based on dependencies."""
        # Simple topological sort
        remaining = list(self.tests)
        ordered = []
        completed = set()
        
        while remaining:
            made_progress = False
            for test in remaining[:]:
                if all(dep in completed for dep in test.dependencies):
                    ordered.append(test)
                    completed.add(test.name)
                    remaining.remove(test)
                    made_progress = True
            
            if not made_progress:
                # Cycle or missing dependency - add remaining anyway
                ordered.extend(remaining)
                break
        
        return ordered
    
    def run(
        self,
        categories: Optional[List[VVCategory]] = None,
        levels: Optional[List[VVLevel]] = None,
        verbose: bool = True,
    ) -> List[VVTestResult]:
        """
        Execute the V&V plan.
        
        Args:
            categories: Filter by categories (None = all)
            levels: Filter by levels (None = all)
            verbose: Print progress
            
        Returns:
            List of test results
        """
        tests = self._resolve_order()
        
        # Apply filters
        if categories:
            tests = [t for t in tests if t.category in categories]
        if levels:
            tests = [t for t in tests if t.level in levels]
        
        # Sort by priority
        tests = sorted(tests, key=lambda t: t.priority)
        
        self.results = []
        
        for test in tests:
            if verbose:
                print(f"[{test.category.name}] {test.name}...", end=" ")
            
            result = test.run()
            self.results.append(result)
            
            if verbose:
                status = "✅" if result.passed else "❌"
                print(f"{status} ({result.duration:.2f}s)")
        
        return self.results
    
    @property
    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total': len(self.results),
            'passed': sum(1 for r in self.results if r.passed),
            'failed': sum(1 for r in self.results if not r.passed),
            'by_category': {
                cat.name: {
                    'total': sum(1 for r in self.results if r.category == cat),
                    'passed': sum(1 for r in self.results if r.category == cat and r.passed),
                }
                for cat in VVCategory
            },
        }


@dataclass 
class VVReport:
    """
    Comprehensive V&V report.
    
    Generates formatted reports of V&V activities and outcomes.
    """
    plan: VVPlan
    generated_at: float = field(default_factory=time.time)
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# V&V Report: {self.plan.name}",
            "",
            f"**Version**: {self.plan.version}",
            f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.generated_at))}",
            "",
        ]
        
        # Summary
        summary = self.plan.summary
        lines.extend([
            "## Summary",
            "",
            f"**Total Tests**: {summary['total']}",
            f"**Passed**: {summary['passed']}",
            f"**Failed**: {summary['failed']}",
            f"**Pass Rate**: {100 * summary['passed'] / max(summary['total'], 1):.1f}%",
            "",
        ])
        
        # By category
        lines.extend([
            "## Results by Category",
            "",
            "| Category | Passed | Total | Rate |",
            "|----------|--------|-------|------|",
        ])
        
        for cat_name, stats in summary['by_category'].items():
            if stats['total'] > 0:
                rate = 100 * stats['passed'] / stats['total']
                lines.append(f"| {cat_name} | {stats['passed']} | {stats['total']} | {rate:.0f}% |")
        
        lines.append("")
        
        # Detailed results
        lines.extend([
            "## Detailed Results",
            "",
        ])
        
        for result in self.plan.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            lines.extend([
                f"### {result.test_name}",
                "",
                f"**Status**: {status}",
                f"**Category**: {result.category.name}",
                f"**Duration**: {result.duration:.3f}s",
                "",
            ])
            
            if result.criteria_results:
                lines.append("**Criteria**:")
                for name, info in result.criteria_results.items():
                    c_status = "✅" if info['passed'] else "❌"
                    lines.append(f"- {c_status} {name}: {info['value']:.6e} (threshold: {info['threshold']:.6e})")
                lines.append("")
            
            if result.error:
                lines.extend([
                    f"**Error**: {result.error}",
                    "",
                ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'plan_name': self.plan.name,
            'version': self.plan.version,
            'generated_at': self.generated_at,
            'summary': self.plan.summary,
            'results': [r.to_dict() for r in self.plan.results],
        }
    
    def save(self, filepath: Union[str, Path], format: str = "markdown"):
        """
        Save report to file.
        
        Args:
            filepath: Output file path
            format: "markdown" or "json"
        """
        with open(filepath, 'w') as f:
            if format == "json":
                json.dump(self.to_dict(), f, indent=2)
            else:
                f.write(self.to_markdown())


class CodeVerification(ABC):
    """
    Abstract base class for code verification tests.
    
    Code verification answers: "Did we solve the equations correctly?"
    """
    
    @abstractmethod
    def verify(self) -> Dict[str, Any]:
        """
        Run verification test.
        
        Returns:
            Dictionary of verification metrics
        """
        pass


class UnitVerification(CodeVerification):
    """
    Unit-level code verification.
    
    Tests individual functions/classes for correctness.
    """
    
    def __init__(
        self,
        function: Callable,
        test_cases: List[Tuple[tuple, Any]],
        comparator: Optional[Callable[[Any, Any], bool]] = None,
    ):
        """
        Initialize unit verification.
        
        Args:
            function: Function to test
            test_cases: List of (inputs, expected_output) tuples
            comparator: Custom comparison function
        """
        self.function = function
        self.test_cases = test_cases
        self.comparator = comparator or (lambda a, b: a == b)
    
    def verify(self) -> Dict[str, Any]:
        """Run all test cases."""
        results = []
        
        for inputs, expected in self.test_cases:
            try:
                actual = self.function(*inputs)
                passed = self.comparator(actual, expected)
                results.append({
                    'inputs': inputs,
                    'expected': expected,
                    'actual': actual,
                    'passed': passed,
                })
            except Exception as e:
                results.append({
                    'inputs': inputs,
                    'expected': expected,
                    'error': str(e),
                    'passed': False,
                })
        
        n_passed = sum(1 for r in results if r['passed'])
        
        return {
            'n_tests': len(results),
            'n_passed': n_passed,
            'pass_rate': n_passed / len(results) if results else 0,
            'all_passed': n_passed == len(results),
            'results': results,
        }


class IntegrationVerification(CodeVerification):
    """
    Integration-level code verification.
    
    Tests interactions between components.
    """
    
    def __init__(
        self,
        workflow: Callable[[], Any],
        expected_properties: Dict[str, Callable[[Any], bool]],
    ):
        """
        Initialize integration verification.
        
        Args:
            workflow: Function that runs the integration test
            expected_properties: Dict of property_name -> checker function
        """
        self.workflow = workflow
        self.expected_properties = expected_properties
    
    def verify(self) -> Dict[str, Any]:
        """Run integration test."""
        try:
            result = self.workflow()
            
            property_results = {}
            all_passed = True
            
            for name, checker in self.expected_properties.items():
                try:
                    passed = checker(result)
                    property_results[name] = passed
                    if not passed:
                        all_passed = False
                except Exception as e:
                    property_results[name] = False
                    all_passed = False
            
            return {
                'executed': True,
                'all_passed': all_passed,
                'property_results': property_results,
            }
            
        except Exception as e:
            return {
                'executed': False,
                'error': str(e),
                'all_passed': False,
            }


@dataclass
class ValidationCase:
    """
    Definition of a validation case.
    
    Validation answers: "Are we solving the right equations?"
    """
    name: str
    description: str
    experimental_data: Optional[Any] = None
    analytical_solution: Optional[Callable] = None
    simulation_runner: Optional[Callable] = None
    metrics: List[str] = field(default_factory=list)


class ExperimentalValidation:
    """
    Validation against experimental data.
    
    Compares simulation results to experimental measurements.
    """
    
    def __init__(
        self,
        case: ValidationCase,
        metrics: Optional[List[str]] = None,
    ):
        """
        Initialize experimental validation.
        
        Args:
            case: Validation case definition
            metrics: Specific metrics to compare
        """
        self.case = case
        self.metrics = metrics or case.metrics
    
    def validate(self) -> Dict[str, Any]:
        """
        Run validation against experimental data.
        
        Returns:
            Dictionary of validation results
        """
        if self.case.simulation_runner is None:
            return {'error': 'No simulation runner defined'}
        
        if self.case.experimental_data is None:
            return {'error': 'No experimental data available'}
        
        # Run simulation
        sim_result = self.case.simulation_runner()
        
        # Compare metrics
        results = {}
        for metric in self.metrics:
            if hasattr(sim_result, metric) and hasattr(self.case.experimental_data, metric):
                sim_val = getattr(sim_result, metric)
                exp_val = getattr(self.case.experimental_data, metric)
                
                if isinstance(sim_val, (int, float)) and isinstance(exp_val, (int, float)):
                    error = abs(sim_val - exp_val)
                    rel_error = error / abs(exp_val) if exp_val != 0 else float('inf')
                    
                    results[metric] = {
                        'simulation': sim_val,
                        'experimental': exp_val,
                        'absolute_error': error,
                        'relative_error': rel_error,
                    }
        
        return {
            'case': self.case.name,
            'metrics': results,
            'n_compared': len(results),
        }


class AnalyticalValidation:
    """
    Validation against analytical solutions.
    
    Compares simulation results to exact analytical solutions
    for convergence rate verification.
    """
    
    def __init__(
        self,
        case: ValidationCase,
        grid_sizes: List[int],
    ):
        """
        Initialize analytical validation.
        
        Args:
            case: Validation case with analytical solution
            grid_sizes: Grid sizes for convergence study
        """
        self.case = case
        self.grid_sizes = grid_sizes
    
    def validate(self) -> Dict[str, Any]:
        """
        Run convergence study against analytical solution.
        
        Returns:
            Dictionary with convergence results
        """
        if self.case.analytical_solution is None:
            return {'error': 'No analytical solution defined'}
        
        errors = []
        
        for n in self.grid_sizes:
            # This would need to be customized per case
            # Here's the general structure
            dx = 1.0 / n
            
            # Compute analytical and numerical solutions
            x = np.linspace(0, 1, n)
            analytical = self.case.analytical_solution(x)
            
            # Placeholder for numerical solution
            # In practice, run simulation at this grid size
            numerical = analytical + np.random.normal(0, dx**2, n)  # Placeholder
            
            error = np.sqrt(np.mean((numerical - analytical)**2))
            errors.append(error)
        
        # Fit convergence rate
        log_h = np.log(1.0 / np.array(self.grid_sizes))
        log_e = np.log(np.array(errors))
        
        # Linear regression for p in e = C * h^p
        A = np.vstack([log_h, np.ones(len(log_h))]).T
        p, log_C = np.linalg.lstsq(A, log_e, rcond=None)[0]
        
        return {
            'case': self.case.name,
            'grid_sizes': self.grid_sizes,
            'errors': errors,
            'convergence_rate': float(p),
            'constant': float(np.exp(log_C)),
        }


@dataclass
class UncertaintyBand:
    """
    Uncertainty band for validation comparisons.
    
    Represents experimental or numerical uncertainty.
    """
    lower: np.ndarray
    upper: np.ndarray
    confidence: float = 0.95
    
    @classmethod
    def from_mean_std(
        cls,
        mean: np.ndarray,
        std: np.ndarray,
        n_sigma: float = 2.0,
    ) -> 'UncertaintyBand':
        """
        Create uncertainty band from mean and standard deviation.
        
        Args:
            mean: Mean values
            std: Standard deviations
            n_sigma: Number of standard deviations
            
        Returns:
            UncertaintyBand
        """
        return cls(
            lower=mean - n_sigma * std,
            upper=mean + n_sigma * std,
            confidence=0.95 if n_sigma == 2.0 else 0.68 if n_sigma == 1.0 else 0.99,
        )
    
    def contains(self, values: np.ndarray) -> np.ndarray:
        """Check which values fall within the band."""
        return (values >= self.lower) & (values <= self.upper)
    
    @property
    def width(self) -> np.ndarray:
        """Get band width at each point."""
        return self.upper - self.lower


class ValidationUncertainty:
    """
    Uncertainty quantification for validation.
    
    Propagates uncertainties through validation comparisons.
    """
    
    def __init__(
        self,
        experimental_uncertainty: UncertaintyBand,
        numerical_uncertainty: Optional[UncertaintyBand] = None,
    ):
        """
        Initialize validation uncertainty.
        
        Args:
            experimental_uncertainty: Experimental data uncertainty
            numerical_uncertainty: Numerical solution uncertainty
        """
        self.exp_unc = experimental_uncertainty
        self.num_unc = numerical_uncertainty
    
    def compute_validation_metric(
        self,
        simulation: np.ndarray,
        experimental_mean: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute validation metric accounting for uncertainties.
        
        Uses ASME V&V 20-2009 approach.
        
        Args:
            simulation: Simulation results
            experimental_mean: Experimental mean values
            
        Returns:
            Dictionary of validation metrics
        """
        # Comparison error
        E = simulation - experimental_mean
        
        # Experimental uncertainty (u_D)
        u_D = (self.exp_unc.upper - self.exp_unc.lower) / 4  # Assuming 95% CI
        
        # Numerical uncertainty (if available)
        if self.num_unc:
            u_num = (self.num_unc.upper - self.num_unc.lower) / 4
        else:
            u_num = np.zeros_like(u_D)
        
        # Validation uncertainty
        u_val = np.sqrt(u_D**2 + u_num**2)
        
        # Normalized comparison error
        E_norm = E / (u_val + 1e-15)
        
        # Validation metric (fraction within uncertainty)
        within_uncertainty = np.abs(E) <= 2 * u_val
        fraction_valid = np.mean(within_uncertainty)
        
        return {
            'mean_error': float(np.mean(np.abs(E))),
            'max_error': float(np.max(np.abs(E))),
            'mean_uncertainty': float(np.mean(u_val)),
            'fraction_within_uncertainty': float(fraction_valid),
            'validation_passed': fraction_valid >= 0.95,
        }


def run_vv_plan(
    plan: VVPlan,
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Tuple[bool, VVReport]:
    """
    Execute a V&V plan and generate report.
    
    Args:
        plan: V&V plan to execute
        output_path: Optional path to save report
        verbose: Print progress
        
    Returns:
        Tuple of (all_passed, report)
    """
    plan.run(verbose=verbose)
    
    report = VVReport(plan=plan)
    
    if output_path:
        report.save(output_path)
    
    all_passed = all(r.passed for r in plan.results)
    
    return all_passed, report


def generate_vv_report(
    plan: VVPlan,
    format: str = "markdown",
) -> str:
    """
    Generate V&V report from completed plan.
    
    Args:
        plan: Completed V&V plan
        format: Output format ("markdown" or "json")
        
    Returns:
        Formatted report string
    """
    report = VVReport(plan=plan)
    
    if format == "json":
        return json.dumps(report.to_dict(), indent=2)
    return report.to_markdown()
