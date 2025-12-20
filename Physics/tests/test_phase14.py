"""
Phase 14 Integration Tests: Documentation Module.

Tests for:
- API reference extraction
- User guide generation
- Sphinx configuration
- Code examples execution
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestAPIReferenceExtraction:
    """Tests for API documentation extraction."""
    
    def test_parameter_doc(self):
        """Test ParameterDoc creation and formatting."""
        from tensornet.docs import ParameterDoc
        
        param = ParameterDoc(
            name="chi_max",
            type_annotation="int",
            description="Maximum bond dimension",
            default="32",
            optional=True,
        )
        
        assert param.name == "chi_max"
        assert param.type_annotation == "int"
        assert param.optional is True
        
        # Test markdown conversion
        md = param.to_markdown()
        assert "chi_max" in md
        assert "int" in md
        assert "Maximum bond dimension" in md
    
    def test_function_doc(self):
        """Test FunctionDoc creation and formatting."""
        from tensornet.docs import FunctionDoc, ParameterDoc, ReturnDoc
        
        func_doc = FunctionDoc(
            name="run_dmrg",
            signature="(mps, mpo, chi_max=32, n_sweeps=10)",
            short_description="Run DMRG to find the ground state.",
            long_description="The Density Matrix Renormalization Group algorithm.",
            parameters=[
                ParameterDoc(name="mps", type_annotation="MPS", description="Initial MPS"),
                ParameterDoc(name="mpo", type_annotation="MPO", description="Hamiltonian"),
            ],
            returns=ReturnDoc(type_annotation="Tuple[MPS, float]", description="Ground state and energy"),
            is_async=False,
        )
        
        assert func_doc.name == "run_dmrg"
        assert len(func_doc.parameters) == 2
        assert func_doc.returns is not None
        
        # Test markdown conversion
        md = func_doc.to_markdown()
        assert "run_dmrg" in md
        assert "DMRG" in md
        assert "Parameters" in md
        assert "Returns" in md
    
    def test_class_doc(self):
        """Test ClassDoc creation and formatting."""
        from tensornet.docs import ClassDoc, AttributeDoc, FunctionDoc
        
        class_doc = ClassDoc(
            name="MPS",
            short_description="Matrix Product State class.",
            long_description="Represents a quantum state as a chain of tensors.",
            bases=["nn.Module"],
            attributes=[
                AttributeDoc(name="L", type_annotation="int", description="Number of sites"),
                AttributeDoc(name="d", type_annotation="int", description="Local dimension"),
            ],
            methods=[
                FunctionDoc(name="norm", signature="()", short_description="Compute the norm"),
            ],
        )
        
        assert class_doc.name == "MPS"
        assert len(class_doc.attributes) == 2
        assert len(class_doc.methods) == 1
        assert "nn.Module" in class_doc.bases
        
        # Test markdown conversion
        md = class_doc.to_markdown()
        assert "class `MPS`" in md
        assert "Attributes" in md
        assert "Methods" in md
    
    def test_module_doc(self):
        """Test ModuleDoc creation and formatting."""
        from tensornet.docs import ModuleDoc, FunctionDoc, ClassDoc
        
        module_doc = ModuleDoc(
            name="tensornet.core",
            short_description="Core tensor network components.",
            functions=[
                FunctionDoc(name="svd_truncated", short_description="Truncated SVD"),
            ],
            classes=[
                ClassDoc(name="MPS", short_description="Matrix Product State"),
            ],
        )
        
        assert module_doc.name == "tensornet.core"
        assert len(module_doc.functions) == 1
        assert len(module_doc.classes) == 1
        
        # Test markdown conversion
        md = module_doc.to_markdown()
        assert "tensornet.core" in md
        assert "Functions" in md
        assert "Classes" in md
    
    def test_docstring_parser_google_style(self):
        """Test parsing Google-style docstrings."""
        from tensornet.docs.api_reference import DocstringParser, DocstringStyle
        
        parser = DocstringParser(DocstringStyle.GOOGLE)
        
        docstring = """
        Compute the ground state energy.
        
        This function uses DMRG to find the ground state.
        
        Args:
            mpo: The Hamiltonian as an MPO.
            chi_max (int): Maximum bond dimension. Default: 32.
            
        Returns:
            float: The ground state energy.
            
        Raises:
            ValueError: If chi_max is negative.
        """
        
        result = parser.parse(docstring)
        
        assert "ground state energy" in result['short_description']
        assert len(result['parameters']) == 2
        assert result['returns'] is not None
        assert len(result['raises']) == 1
    
    def test_api_extractor(self):
        """Test API extraction from modules."""
        from tensornet.docs import APIExtractor
        
        extractor = APIExtractor(include_private=False)
        
        # Test extraction from a simple module
        import tensornet.core.decompositions as decomp_module
        
        doc = extractor.extract_module(decomp_module)
        
        assert doc.name == "tensornet.core.decompositions"
        assert len(doc.functions) > 0


class TestUserGuides:
    """Tests for user guide generation."""
    
    def test_code_example(self):
        """Test CodeExample creation and formatting."""
        from tensornet.docs import CodeExample
        
        example = CodeExample(
            code="x = torch.randn(10)\nprint(x.shape)",
            description="Create a random tensor",
            expected_output="torch.Size([10])",
            title="Random Tensor",
        )
        
        assert example.code.strip().startswith("x = torch.randn")
        assert example.title == "Random Tensor"
        
        # Test markdown conversion
        md = example.to_markdown()
        assert "```python" in md
        assert "Random Tensor" in md
        assert "Output:" in md
    
    def test_guide_section(self):
        """Test GuideSection creation and formatting."""
        from tensornet.docs import GuideSection, CodeExample
        
        section = GuideSection(
            title="Introduction",
            content="Welcome to HyperTensor!",
            level=2,
            examples=[
                CodeExample(code="import tensornet", description="Import the library"),
            ],
            notes=[{"type": "tip", "content": "Start with the tutorials."}],
        )
        
        assert section.title == "Introduction"
        assert len(section.examples) == 1
        assert len(section.notes) == 1
        
        # Test markdown conversion
        md = section.to_markdown()
        assert "## Introduction" in md
        assert "Welcome to HyperTensor" in md
        assert "TIP" in md
    
    def test_tutorial(self):
        """Test Tutorial creation and formatting."""
        from tensornet.docs import Tutorial, GuideSection, DifficultyLevel
        
        tutorial = Tutorial(
            title="Getting Started",
            description="Learn to use HyperTensor.",
            difficulty=DifficultyLevel.BEGINNER,
            prerequisites=["Python 3.9+"],
            objectives=["Install HyperTensor"],
            sections=[
                GuideSection(title="Installation", content="Run pip install..."),
            ],
        )
        
        assert tutorial.title == "Getting Started"
        assert tutorial.difficulty == DifficultyLevel.BEGINNER
        assert len(tutorial.prerequisites) == 1
        assert len(tutorial.sections) == 1
        
        # Test markdown conversion
        md = tutorial.to_markdown()
        assert "# Getting Started" in md
        assert "Beginner" in md
        assert "Prerequisites" in md
    
    def test_guide_builder(self):
        """Test GuideBuilder fluent interface."""
        from tensornet.docs import GuideBuilder, DifficultyLevel
        
        builder = GuideBuilder("My Tutorial")
        
        tutorial = (
            builder
            .set_description("A test tutorial")
            .set_difficulty(DifficultyLevel.INTERMEDIATE)
            .add_prerequisite("Basic Python")
            .add_objective("Learn testing")
            .add_section("Introduction", "Welcome!")
            .add_example("print('hello')", description="Say hello")
            .add_note("This is important", note_type="warning")
            .build()
        )
        
        assert tutorial.title == "My Tutorial"
        assert tutorial.difficulty == DifficultyLevel.INTERMEDIATE
        assert len(tutorial.prerequisites) == 1
        assert len(tutorial.objectives) == 1
        assert len(tutorial.sections) == 1
        assert len(tutorial.sections[0].examples) == 1
        assert len(tutorial.sections[0].notes) == 1
    
    def test_getting_started_guide(self):
        """Test Getting Started guide generation."""
        from tensornet.docs import create_getting_started
        
        guide = create_getting_started()
        
        assert guide.title == "Getting Started with HyperTensor"
        assert len(guide.sections) > 0
        assert len(guide.prerequisites) > 0
        assert len(guide.objectives) > 0
        
        # Check it generates valid markdown
        md = guide.to_markdown()
        assert len(md) > 1000  # Should be substantial content
        assert "Installation" in md
    
    def test_cfd_tutorial(self):
        """Test CFD tutorial generation."""
        from tensornet.docs import create_cfd_tutorial
        
        guide = create_cfd_tutorial()
        
        assert "Fluid Dynamics" in guide.title or "CFD" in guide.title
        assert len(guide.sections) > 0
        
        md = guide.to_markdown()
        assert "Euler" in md
        assert "Navier-Stokes" in md
    
    def test_tensor_network_primer(self):
        """Test Tensor Network primer generation."""
        from tensornet.docs import create_tensor_network_primer
        
        guide = create_tensor_network_primer()
        
        assert "Tensor" in guide.title
        assert len(guide.sections) > 0
        
        md = guide.to_markdown()
        assert "MPS" in md
        assert "DMRG" in md
    
    def test_deployment_guide(self):
        """Test Deployment guide generation."""
        from tensornet.docs import create_deployment_guide
        
        guide = create_deployment_guide()
        
        assert "Deploy" in guide.title
        assert len(guide.sections) > 0
        
        md = guide.to_markdown()
        assert "ONNX" in md or "TensorRT" in md


class TestSphinxConfig:
    """Tests for Sphinx configuration generation."""
    
    def test_sphinx_config_creation(self):
        """Test SphinxConfig creation."""
        from tensornet.docs import SphinxConfig, SphinxTheme
        
        config = SphinxConfig(
            project="TestProject",
            author="Test Author",
            version="1.0",
            theme=SphinxTheme.FURO,
        )
        
        assert config.project == "TestProject"
        assert config.author == "Test Author"
        assert config.theme == SphinxTheme.FURO
        assert len(config.extensions) > 0  # Default extensions
    
    def test_generate_conf_py(self):
        """Test conf.py generation."""
        from tensornet.docs import SphinxConfig, generate_conf_py
        
        config = SphinxConfig(
            project="HyperTensor",
            author="TiganticLabz",
            version="2.2.0",
        )
        
        content = generate_conf_py(config)
        
        assert "HyperTensor" in content
        assert "TiganticLabz" in content
        assert "extensions = [" in content
        assert "sphinx.ext.autodoc" in content
        assert "html_theme" in content
    
    def test_sphinx_builder(self):
        """Test SphinxBuilder initialization."""
        from tensornet.docs import SphinxConfig, SphinxBuilder
        from pathlib import Path
        import tempfile
        
        config = SphinxConfig(project="Test")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = SphinxBuilder(
                config=config,
                source_dir=Path(tmpdir) / "docs",
            )
            
            assert builder.config.project == "Test"
            assert builder.source_dir.name == "docs"


class TestCodeExamples:
    """Tests for runnable code examples."""
    
    def test_example_config(self):
        """Test ExampleConfig creation."""
        from tensornet.docs import ExampleConfig
        
        config = ExampleConfig(
            timeout_seconds=10.0,
            capture_output=True,
            show_traceback=True,
        )
        
        assert config.timeout_seconds == 10.0
        assert config.capture_output is True
    
    def test_runnable_example_success(self):
        """Test successful example execution."""
        from tensornet.docs import RunnableExample, ExampleStatus
        
        example = RunnableExample(
            code="x = 2 + 2\nprint(x)",
            name="simple_add",
            expected_output="4",
        )
        
        result = example.run()
        
        assert result.status == ExampleStatus.PASSED
        assert "4" in result.output
        assert result.duration_seconds >= 0  # Can be 0 for fast operations
    
    def test_runnable_example_failure(self):
        """Test failed example (wrong output)."""
        from tensornet.docs import RunnableExample, ExampleStatus
        
        example = RunnableExample(
            code="print(5)",
            name="wrong_output",
            expected_output="10",
        )
        
        result = example.run()
        
        assert result.status == ExampleStatus.FAILED
        assert "5" in result.output
    
    def test_runnable_example_error(self):
        """Test example with runtime error."""
        from tensornet.docs import RunnableExample, ExampleStatus
        
        example = RunnableExample(
            code="raise ValueError('test error')",
            name="error_example",
        )
        
        result = example.run()
        
        assert result.status == ExampleStatus.ERROR
        assert result.error is not None
        # Check that ValueError is captured (either in error type or traceback)
        assert "ValueError" in result.traceback_str or isinstance(result.error, ValueError)
    
    def test_runnable_example_skip(self):
        """Test skipped example."""
        from tensornet.docs import RunnableExample, ExampleStatus
        
        example = RunnableExample(
            code="print('hello')",
            name="skipped",
            skip=True,
            skip_reason="Not needed for test",
        )
        
        result = example.run()
        
        assert result.status == ExampleStatus.SKIPPED
    
    def test_example_runner(self):
        """Test ExampleRunner with multiple examples."""
        from tensornet.docs import ExampleRunner, RunnableExample
        
        runner = ExampleRunner()
        
        runner.add_example(RunnableExample(
            code="print(1)",
            name="ex1",
            expected_output="1",
        ))
        runner.add_example(RunnableExample(
            code="print(2)",
            name="ex2",
            expected_output="2",
        ))
        runner.add_example(RunnableExample(
            code="print(3)",
            name="ex3",
            expected_output="wrong",  # Will fail
        ))
        
        results = runner.run_all(verbose=False)
        
        assert results["total"] == 3
        assert results["passed"] == 2
        assert results["failed"] == 1
    
    def test_example_runner_report(self):
        """Test ExampleRunner report generation."""
        from tensornet.docs import ExampleRunner, RunnableExample
        
        runner = ExampleRunner()
        runner.add_example(RunnableExample(code="print(1)", name="test"))
        runner.run_all(verbose=False)
        
        # Text report
        text_report = runner.generate_report("text")
        assert "Example Execution Report" in text_report
        
        # Markdown report
        md_report = runner.generate_report("markdown")
        assert "# Example Execution Report" in md_report
        assert "✅" in md_report or "passed" in md_report.lower()
        
        # JSON report
        json_report = runner.generate_report("json")
        assert '"total"' in json_report
    
    def test_validate_example(self):
        """Test validate_example convenience function."""
        from tensornet.docs import validate_example, ExampleStatus
        
        result = validate_example("x = 1 + 1\nprint(x)", expected_output="2")
        
        assert result.status == ExampleStatus.PASSED
    
    def test_validate_syntax(self):
        """Test syntax validation without execution."""
        from tensornet.docs.examples import validate_syntax
        
        # Valid syntax
        is_valid, error = validate_syntax("x = 1 + 1")
        assert is_valid is True
        assert error is None
        
        # Invalid syntax
        is_valid, error = validate_syntax("x = = 1")
        assert is_valid is False
        assert error is not None
    
    def test_example_to_notebook_cell(self):
        """Test converting example to notebook cell."""
        from tensornet.docs import RunnableExample
        
        example = RunnableExample(
            code="import torch\nx = torch.randn(10)",
            name="notebook_example",
            expected_output="tensor([...])",
        )
        
        cell = example.to_notebook_cell()
        
        assert cell["cell_type"] == "code"
        assert "import torch" in cell["source"][0]
        assert len(cell["outputs"]) > 0


class TestDocumentationImports:
    """Test that all documentation module exports work."""
    
    def test_api_reference_imports(self):
        """Test API reference module imports."""
        from tensornet.docs import (
            ModuleDoc,
            ClassDoc,
            FunctionDoc,
            ParameterDoc,
            APIExtractor,
            extract_module_docs,
            generate_api_markdown,
            generate_api_rst,
        )
        
        assert ModuleDoc is not None
        assert ClassDoc is not None
        assert FunctionDoc is not None
        assert APIExtractor is not None
    
    def test_user_guides_imports(self):
        """Test user guides module imports."""
        from tensornet.docs import (
            GuideSection,
            Tutorial,
            CodeExample,
            GuideBuilder,
            create_getting_started,
            create_cfd_tutorial,
            create_tensor_network_primer,
            create_deployment_guide,
        )
        
        assert GuideSection is not None
        assert Tutorial is not None
        assert GuideBuilder is not None
    
    def test_sphinx_imports(self):
        """Test Sphinx config module imports."""
        from tensornet.docs import (
            SphinxConfig,
            SphinxTheme,
            SphinxBuilder,
            generate_conf_py,
            build_documentation,
        )
        
        assert SphinxConfig is not None
        assert SphinxTheme is not None
        assert SphinxBuilder is not None
    
    def test_examples_imports(self):
        """Test examples module imports."""
        from tensornet.docs import (
            ExampleConfig,
            RunnableExample,
            ExampleRunner,
            validate_example,
            extract_examples_from_docstrings,
        )
        
        assert ExampleConfig is not None
        assert RunnableExample is not None
        assert ExampleRunner is not None
    
    def test_main_package_exports(self):
        """Test that main package exports documentation module."""
        import tensornet
        
        # Check key exports exist
        assert hasattr(tensornet, 'ModuleDoc')
        assert hasattr(tensornet, 'ClassDoc')
        assert hasattr(tensornet, 'Tutorial')
        assert hasattr(tensornet, 'GuideBuilder')
        assert hasattr(tensornet, 'SphinxConfig')
        assert hasattr(tensornet, 'RunnableExample')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
