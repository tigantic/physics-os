"""
Documentation Module for Project HyperTensor.

Phase 14: Comprehensive documentation framework including:
- API reference generation from docstrings
- User guides and tutorials
- Sphinx integration for static site generation
- Interactive examples and code snippets

Components:
    - api_reference: Automated API documentation extraction
    - user_guides: Tutorial and guide generation
    - sphinx_config: Sphinx configuration utilities
    - examples: Runnable code examples
"""

from .api_reference import (
                            APIExtractor,
                            AttributeDoc,
                            ClassDoc,
                            DocstringParser,
                            DocstringStyle,
                            ExampleDoc,
                            FunctionDoc,
                            ModuleDoc,
                            ParameterDoc,
                            RaisesDoc,
                            ReturnDoc,
                            extract_module_docs,
                            generate_api_markdown,
                            generate_api_rst,
)
from .examples import (
                            ExampleConfig,
                            ExampleResult,
                            ExampleRunner,
                            ExampleStatus,
                            ExampleType,
                            RunnableExample,
                            extract_examples_from_docstrings,
                            validate_example,
)
from .sphinx_config import (
                            OutputFormat,
                            SphinxBuilder,
                            SphinxConfig,
                            SphinxExtension,
                            SphinxTheme,
                            build_documentation,
                            generate_conf_py,
                            generate_index_rst,
)
from .user_guides import (
                            CodeExample,
                            DifficultyLevel,
                            GuideBuilder,
                            GuideSection,
                            GuideType,
                            Tutorial,
                            create_cfd_tutorial,
                            create_deployment_guide,
                            create_getting_started,
                            create_tensor_network_primer,
)

__all__ = [
    # API Reference
    "ModuleDoc",
    "ClassDoc",
    "FunctionDoc",
    "ParameterDoc",
    "ReturnDoc",
    "RaisesDoc",
    "ExampleDoc",
    "AttributeDoc",
    "DocstringStyle",
    "DocstringParser",
    "APIExtractor",
    "extract_module_docs",
    "generate_api_markdown",
    "generate_api_rst",
    # User Guides
    "GuideSection",
    "Tutorial",
    "CodeExample",
    "GuideBuilder",
    "DifficultyLevel",
    "GuideType",
    "create_getting_started",
    "create_cfd_tutorial",
    "create_tensor_network_primer",
    "create_deployment_guide",
    # Sphinx
    "SphinxConfig",
    "SphinxTheme",
    "SphinxExtension",
    "OutputFormat",
    "SphinxBuilder",
    "generate_conf_py",
    "generate_index_rst",
    "build_documentation",
    # Examples
    "ExampleConfig",
    "ExampleResult",
    "ExampleType",
    "ExampleStatus",
    "RunnableExample",
    "ExampleRunner",
    "validate_example",
    "extract_examples_from_docstrings",
]
