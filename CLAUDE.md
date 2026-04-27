---
name: python-style
description: Python coding style guidelines. Use this skill whenever writing, reviewing, or modifying Python code. Applies to all Python files (.py), code snippets, and Python-related programming tasks.
---

# Python Style Guidelines

## Overview

This skill defines coding style preferences for Python development, ensuring consistent code formatting and readability across all Python projects.

## PEP 8 Compliance

Always follow PEP 8 as the baseline style standard for all Python code. This includes but is not limited to:

- 4-space indentation (no tabs)
- Maximum line length of 79 characters (99 for comments/docstrings)
- Two blank lines before top-level definitions, one blank line before methods
- Spaces around operators and after commas
- Naming conventions: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
When any other rule in this skill conflicts with PEP 8, the rule in this skill takes precedence.

## Type Annotations

Always use type annotations wherever possible:

- Annotate all function parameters and return types
- Use type hints for class attributes
- Annotate variables when the type is not immediately obvious
- Import types from `typing` module as needed (List, Dict, Optional, Tuple, Set, Generator, etc.)
**Example:**

```python
from typing import List, Dict, Optional

def process_data(items: List[str], config: Dict[str, int]) -> Optional[str]:
    result: Optional[str] = None
    # ... implementation
    return result
```

## Import Statement Ordering

Organize imports in descending order by line length (longest first), following this pattern:

1. Start with the longest import statement
2. Continue in descending order of character length
3. Within similar lengths, maintain logical grouping
**Example:**

```python
from typing import List, Dict, Generator, Optional
from pathlib import Path
import pandas as pd
import re
```

**Multi-line import example:**

```python
from mypackage.submodule import VeryLongClassName, AnotherLongClassName
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import json
import os
```

## Docstrings — Google Style

Always write docstrings following the Google Python Style Guide. Apply docstrings to all public modules, classes, functions, and methods.

### Functions and Methods

Include a summary line, then `Args:`, `Returns:`, and `Raises:` sections as applicable. Each arg entry should list the name, type (if not already annotated), and description.

**Example:**

```python
def fetch_records(user_id: int, limit: int = 10) -> List[Dict[str, str]]:
    """Fetch user records from the database.

    Retrieves the most recent records for the given user, ordered
    by creation date descending.

    Args:
        user_id: The unique identifier of the user.
        limit: Maximum number of records to return. Defaults to 10.

    Returns:
        A list of dicts, each containing 'id', 'name', and 'created_at'
        keys with string values.

    Raises:
        ValueError: If user_id is negative.
        ConnectionError: If the database is unreachable.
    """
```

### Classes

Include a summary line describing the class purpose, followed by an `Attributes:` section for public attributes.

**Example:**

```python
class Config:
    """Application configuration loaded from environment.

    Attributes:
        db_url: Database connection string.
        debug: Whether debug mode is enabled.
        max_retries: Maximum number of retry attempts for failed requests.
    """
```

### Modules

Place a module-level docstring at the top of the file describing the module's purpose.

**Example:**

```python
"""Utilities for parsing and validating CSV uploads.

Provides functions to read CSV files, validate column schemas,
and convert rows into domain objects.
"""
```

## General Guidelines

- Follow all rules above consistently in all Python code
- When multiple rules apply, prioritize in this order: PEP 8 baseline → type annotations → import ordering → docstrings
- Custom rules in this skill (import ordering, type annotations) take precedence over PEP 8 where they conflict
