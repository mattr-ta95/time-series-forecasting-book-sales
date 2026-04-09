# Contributing to Time Series Forecasting Project

Thank you for your interest in contributing to this time series forecasting project! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of time series analysis and machine learning

### Setup Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/time-series-forecasting.git
   cd time-series-forecasting
   ```

3. **Run the setup script**:
   ```bash
   python setup.py
   ```

4. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Contribution Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Fixes**: Fix issues in existing code
2. **Feature Enhancements**: Add new forecasting models or analysis methods
3. **Documentation**: Improve README, code comments, or add tutorials
4. **Data Analysis**: Add analysis of additional books or time periods
5. **Performance Improvements**: Optimize existing code
6. **Testing**: Add unit tests or improve test coverage

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Add comments for complex logic

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, well-documented code
   - Test your changes thoroughly
   - Update documentation if needed

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots if applicable

### Commit Message Format

Use clear, descriptive commit messages:
- `Add: new feature description`
- `Fix: bug description`
- `Update: what was updated`
- `Remove: what was removed`
- `Docs: documentation changes`

## Development Areas

### High Priority
- [ ] Add unit tests for core functions
- [ ] Implement additional forecasting models (Prophet, NeuralProphet)
- [ ] Add interactive visualizations
- [ ] Improve error handling and validation
- [ ] Add configuration file support

### Medium Priority
- [ ] Add more books to the analysis
- [ ] Implement ensemble methods
- [ ] Add real-time forecasting capabilities
- [ ] Create Jupyter notebook tutorials
- [ ] Add model comparison dashboard

### Low Priority
- [ ] Add support for different data sources
- [ ] Implement automated model selection
- [ ] Add confidence interval visualization
- [ ] Create API endpoints
- [ ] Add Docker support

## Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=src
```

### Writing Tests
- Create test files in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies when appropriate

## Documentation

### Code Documentation
- Add docstrings to all functions and classes
- Use type hints where appropriate
- Include examples in docstrings for complex functions

### User Documentation
- Update README.md for new features
- Add usage examples
- Document any breaking changes

## Data Handling

### Adding New Data
- Place data files in the `data/` directory
- Update `data/README.md` with new data descriptions
- Ensure data follows the expected format
- Add data validation checks

### Data Privacy
- Never commit sensitive or personal data
- Use sample data for examples
- Follow data licensing requirements

## Performance Considerations

### Code Optimization
- Profile code before optimizing
- Use vectorized operations with NumPy/Pandas
- Consider memory usage for large datasets
- Add progress bars for long-running operations

### Model Performance
- Document model training times
- Provide memory requirements
- Include performance benchmarks
- Consider model size and deployment

## Issue Reporting

### Bug Reports
When reporting bugs, please include:
- Python version and operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces
- Sample data if applicable

### Feature Requests
For feature requests, please include:
- Clear description of the feature
- Use case and motivation
- Potential implementation approach
- Any relevant examples or references

## Code Review Process

### Review Criteria
- Code correctness and functionality
- Code style and documentation
- Performance implications
- Test coverage
- Security considerations

### Review Process
1. Automated checks (linting, tests)
2. Peer review by maintainers
3. Discussion and feedback
4. Approval and merge

## Community Guidelines

### Communication
- Be respectful and constructive
- Use clear, professional language
- Provide helpful feedback
- Ask questions when needed

### Collaboration
- Help other contributors
- Share knowledge and best practices
- Participate in discussions
- Follow the project's code of conduct

## Getting Help

### Resources
- Check existing issues and discussions
- Review the README and documentation
- Ask questions in GitHub discussions
- Contact maintainers for urgent issues

### Support Channels
- GitHub Issues for bug reports
- GitHub Discussions for questions
- Pull Request comments for code review
- Email for sensitive issues

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- GitHub contributor statistics

Thank you for contributing to this project!
