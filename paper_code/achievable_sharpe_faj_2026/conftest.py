def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full 500-draw bootstrap (~1 min)")
