import pytest


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    # Tests must not use a developer's real contribution or GitHub settings.
    # Setting (rather than deleting) also restores env changes made directly by
    # runner-under-test code, even when the original variable did not exist.
    for key in ('GH_TOKEN', 'GITHUB_TOKEN', 'MINDIE_COMMUNITY_POLICY'):
        monkeypatch.setenv(key, '')
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_ROOT', str(tmp_path.resolve() / 'isolated-diagnostics'))
    monkeypatch.setenv('MINDIE_DIAGNOSTICS_CONFIG', str(tmp_path.resolve() / 'reporting.json'))
    # mindie-sim is a standalone package and its test extra does not install
    # the optional diagnostics sibling. Keep its transport guard when present,
    # without making every offline simulation test depend on that package.
    try:
        from mindie_diagnostics import reporter
    except ModuleNotFoundError as exc:
        if exc.name != 'mindie_diagnostics':
            raise
        return
    def unexpected_network(*args, **kwargs):
        pytest.fail('unit tests must explicitly mock the HTTPS transport')
    monkeypatch.setattr(reporter.GitHub, 'request', unexpected_network)


@pytest.fixture
def reporting_consent(tmp_path, monkeypatch, isolated_runtime):
    """Independent explicit public fault-reporting consent, no community scope."""
    from mindie_diagnostics.integration import configure_reporting
    from mindie_diagnostics.reporting import current_consent
    configure_reporting(True, roots=[str(tmp_path.resolve())])
    return current_consent()
