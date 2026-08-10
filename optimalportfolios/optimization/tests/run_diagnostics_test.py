"""
the run-level logging and aggregation layer in ``optimization.solver_diagnostics``.

``solver_diagnostics_test.py`` covers the per-solve validators. This file covers what sits
above them: the aggregation handlers that turn thousands of per-rebalance records into one
run summary, the ``configure_run_logging`` entry point that installs them, and the fallback
gate that fails a run whose track is partly held-prior.

That layer is worth testing precisely because it is *not* exercised by any solve. It reads
structured records off ``logging.LogRecord`` attributes rather than parsing message text —
so a handler silently tallying nothing looks identical to a clean run, and the run gate that
should have failed a bad backtest simply passes.

**Logging is global state.** Every case here either builds handlers directly or goes through
``configure_run_logging`` inside the ``restore_logging`` fixture, which snapshots the root and
package loggers and puts them back afterwards. Without that a single test would reconfigure
logging for the whole session.
"""
# packages
import importlib
import logging
import warnings
import pytest
# optimalportfolios
from optimalportfolios.optimization.solver_diagnostics import (
    DroppedGroupRecord,
    DroppedGroupSummary,
    InputContractRecord,
    InputContractSummary,
    RelaxationSummary,
    RunDiagnostics,
    SolverDiagnostic,
    SolverRejectionSummary,
    WarningSummary,
    _RepeatedWarningFilter,
    configure_run_logging,
    log_environment,
)

MANAGED_LOGGER_NAMES = [
    'optimalportfolios.optimization.solver_diagnostics',
    'optimalportfolios.optimization.constraints',
    'py.warnings',
]


@pytest.fixture
def restore_logging():
    """Snapshot the loggers ``configure_run_logging`` touches and restore them afterwards."""
    root = logging.getLogger()
    saved = [(root, list(root.handlers), root.level)]
    for name in MANAGED_LOGGER_NAMES:
        managed = logging.getLogger(name)
        saved.append((managed, list(managed.handlers), managed.level))
    captured = logging.raiseExceptions
    yield
    for managed, handlers, level in saved:
        managed.handlers = handlers
        managed.setLevel(level)
    logging.captureWarnings(False)
    logging.raiseExceptions = captured


def make_diagnostic(accepted: bool = True, severity: int = logging.DEBUG,
                    context: str = 'GROWM 2024-03-31') -> SolverDiagnostic:
    """One structured per-solve outcome of the shape the validators attach."""
    return SolverDiagnostic(
        context=context, solver='CLARABEL',
        status='optimal' if accepted else 'infeasible',
        outcome='accepted' if accepted else 'rejected', accepted=accepted,
        reason='' if accepted else 'budget residual', fallback_source=None if accepted
        else 'weights_0', severity=severity, n_assets=3, sum_w=1.0,
        budget_residual=0.0, max_box_violation=0.0)


def emit_to(handler: logging.Handler, level: int = logging.INFO, message: str = 'solve',
            **extra) -> None:
    """Push one record carrying ``extra`` through a handler, as the logging machinery would."""
    record = logging.LogRecord(name='test', level=level, pathname=__file__, lineno=1,
                               msg=message, args=(), exc_info=None)
    for key, value in extra.items():
        setattr(record, key, value)
    handler.handle(record)


# --------------------------------------------------------------------------- #
# SolverRejectionSummary
# --------------------------------------------------------------------------- #
def test_rejection_summary_tallies_accepted_and_rejected_solves() -> None:
    """the handler counts by severity, which is what makes the fallback rate meaningful"""
    handler = SolverRejectionSummary()
    for _ in range(7):
        emit_to(handler, solver_diag=make_diagnostic(accepted=True))
    emit_to(handler, solver_diag=make_diagnostic(accepted=False, severity=logging.ERROR))
    emit_to(handler, solver_diag=make_diagnostic(accepted=False, severity=logging.WARNING))
    emit_to(handler, solver_diag=make_diagnostic(accepted=False, severity=logging.WARNING))

    assert handler.n_total == 10
    assert handler.n_accepted == 7
    assert handler.n_rejected == 3
    assert handler.n_blowup == 1                      # ERROR: solver said optimal, we rejected
    assert handler.n_infeasible_fallback == 2         # below ERROR: no usable solution
    assert handler.fallback_fraction == pytest.approx(0.3)


def test_rejection_summary_ignores_records_without_a_diagnostic() -> None:
    """the handler reads a structured attribute, so ordinary log lines are not counted"""
    handler = SolverRejectionSummary()
    emit_to(handler, message='some unrelated message')
    emit_to(handler, solver_diag='not a SolverDiagnostic')
    assert handler.n_total == 0
    assert handler.fallback_fraction == 0.0           # no solves is not a division by zero


def test_rejection_summary_line_names_every_count() -> None:
    """the one-line summary is what lands in the run log, so it carries the whole tally"""
    handler = SolverRejectionSummary()
    emit_to(handler, solver_diag=make_diagnostic(accepted=True))
    emit_to(handler, solver_diag=make_diagnostic(accepted=False, severity=logging.ERROR))
    text = handler.summary()
    assert '2 solves' in text and '1 rejected' in text
    assert 'fallback rate 50.0%' in text


def test_fallback_gate_passes_within_the_threshold() -> None:
    """a run that fell back rarely is a clean track and the gate says so"""
    handler = SolverRejectionSummary()
    for _ in range(99):
        emit_to(handler, solver_diag=make_diagnostic(accepted=True))
    emit_to(handler, solver_diag=make_diagnostic(accepted=False, severity=logging.WARNING))
    assert handler.check_fallback_gate(max_fraction=0.05) is True


def test_fallback_gate_reports_a_breach_without_raising_by_default() -> None:
    """breaching the gate returns False and logs; it does not abort unless asked"""
    handler = SolverRejectionSummary()
    for _ in range(4):
        emit_to(handler, solver_diag=make_diagnostic(accepted=True))
    for _ in range(6):
        emit_to(handler, solver_diag=make_diagnostic(accepted=False, severity=logging.WARNING))
    assert handler.check_fallback_gate(max_fraction=0.05) is False


def test_fallback_gate_can_abort_a_production_run() -> None:
    """raise_on_breach exists so a pipeline stops before consuming a held-prior track"""
    handler = SolverRejectionSummary()
    emit_to(handler, solver_diag=make_diagnostic(accepted=False, severity=logging.WARNING))
    with pytest.raises(RuntimeError, match='FALLBACK GATE breached'):
        handler.check_fallback_gate(max_fraction=0.05, raise_on_breach=True)


def test_fallback_gate_passes_a_run_with_no_solves() -> None:
    """an empty run is vacuously within the gate rather than a division by zero"""
    assert SolverRejectionSummary().check_fallback_gate() is True


# --------------------------------------------------------------------------- #
# the other aggregation handlers
# --------------------------------------------------------------------------- #
def test_dropped_group_summary_reports_none_when_nothing_was_dropped() -> None:
    """a quiet run says so explicitly rather than returning an empty string"""
    assert DroppedGroupSummary().summary() == 'zero-loading groups dropped: none'


def test_dropped_group_summary_ranks_the_groups_it_dropped_most_often() -> None:
    """the tally names the worst offenders, which is what makes it actionable

    A group with a zero loading column is dropped from the aligned constraint set silently —
    the solve still runs, the mandate limit simply is not applied. Over a backtest that is
    thousands of dates and one line of output, so the line has to say *which* groups, not
    only how many times something happened.
    """
    handler = DroppedGroupSummary()
    for _ in range(4):
        emit_to(handler, dropped_groups=DroppedGroupRecord(groups=('Alternatives',),
                                                           no_groups_remain=False))
    emit_to(handler, dropped_groups=DroppedGroupRecord(groups=('Alternatives', 'Bonds'),
                                                       no_groups_remain=False))
    emit_to(handler, dropped_groups='not a record')      # foreign record, ignored
    text = handler.summary()
    assert '5 aligned constraint sets' in text
    assert text.index('Alternatives (5)') < text.index('Bonds (1)')


def test_relaxation_summary_ignores_foreign_records() -> None:
    """each handler filters by its own attribute, so they can share a logger"""
    handler = RelaxationSummary()
    emit_to(handler, solver_diag=make_diagnostic())    # belongs to the rejection handler
    assert handler.records == []


def test_warning_summary_counts_by_category() -> None:
    """warnings are tallied by a stable label, not by their full text"""
    handler = WarningSummary()
    for _ in range(3):
        emit_to(handler, level=logging.WARNING,
                message='/path/x.py:1: UserWarning: covariance is not positive definite')
    emit_to(handler, level=logging.WARNING,
            message='/path/y.py:2: UserWarning: something else entirely')
    assert sum(handler.counts.values()) == 4
    assert len(handler.counts) == 2
    text = handler.summary()
    assert 'captured warnings: 4' in text
    assert 'covariance is not positive definite (3)' in text


def test_warning_summary_collapses_the_factorlasso_warmup_message() -> None:
    """the repeated warm-up notice gets one stable category rather than one per asset"""
    handler = WarningSummary()
    for asset in ('spx', 'ust', 'gold'):
        emit_to(handler, level=logging.WARNING,
                message=f'factorlasso: {asset} has fewer than warmup_period observations')
    assert handler.counts == {'factorlasso warm-up assets zeroed': 3}


def test_warning_summary_reports_none_for_a_clean_run() -> None:
    """no warnings is stated, not left blank"""
    assert WarningSummary().summary() == 'captured warnings: none'


def test_repeated_warning_filter_throttles_only_the_warmup_category() -> None:
    """human-readable logs keep the first few warm-up notices and drop the rest"""
    warning_filter = _RepeatedWarningFilter(max_per_category=3)

    def warmup_record() -> logging.LogRecord:
        """A py.warnings record carrying the throttled warm-up message."""
        return logging.LogRecord(
            name='py.warnings', level=logging.WARNING, pathname=__file__, lineno=1,
            msg='factorlasso: spx has fewer than warmup_period observations', args=(),
            exc_info=None)

    kept = [warning_filter.filter(warmup_record()) for _ in range(6)]
    assert kept == [True, True, True, False, False, False]
    # every occurrence is still counted, so the summary stays accurate
    assert warning_filter.counts['factorlasso warm-up assets zeroed'] == 6


def test_repeated_warning_filter_passes_other_loggers_untouched() -> None:
    """the throttle is scoped to captured warnings, not to logging generally"""
    warning_filter = _RepeatedWarningFilter(max_per_category=1)
    record = logging.LogRecord(name='optimalportfolios', level=logging.WARNING,
                               pathname=__file__, lineno=1, msg='anything', args=(),
                               exc_info=None)
    assert all(warning_filter.filter(record) for _ in range(5))


# --------------------------------------------------------------------------- #
# RunDiagnostics
# --------------------------------------------------------------------------- #
def make_run_diagnostics() -> RunDiagnostics:
    """A bundle with a few solves recorded, built without touching global logging."""
    rejections = SolverRejectionSummary()
    for _ in range(3):
        emit_to(rejections, solver_diag=make_diagnostic(accepted=True))
    emit_to(rejections, solver_diag=make_diagnostic(accepted=False, severity=logging.ERROR))
    warnings_summary = WarningSummary()
    emit_to(warnings_summary, level=logging.WARNING,
            message='/x.py:1: UserWarning: covariance is not positive definite')
    return RunDiagnostics(rejections=rejections, relaxations=RelaxationSummary(),
                          dropped_groups=DroppedGroupSummary(),
                          warnings_summary=warnings_summary)


def test_run_diagnostics_summary_concatenates_every_section() -> None:
    """one call produces the whole run report, one section per attached handler"""
    text = make_run_diagnostics().summary()
    assert 'solver outcomes this run' in text
    assert 'zero-loading groups dropped' in text
    assert 'captured warnings' in text


def test_run_diagnostics_to_frame_is_workbook_ready() -> None:
    """the tabular view carries the same counts, indexed by category and metric"""
    frame = make_run_diagnostics().to_frame()
    assert frame.index.names == ['category', 'metric']
    assert frame.loc[('batch_solver', 'solves'), 'value'] == 4
    assert frame.loc[('batch_solver', 'rejected'), 'value'] == 1
    assert frame.loc[('batch_solver', 'numerical_blowups'), 'value'] == 1
    assert frame.loc[('batch_solver', 'fallback_fraction'), 'value'] == pytest.approx(0.25)
    assert ('batch_warning', 'covariance is not positive definite') in frame.index


def test_run_diagnostics_omits_absent_sections() -> None:
    """the optional handlers are genuinely optional, not empty placeholders"""
    minimal = RunDiagnostics(rejections=SolverRejectionSummary(),
                             relaxations=RelaxationSummary())
    text = minimal.summary()
    assert 'captured warnings' not in text
    assert 'zero-loading groups dropped' not in text
    assert not any(category == 'batch_input'
                   for category, _ in minimal.to_frame().index)


def test_run_diagnostics_delegates_the_fallback_gate() -> None:
    """the bundle's gate is the rejection handler's gate, not a second implementation"""
    diagnostics = make_run_diagnostics()
    assert diagnostics.check_fallback_gate(max_fraction=0.50) is True
    assert diagnostics.check_fallback_gate(max_fraction=0.01) is False


def test_run_diagnostics_log_summary_emits_through_logging(caplog) -> None:
    """the summary is persisted through the configured handlers, not printed"""
    with caplog.at_level(logging.INFO):
        make_run_diagnostics().log_summary()
    assert 'run diagnostics summary' in caplog.text


# --------------------------------------------------------------------------- #
# configure_run_logging
# --------------------------------------------------------------------------- #
def test_configure_run_logging_installs_a_console_handler(restore_logging) -> None:
    """the plain call wires the console and returns nothing to close"""
    assert configure_run_logging(console_level=logging.ERROR) is None
    root = logging.getLogger()
    installed = [h for h in root.handlers
                 if getattr(h, '_optimalportfolios_run_handler', False)]
    assert len(installed) == 1
    assert installed[0].level == logging.ERROR


def test_configure_run_logging_writes_a_run_log_file(restore_logging, tmp_path) -> None:
    """a log path adds a file handler that receives the environment banner"""
    log_path = tmp_path / 'run.log'
    configure_run_logging(log_path=str(log_path), file_level=logging.INFO)
    logging.getLogger('optimalportfolios').warning('a rebalance fell back')
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = log_path.read_text(encoding='utf-8-sig')
    assert 'a rebalance fell back' in text


def test_configure_run_logging_is_idempotent(restore_logging) -> None:
    """calling twice replaces its own handlers rather than duplicating output"""
    configure_run_logging()
    configure_run_logging()
    root = logging.getLogger()
    installed = [h for h in root.handlers
                 if getattr(h, '_optimalportfolios_run_handler', False)]
    assert len(installed) == 1, "a second call duplicated the console handler"


def test_configure_run_logging_leaves_application_handlers_alone(restore_logging) -> None:
    """only handlers this helper installed are removed; the application keeps its own"""
    root = logging.getLogger()
    foreign = logging.NullHandler()
    root.addHandler(foreign)
    try:
        configure_run_logging()
        configure_run_logging()
        assert foreign in root.handlers
    finally:
        root.removeHandler(foreign)


def test_configure_run_logging_with_summary_returns_a_closable_bundle(
        restore_logging) -> None:
    """attach_summary wires the aggregation handlers and hands back their owner"""
    diagnostics = configure_run_logging(attach_summary=True)
    assert isinstance(diagnostics, RunDiagnostics)
    diag_logger = logging.getLogger('optimalportfolios.optimization.solver_diagnostics')
    cons_logger = logging.getLogger('optimalportfolios.optimization.constraints')
    # accepted solves are emitted at DEBUG, so both loggers must be lifted to see them
    assert diag_logger.level == logging.DEBUG
    assert cons_logger.level == logging.DEBUG
    assert diagnostics.rejections in diag_logger.handlers
    assert diagnostics.relaxations in cons_logger.handlers

    # a solve emitted on the real logger reaches the summary
    diag_logger.debug('accepted', extra={'solver_diag': make_diagnostic(accepted=True)})
    assert diagnostics.rejections.n_total == 1

    diagnostics.close()
    assert diagnostics.rejections not in diag_logger.handlers
    assert diagnostics.relaxations not in cons_logger.handlers


def test_configure_run_logging_captures_python_warnings(restore_logging) -> None:
    """warnings.warn is routed through logging so it lands in the run log too"""
    diagnostics = configure_run_logging(attach_summary=True, capture_warnings=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            warnings.warn('covariance is not positive definite', UserWarning)
        assert sum(diagnostics.warnings_summary.counts.values()) >= 1
    finally:
        diagnostics.close()


def test_log_environment_emits_a_reproducibility_banner(caplog) -> None:
    """the banner records solver and library versions for reproducibility"""
    with caplog.at_level(logging.INFO):
        log_environment(config_hash='abc123')
    assert caplog.text.strip() != ''
    assert 'abc123' in caplog.text


def test_log_environment_reports_a_missing_library_rather_than_failing(caplog,
                                                                       monkeypatch) -> None:
    """a library that will not import is recorded as 'n/a', not allowed to end the run

    The banner is diagnostic output at the very start of a run. An optional backend that is
    absent — or present but without a ``__version__`` — must degrade to a missing entry;
    raising here would abort a backtest before it had done anything.
    """
    def _no_import(name):
        """Stand in for an environment where none of the libraries can be imported."""
        raise ImportError(name)

    monkeypatch.setattr(importlib, 'import_module', _no_import)
    with caplog.at_level(logging.INFO):
        log_environment()
    assert 'clarabel=n/a' in caplog.text
    assert 'numpy=n/a' in caplog.text


# --------------------------------------------------------------------------- #
# InputContractSummary
# --------------------------------------------------------------------------- #
def make_contract_record(**overrides) -> InputContractRecord:
    """One clean per-solve input-contract record, with any finding switched on."""
    kwargs = dict(context='2024-03-31', ok=True, ill_conditioned=False,
                  cond=float('nan'), min_eig=float('nan'), collinear_pair=None,
                  groups=(), benchmarks=(), structural=(), covar_issues=(),
                  factorized=False, n_eigenvalues_floored=0,
                  stabilized_min_eig=float('nan'), stabilized_cond=float('nan'))
    kwargs.update(overrides)
    return InputContractRecord(**kwargs)


def test_input_contract_summary_says_so_when_nothing_was_recorded() -> None:
    """a summary of no solves is not a summary of a clean run — the two differ"""
    assert InputContractSummary().summary() == 'input contract: no solves recorded'


def test_input_contract_summary_reports_a_clean_run_explicitly() -> None:
    """solves recorded and nothing wrong with any of them is its own statement

    Without this line a clean run and a run whose handler silently tallied nothing produce
    the same output, which is exactly the failure this layer is meant to make visible.
    """
    handler = InputContractSummary()
    for _ in range(3):
        emit_to(handler, input_contract=make_contract_record())
    assert handler.summary() == 'input contract: no issues across 3 rebalances'


def test_input_contract_summary_names_the_structural_and_covariance_findings() -> None:
    """a box-vs-budget infeasibility and a broken covariance get one line each"""
    handler = InputContractSummary()
    emit_to(handler, input_contract=make_contract_record(
        ok=False, structural=('box caps sum to 0.75 < 1.0',)))
    emit_to(handler, input_contract=make_contract_record(
        ok=False, covar_issues=('non_finite',)))
    emit_to(handler, input_contract=make_contract_record())
    text = handler.summary()
    assert 'structural: box-vs-budget infeasible on 1/3 rebalances' in text
    assert 'covariance integrity: 1/3' in text


def test_run_diagnostics_tabulates_the_input_contract_section() -> None:
    """when the contract handler is attached, its counts join the workbook frame"""
    contract = InputContractSummary()
    emit_to(contract, input_contract=make_contract_record(
        ok=False, ill_conditioned=True, cond=1e14, min_eig=-1e-9,
        collinear_pair=('PE', 'HF'), benchmarks=((0, 'cap_exceeded'),),
        groups=(('Alternatives', 'floor_unreachable'),),
        structural=('box caps sum to 0.75 < 1.0',)))
    diagnostics = RunDiagnostics(rejections=SolverRejectionSummary(),
                                 relaxations=RelaxationSummary(), contract=contract)
    frame = diagnostics.to_frame()
    assert frame.loc[('batch_input', 'solves'), 'value'] == 1
    assert frame.loc[('batch_input', 'raw_ill_conditioned'), 'value'] == 1
    assert frame.loc[('batch_input', 'benchmark_outside_box'), 'value'] == 1
    assert 'input contract findings' in diagnostics.summary()
