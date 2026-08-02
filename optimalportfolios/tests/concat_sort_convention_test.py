"""
every axis=1 ``pd.concat`` in library code states ``sort=`` explicitly.

``pd.concat(objs, axis=1)`` joins the frames on their index, and whether the resulting union is
sorted has been changing under us. pandas 2.2 sorted the union of DatetimeIndexes whatever
``sort=`` said; pandas 3.0 honours an explicit ``sort=False`` and leaves the union in appearance
order; pandas 3.0 still sorts when no ``sort=`` is passed, under a ``Pandas4Warning`` announcing
that pandas 4 will not. A call that says nothing therefore means one thing today and another
after the next major release, in code where the difference is a scrambled time axis rather than
an error.

This bit for real. A sweep report built its panel with
``pd.concat([reference_navs, strategy_navs], axis=1, sort=False)``, the two legs sat on different
calendars, and the annual returns table died inside `qis` with
``ValueError: index must be monotonic increasing or decreasing``. `qis` sorts defensively now,
but the panel it was handed was still wrong, and the two frames concatenated here that join
non-identical date indexes by construction - the per-frequency residual blocks in
``estimate_lasso_factor_covar_data`` and the per-frequency excess returns in
``managers_alpha`` - would have failed the same way with nothing to catch them.

So every such call states what it wants:

- ``sort=True`` where the joined index is a DatetimeIndex - navs, signal scores, residuals.
  Chronological order is the meaning of the axis, and this is what pandas 2.2 did.
- ``sort=False`` where the joined index is a label index - assets, groups, mixture clusters.
  Row order there is the caller's, pandas has never sorted it, and sorting it would reorder a
  report table.

Only ``axis=1`` is covered. An ``axis=0`` concat joins on the columns, which in this package are
asset or statistic labels rather than dates, and pandas 4 does not change their handling.

To confirm this check can fail, drop ``sort=True`` from any concat in ``alphas/signals/``: the
call site is reported below by file, line and the object being concatenated. That was run before
this file was committed.
"""
# packages
import ast
from pathlib import Path
from typing import List, Tuple
# optimalportfolios
import optimalportfolios

PACKAGE_ROOT: Path = Path(optimalportfolios.__file__).parent

# directories whose contents are scripts rather than library code: an example is read by a user
# and a test states its own frames, so neither carries the convention
EXCLUDED_PARTS: Tuple[str, ...] = ('examples', 'tests', 'notebooks')


def _is_pd_concat(node: ast.Call) -> bool:
    """True for a ``pd.concat(...)`` call node"""
    func = node.func
    return (isinstance(func, ast.Attribute) and func.attr == 'concat'
            and isinstance(func.value, ast.Name) and func.value.id == 'pd')


def find_implicit_sort_sites() -> List[str]:
    """Return one line per axis=1 pd.concat call in library code that omits sort=."""
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob('*.py')):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.name.endswith(('_test.py', '_tests.py')):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_pd_concat(node):
                continue
            keywords = {kw.arg: kw for kw in node.keywords if kw.arg is not None}
            axis = keywords.get('axis')
            if axis is None or not isinstance(axis.value, ast.Constant):
                continue
            if axis.value.value not in (1, 'columns') or 'sort' in keywords:
                continue
            objs = ast.unparse(node.args[0]) if node.args else '<no positional objs>'
            rel = path.relative_to(PACKAGE_ROOT.parent).as_posix()
            offenders.append(f"{rel}:{node.lineno}: pd.concat({objs[:60]}, axis=1) omits sort=")
    return offenders


def test_axis1_concat_states_sort() -> None:
    """a concat that does not say whether it sorts means different things in pandas 3 and 4"""
    offenders = find_implicit_sort_sites()
    assert not offenders, (
            "axis=1 pd.concat without an explicit sort=; pass sort=True when the index is dates, "
            "sort=False when it is labels:\n" + '\n'.join(offenders))


if __name__ == '__main__':
    for offender in find_implicit_sort_sites():
        print(offender)
