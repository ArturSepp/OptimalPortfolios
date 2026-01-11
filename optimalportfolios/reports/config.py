"""
set common configuration for different reports
"""

from qis import PerfStat, PerfParams, BenchmarkReturnsQuantilesRegime

BENCHMARK_TABLE_COLUMNS2 = (PerfStat.PA_RETURN,
                            PerfStat.VOL,
                            PerfStat.SHARPE_EXCESS,
                            PerfStat.MAX_DD,
                            # PerfStat.MAX_DD_VOL,
                            PerfStat.BEST,
                            PerfStat.WORST,
                            PerfStat.SKEWNESS,
                            PerfStat.ALPHA_AN,
                            PerfStat.BETA,
                            PerfStat.R2)

DATE_FORMAT = '%d%b%Y'
FIG_SIZE1 = (14, 3)  # one figure for whole page
FIG_SIZE11 = (4.65, 2.35)  # one figure for half page
FIG_SIZE11_2 = (4.70, 0.95)
FIG_SIZE11_2a = (4.70, 0.6)


PERF_PARAMS = PerfParams(freq_vol='ME', freq_reg='ME', freq_drawdown='ME')

REGIME_CLASSIFIER = BenchmarkReturnsQuantilesRegime(freq='ME')

KWARGS = dict(fontsize=7,
              linewidth=0.5,
              digits_to_show=1, sharpe_digits=2,
              weight='normal',
              markersize=2,
              framealpha=0.8,
              date_format='%b-%y',
              trend_line_colors=['darkred'],
              trend_linewidth=2.0,
              x_date_freq='QE',
              short=True)

# for py blocks
margin_top = 0.0
margin_bottom = 0.0
line_height = 1.0
font_family = 'Calibri'

KWARGS_SUPTITLE = {'title_wrap': True, 'text_align': 'center', 'color': 'blue', 'font_size': "12px", 'font-weight': 'normal',
                   'title_level': 1, 'line_height': 0.7, 'inherit_cfg': False,
                   'margin_top': 0, 'margin_bottom': 0,
                   'font-family': 'sans-serif'}
KWARGS_TITLE = {'title_wrap': True, 'text_align': 'left', 'color': 'blue', 'font_size': "12px",
                'title_level': 2, 'line_height': line_height, 'inherit_cfg': False,
                'margin_top': margin_top,  'margin_bottom': margin_bottom,
                'font-family': font_family}
KWARGS_DESC = {'title_wrap': True, 'text_align': 'left', 'font_size': "12px", 'font-weight': 'normal',
               'title_level': 3, 'line_height': line_height, 'inherit_cfg': False,
               'margin_top': margin_top, 'margin_bottom': margin_bottom,
               'font-family': font_family}
KWARGS_TEXT = {'title_wrap': True, 'text_align': 'left', 'font_size': "12px", 'font-weight': 'normal',
               'title_level': 3, 'line_height': line_height, 'inherit_cfg': False,
               'margin_top': margin_top, 'margin_bottom': margin_bottom,
               'font-family': font_family}
KWARGS_FIG = {'title_wrap': True, 'text_align': 'left', 'font_size': "12px",
              'title_level': 3, 'line_height': line_height, 'inherit_cfg': False,
              'margin_top': margin_top, 'margin_bottom': margin_bottom,
              'font-family': font_family}
KWARGS_FOOTNOTE = {'title_wrap': True, 'text_align': 'left', 'font_size': "12px", 'font-weight': 'normal',
                   'title_level': 8, 'line_height': line_height, 'inherit_cfg': False,
                   'margin_top': 0, 'margin_bottom': 0,
                   'font-family': font_family}

RA_TABLE_FOOTNOTE = (u"\u002A" + f"Vol (annualized volatility) and Skew (Skeweness) are computed using daily returns, "
                                 f"Sharpe is computed assuming zero risk-free rate, "
                                 f"Max DD is maximum drawdown, "
                                 f"Best and Worst are the highest and lowest daily returns, "
                                 f"Alpha (annualized daily alpha), Beta, R2 (R squared) are estimated using regression "
                                 f"of daily returns explained by underlying coin")
