Ongoing project applying cointegration and mean-reversion modeling to financial 
asset pairs, motivated by parallels between stochastic mean-reversion (Ornstein-
Uhlenbeck processes) and concepts from physics such as the damped harmonic 
oscillator.

Current analysis: GLD/SLV pair.

Current state:
- Engle-Granger test between two tickers: cointegration is tested using an OLS regression to calculate the spread, which
then is passed through an ADF test to determine cointegration to confidence values of 1%, 5% and 10%
- Half-life calculation: Calculates the half life of mean-reversion, which gives a timeframe for the spread to return to its usual value following
a deviation
- Rolling z-score: using a predetermined timeframe and the spread, a z-score is calculated for the spread. This data point reveals how many standard deviations
the spread has moved from its mean, returning buy and sell signals
- Refactored pipeline into reusable functions
- Sub-period stability check 
- Simple backtest using z-score thresholds

Future workpath:
- Transaction costs / slippage
- Walk-forward validation (train on early period, test on unseen later period)
- Performance metrics: Sharpe ratio, max drawdown
- Synthetic OU sanity check (recover known parameters from simulated data)
- Deflated Sharpe Ratio
### Multi-asset basket (Johansen test)
- Select a 3–5 asset basket with economic justification (e.g. precious metals: GLD/SLV/PPLT)
- Pairwise correlation prescreening
- Johansen cointegration test on the full basket
- Identify and validate the dominant cointegration vector
- ADF test on the resulting basket spread
- Rolling Johansen check across sub-periods
### Dynamic hedge ratio (Kalman filter)
- Formulate as a state-space model (hedge ratio as latent state)
- Initial implementation via `pykalman`
- Tune process/observation noise covariances
- Compare static (OLS) vs. dynamic (Kalman) hedge ratio over time
- Manual NumPy reimplementation of the Kalman filter
### Final comparison and writeup
- Backtest, walk-forward, Sharpe/drawdown for static vs. Kalman, pairs vs. basket
- Benchmark against buy-and-hold
- Structural break / regime stability analysis
- 
