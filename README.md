# Financial Market Simulator

A Python-based simulation platform that models financial markets, incorporating various trading strategies, market dynamics, and economic indicators. This tool serves as a testbed for algorithmic trading strategies and risk assessment.

## Features

### Market Modeling
- Different market states (bull, bear, volatile, stable)
- Economic indicators (GDP growth, unemployment, interest rates, inflation, etc.)
- Realistic asset price movements based on market conditions and volatility

### Trading Strategies
- **Moving Average Crossover**: Trades based on short and long-term moving average crossovers
- **Mean Reversion**: Buys when prices are significantly below their mean and sells when above
- **Trend Following**: Identifies and follows market momentum
- **Macroeconomic Strategy**: Trades based on economic indicators

### Order Types
- Market Orders: Executed immediately at current price
- Limit Orders: Executed only at specified price or better
- Stop Orders: Activated when price reaches a certain level

### Portfolio Management
- Cash and holdings tracking
- Trade history recording
- Performance metrics calculation (returns, volatility, Sharpe ratio, drawdown)

### Simulation Capabilities
- Configurable time period simulations
- Random market shocks
- Results analysis and visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-market-simulator.git
cd financial-market-simulator

# Install required packages
pip install numpy pandas matplotlib
```

## Quick Start

```python
from market_simulator import MarketSimulator, MovingAverageStrategy, MeanReversionStrategy

# Initialize simulator
simulator = MarketSimulator(seed=42)

# Add assets (symbol, initial price, volatility)
simulator.add_asset("AAPL", 150.0, 0.015)
simulator.add_asset("MSFT", 300.0, 0.012)
simulator.add_asset("GOOGL", 2800.0, 0.018)

# Add strategies
simulator.add_strategy(MovingAverageStrategy("MA Crossover", 10, 50))
simulator.add_strategy(MeanReversionStrategy("Mean Reversion", 20, 1.5))

# Run simulation for 252 trading days (~ 1 year)
results = simulator.run_simulation(252, verbose=True)

# Visualize results
simulator.plot_results(results)
```

## Example Output

Running the simulation produces performance statistics and visualizations:

```
Starting simulation for 252 days...
Day 0: Portfolio value = $100000.00
Day 30: Portfolio value = $101234.56
...
Day 240: Portfolio value = $112345.67

Simulation complete.
Final portfolio value: $115678.90
Total return: 15.68%
Annualized return: 15.68%
Annualized volatility: 10.25%
Sharpe ratio: 1.34
Maximum drawdown: 8.76%
```

The visualization includes:
- Portfolio value over time
- Normalized asset prices
- Portfolio drawdown

## Advanced Usage

### Creating Custom Strategies

```python
from market_simulator import TradingStrategy

class MyCustomStrategy(TradingStrategy):
    def __init__(self, name, parameter1=10, parameter2=20):
        super().__init__(name)
        self.parameter1 = parameter1
        self.parameter2 = parameter2
    
    def generate_signal(self, market, portfolio, asset):
        # Your strategy logic here
        # Return is_buy (bool), order_type (OrderType), price (float or None)
        return True, OrderType.MARKET, None

# Add to simulator
simulator.add_strategy(MyCustomStrategy("My Strategy", 15, 25))
```

### Customizing Economic Conditions

```python
# Set initial economic indicators
simulator.market.economic_indicators = {
    "gdp_growth": 3.0,        # percent
    "unemployment": 4.0,       # percent
    "interest_rate": 3.0,      # percent
    "inflation": 2.5,          # percent
    "consumer_confidence": 110.0,  # index
    "vix": 18.0                # volatility index
}
```

### Running Multiple Simulations

```python
results_list = []
for seed in range(10):
    simulator = MarketSimulator(seed=seed)
    # Add assets and strategies
    results = simulator.run_simulation(252, verbose=False)
    results_list.append(results)

# Analyze distribution of outcomes
returns = [r["metrics"]["total_return"] for r in results_list]
print(f"Average return: {np.mean(returns):.2%}")
print(f"Return standard deviation: {np.std(returns):.2%}")
```

## Project Structure

- `market_simulator.py`: Main simulator code
- `examples/`: Example scripts demonstrating different use cases
- `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks with analyses and tutorials

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- Modern portfolio theory
- Algorithmic trading literature
- Financial market simulation research

## Contact

Your Name - [@yourusername](https://twitter.com/yourusername) - email@example.com

Project Link: [https://github.com/yourusername/financial-market-simulator](https://github.com/yourusername/financial-market-simulator)
