import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Dict, Callable, Tuple, Optional, Union
import datetime as dt
import random

class MarketState(Enum):
    BULL = "bull"
    BEAR = "bear"
    VOLATILE = "volatile"
    STABLE = "stable"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class Order:
    def __init__(self, 
                 symbol: str, 
                 quantity: int, 
                 order_type: OrderType, 
                 is_buy: bool, 
                 price: Optional[float] = None,
                 expiration: Optional[dt.datetime] = None):
        """
        Initialize an order
        
        Args:
            symbol: Stock ticker symbol
            quantity: Number of shares
            order_type: Market, limit, or stop order
            is_buy: True for buy, False for sell
            price: Relevant for limit and stop orders
            expiration: Optional expiration date for the order
        """
        self.symbol = symbol
        self.quantity = quantity
        self.order_type = order_type
        self.is_buy = is_buy
        self.price = price
        self.expiration = expiration
        self.status = "pending"
        self.filled_price = None
        self.timestamp = dt.datetime.now()
    
    def __str__(self):
        action = "BUY" if self.is_buy else "SELL"
        return f"{action} {self.quantity} {self.symbol} @ {self.price if self.price else 'MARKET'}"

class Asset:
    def __init__(self, symbol: str, initial_price: float, volatility: float = 0.01):
        """
        Initialize an asset
        
        Args:
            symbol: Stock ticker symbol
            initial_price: Starting price
            volatility: Daily volatility (standard deviation)
        """
        self.symbol = symbol
        self.price = initial_price
        self.volatility = volatility
        self.price_history = [initial_price]
        self.returns_history = [0]
        
    def update_price(self, market_state: MarketState, economic_indicators: Dict[str, float]) -> float:
        """
        Update the asset price based on market state and economic indicators
        
        Args:
            market_state: Current market state (bull, bear, etc.)
            economic_indicators: Dict of economic indicators and their values
            
        Returns:
            New price
        """
        # Base drift based on market state
        if market_state == MarketState.BULL:
            drift = 0.0005 + 0.0002 * economic_indicators.get("gdp_growth", 0)
        elif market_state == MarketState.BEAR:
            drift = -0.0005 - 0.0002 * economic_indicators.get("unemployment", 0)
        elif market_state == MarketState.VOLATILE:
            drift = 0.0001 * economic_indicators.get("consumer_confidence", 0) - 0.0001
        else:  # STABLE
            drift = 0.0001
        
        # Adjust volatility based on VIX-like indicator
        current_volatility = self.volatility * economic_indicators.get("vix", 1.0)
        
        # Generate random return
        daily_return = np.random.normal(drift, current_volatility)
        
        # Apply interest rate effect
        interest_rate = economic_indicators.get("interest_rate", 0.02)
        daily_return -= 0.01 * (interest_rate - 0.02)  # Higher rates typically reduce equity returns
        
        # Update price
        self.price *= (1 + daily_return)
        
        # Store history
        self.price_history.append(self.price)
        self.returns_history.append(daily_return)
        
        return self.price

class TradingStrategy:
    def __init__(self, name: str):
        """
        Base class for trading strategies
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.performance = []
    
    def generate_signal(self, 
                       market: 'Market', 
                       portfolio: 'Portfolio', 
                       asset: Asset) -> Tuple[bool, OrderType, float]:
        """
        Generate trading signal
        
        Args:
            market: Market instance
            portfolio: Portfolio instance
            asset: Asset to generate signal for
            
        Returns:
            Tuple of (is_buy, order_type, price)
        """
        raise NotImplementedError("Subclasses must implement this method")

class MovingAverageStrategy(TradingStrategy):
    def __init__(self, name: str, short_window: int = 10, long_window: int = 50):
        """
        Moving average crossover strategy
        
        Args:
            name: Strategy name
            short_window: Short moving average window
            long_window: Long moving average window
        """
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signal(self, 
                       market: 'Market', 
                       portfolio: 'Portfolio', 
                       asset: Asset) -> Tuple[bool, OrderType, Optional[float]]:
        """
        Generate trading signal based on moving average crossover
        
        Args:
            market: Market instance
            portfolio: Portfolio instance
            asset: Asset to generate signal for
            
        Returns:
            Tuple of (is_buy, order_type, price)
        """
        if len(asset.price_history) < self.long_window:
            return None, None, None
        
        prices = asset.price_history
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        
        # Previous values
        if len(prices) > self.long_window + 1:
            prev_short_ma = np.mean(prices[-(self.short_window+1):-1])
            prev_long_ma = np.mean(prices[-(self.long_window+1):-1])
            
            # Crossover detection
            if prev_short_ma < prev_long_ma and short_ma > long_ma:
                # Golden cross - buy signal
                return True, OrderType.MARKET, None
            elif prev_short_ma > prev_long_ma and short_ma < long_ma:
                # Death cross - sell signal
                return False, OrderType.MARKET, None
        
        return None, None, None

class MeanReversionStrategy(TradingStrategy):
    def __init__(self, name: str, window: int = 20, z_threshold: float = 1.5):
        """
        Mean reversion strategy
        
        Args:
            name: Strategy name
            window: Lookback window
            z_threshold: Z-score threshold for trading signals
        """
        super().__init__(name)
        self.window = window
        self.z_threshold = z_threshold
    
    def generate_signal(self, 
                       market: 'Market', 
                       portfolio: 'Portfolio', 
                       asset: Asset) -> Tuple[bool, OrderType, Optional[float]]:
        """
        Generate trading signal based on mean reversion
        
        Args:
            market: Market instance
            portfolio: Portfolio instance
            asset: Asset to generate signal for
            
        Returns:
            Tuple of (is_buy, order_type, price)
        """
        if len(asset.price_history) < self.window:
            return None, None, None
        
        prices = asset.price_history[-self.window:]
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        if std_price == 0:
            return None, None, None
            
        current_price = asset.price
        z_score = (current_price - mean_price) / std_price
        
        if z_score < -self.z_threshold:
            # Price is significantly below mean - buy
            return True, OrderType.MARKET, None
        elif z_score > self.z_threshold:
            # Price is significantly above mean - sell
            return False, OrderType.MARKET, None
        
        return None, None, None

class TrendFollowingStrategy(TradingStrategy):
    def __init__(self, name: str, window: int = 50, momentum_threshold: float = 0.05):
        """
        Trend following strategy
        
        Args:
            name: Strategy name
            window: Lookback window
            momentum_threshold: Minimum momentum for signal
        """
        super().__init__(name)
        self.window = window
        self.momentum_threshold = momentum_threshold
    
    def generate_signal(self, 
                       market: 'Market', 
                       portfolio: 'Portfolio', 
                       asset: Asset) -> Tuple[bool, OrderType, Optional[float]]:
        """
        Generate trading signal based on trend following
        
        Args:
            market: Market instance
            portfolio: Portfolio instance
            asset: Asset to generate signal for
            
        Returns:
            Tuple of (is_buy, order_type, price)
        """
        if len(asset.price_history) < self.window:
            return None, None, None
        
        prices = asset.price_history[-self.window:]
        momentum = (prices[-1] / prices[0]) - 1
        
        if momentum > self.momentum_threshold:
            # Strong positive momentum - buy
            return True, OrderType.MARKET, None
        elif momentum < -self.momentum_threshold:
            # Strong negative momentum - sell
            return False, OrderType.MARKET, None
        
        return None, None, None

class MacroeconomicStrategy(TradingStrategy):
    def __init__(self, name: str):
        """
        Strategy based on macroeconomic indicators
        
        Args:
            name: Strategy name
        """
        super().__init__(name)
    
    def generate_signal(self, 
                       market: 'Market', 
                       portfolio: 'Portfolio', 
                       asset: Asset) -> Tuple[bool, OrderType, Optional[float]]:
        """
        Generate trading signal based on macroeconomic indicators
        
        Args:
            market: Market instance
            portfolio: Portfolio instance
            asset: Asset to generate signal for
            
        Returns:
            Tuple of (is_buy, order_type, price)
        """
        # Get economic indicators
        indicators = market.economic_indicators
        
        # Simple logic: buy when good economic conditions, sell otherwise
        gdp_growth = indicators.get("gdp_growth", 0)
        unemployment = indicators.get("unemployment", 5)
        interest_rate = indicators.get("interest_rate", 2)
        inflation = indicators.get("inflation", 2)
        
        economic_score = (gdp_growth * 0.4) - (unemployment * 0.2) - (max(0, interest_rate - 3) * 0.2) - (max(0, inflation - 2) * 0.2)
        
        if economic_score > 1:
            # Favorable economic conditions - buy
            return True, OrderType.MARKET, None
        elif economic_score < -1:
            # Unfavorable economic conditions - sell
            return False, OrderType.MARKET, None
        
        return None, None, None

class Portfolio:
    def __init__(self, initial_cash: float = 100000.0):
        """
        Initialize a portfolio
        
        Args:
            initial_cash: Starting cash balance
        """
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.holdings = {}  # symbol -> quantity
        self.trades = []
        self.performance_history = [initial_cash]
        self.returns_history = [0]
    
    def place_order(self, market: 'Market', order: Order) -> bool:
        """
        Place an order in the market
        
        Args:
            market: Market instance
            order: Order to place
            
        Returns:
            Whether the order was placed successfully
        """
        # Check if we have enough cash/holdings
        if order.is_buy:
            asset = market.assets.get(order.symbol)
            if not asset:
                return False
            
            estimated_cost = order.quantity * asset.price
            if estimated_cost > self.cash:
                return False
        else:
            current_holdings = self.holdings.get(order.symbol, 0)
            if current_holdings < order.quantity:
                return False
        
        # Add order to market's order book
        market.add_order(order)
        return True
    
    def update_performance(self, market: 'Market') -> float:
        """
        Update the portfolio performance
        
        Args:
            market: Market instance
            
        Returns:
            Current portfolio value
        """
        portfolio_value = self.cash
        
        # Add value of all holdings
        for symbol, quantity in self.holdings.items():
            asset = market.assets.get(symbol)
            if asset:
                portfolio_value += quantity * asset.price
        
        # Store history
        prev_value = self.performance_history[-1]
        self.performance_history.append(portfolio_value)
        
        # Calculate return
        if prev_value > 0:
            daily_return = (portfolio_value / prev_value) - 1
        else:
            daily_return = 0
        self.returns_history.append(daily_return)
        
        return portfolio_value
    
    def process_fill(self, order: Order, fill_price: float) -> None:
        """
        Process a filled order
        
        Args:
            order: Filled order
            fill_price: Price at which the order was filled
        """
        if order.is_buy:
            # Update cash
            cost = order.quantity * fill_price
            self.cash -= cost
            
            # Update holdings
            current_holdings = self.holdings.get(order.symbol, 0)
            self.holdings[order.symbol] = current_holdings + order.quantity
        else:
            # Update cash
            proceeds = order.quantity * fill_price
            self.cash += proceeds
            
            # Update holdings
            current_holdings = self.holdings.get(order.symbol, 0)
            self.holdings[order.symbol] = current_holdings - order.quantity
        
        # Record trade
        self.trades.append({
            "timestamp": dt.datetime.now(),
            "symbol": order.symbol,
            "quantity": order.quantity,
            "is_buy": order.is_buy,
            "price": fill_price,
            "value": order.quantity * fill_price
        })
    
    def get_position_value(self, market: 'Market', symbol: str) -> float:
        """
        Get the value of a position
        
        Args:
            market: Market instance
            symbol: Asset symbol
            
        Returns:
            Position value
        """
        quantity = self.holdings.get(symbol, 0)
        asset = market.assets.get(symbol)
        if not asset:
            return 0
        
        return quantity * asset.price
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics
        
        Returns:
            Dict of metrics
        """
        # Total return
        total_return = (self.performance_history[-1] / self.initial_cash) - 1
        
        # Calculate annualized return
        n_days = len(self.performance_history) - 1
        if n_days > 0:
            annualized_return = ((1 + total_return) ** (252 / n_days)) - 1
        else:
            annualized_return = 0
        
        # Calculate volatility
        if len(self.returns_history) > 1:
            daily_volatility = np.std(self.returns_history[1:])  # Skip the first 0
            annualized_volatility = daily_volatility * np.sqrt(252)
        else:
            daily_volatility = 0
            annualized_volatility = 0
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        if annualized_volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        else:
            sharpe_ratio = 0
        
        # Calculate drawdown
        peak = np.maximum.accumulate(self.performance_history)
        drawdown = (peak - self.performance_history) / peak
        max_drawdown = np.max(drawdown)
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "daily_volatility": daily_volatility,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }

class Market:
    def __init__(self):
        """
        Initialize a market
        """
        self.assets = {}
        self.order_book = []
        self.market_state = MarketState.STABLE
        self.economic_indicators = {
            "gdp_growth": 2.0,  # percent
            "unemployment": 5.0,  # percent
            "interest_rate": 2.0,  # percent
            "inflation": 2.0,  # percent
            "consumer_confidence": 100.0,  # index
            "vix": 15.0  # volatility index
        }
        self.date = dt.datetime.now()
        self.trading_day = 0
    
    def add_asset(self, asset: Asset) -> None:
        """
        Add an asset to the market
        
        Args:
            asset: Asset to add
        """
        self.assets[asset.symbol] = asset
    
    def add_order(self, order: Order) -> None:
        """
        Add an order to the order book
        
        Args:
            order: Order to add
        """
        self.order_book.append(order)
    
    def update_market_state(self) -> None:
        """
        Update the market state based on economic indicators
        """
        gdp_growth = self.economic_indicators["gdp_growth"]
        unemployment = self.economic_indicators["unemployment"]
        vix = self.economic_indicators["vix"]
        
        # Simple state transition logic
        if gdp_growth > 3.0 and unemployment < 4.0:
            self.market_state = MarketState.BULL
        elif gdp_growth < 1.0 or unemployment > 6.0:
            self.market_state = MarketState.BEAR
        elif vix > 25.0:
            self.market_state = MarketState.VOLATILE
        else:
            self.market_state = MarketState.STABLE
    
    def update_economic_indicators(self, shock: bool = False) -> None:
        """
        Update economic indicators
        
        Args:
            shock: Whether to introduce a market shock
        """
        # Add random noise to indicators
        self.economic_indicators["gdp_growth"] += np.random.normal(0, 0.05)
        self.economic_indicators["unemployment"] += np.random.normal(0, 0.1)
        self.economic_indicators["interest_rate"] += np.random.normal(0, 0.05)
        self.economic_indicators["inflation"] += np.random.normal(0, 0.05)
        self.economic_indicators["consumer_confidence"] += np.random.normal(0, 1.0)
        self.economic_indicators["vix"] += np.random.normal(0, 0.5)
        
        # Keep indicators in reasonable ranges
        self.economic_indicators["gdp_growth"] = max(-5.0, min(10.0, self.economic_indicators["gdp_growth"]))
        self.economic_indicators["unemployment"] = max(2.0, min(15.0, self.economic_indicators["unemployment"]))
        self.economic_indicators["interest_rate"] = max(0.0, min(10.0, self.economic_indicators["interest_rate"]))
        self.economic_indicators["inflation"] = max(0.0, min(15.0, self.economic_indicators["inflation"]))
        self.economic_indicators["consumer_confidence"] = max(50.0, min(150.0, self.economic_indicators["consumer_confidence"]))
        self.economic_indicators["vix"] = max(5.0, min(50.0, self.economic_indicators["vix"]))
        
        # Market shock
        if shock:
            shock_type = random.choice(["recession", "boom", "crisis"])
            
            if shock_type == "recession":
                self.economic_indicators["gdp_growth"] -= 2.0
                self.economic_indicators["unemployment"] += 2.0
                self.economic_indicators["consumer_confidence"] -= 20.0
                self.economic_indicators["vix"] += 10.0
            elif shock_type == "boom":
                self.economic_indicators["gdp_growth"] += 2.0
                self.economic_indicators["unemployment"] -= 1.0
                self.economic_indicators["consumer_confidence"] += 15.0
            elif shock_type == "crisis":
                self.economic_indicators["vix"] += 20.0
                self.economic_indicators["consumer_confidence"] -= 30.0
    
    def process_orders(self, portfolio: Portfolio) -> None:
        """
        Process all orders in the order book
        
        Args:
            portfolio: Portfolio to update
        """
        fulfilled_orders = []
        
        for order in self.order_book:
            asset = self.assets.get(order.symbol)
            if not asset:
                continue
            
            current_price = asset.price
            
            # Check if order can be executed
            can_execute = False
            
            if order.order_type == OrderType.MARKET:
                can_execute = True
                fill_price = current_price
            elif order.order_type == OrderType.LIMIT:
                if order.is_buy and current_price <= order.price:
                    can_execute = True
                    fill_price = current_price
                elif not order.is_buy and current_price >= order.price:
                    can_execute = True
                    fill_price = current_price
            elif order.order_type == OrderType.STOP:
                if order.is_buy and current_price >= order.price:
                    can_execute = True
                    fill_price = current_price
                elif not order.is_buy and current_price <= order.price:
                    can_execute = True
                    fill_price = current_price
            
            # Check if order has expired
            if order.expiration and order.expiration < self.date:
                fulfilled_orders.append(order)
                continue
            
            # Execute order
            if can_execute:
                order.status = "filled"
                order.filled_price = fill_price
                portfolio.process_fill(order, fill_price)
                fulfilled_orders.append(order)
        
        # Remove fulfilled orders
        self.order_book = [order for order in self.order_book if order not in fulfilled_orders]
    
    def simulate_day(self, portfolio: Portfolio, strategies: List[TradingStrategy]) -> None:
        """
        Simulate a trading day
        
        Args:
            portfolio: Portfolio to simulate with
            strategies: List of trading strategies to use
        """
        # Update economic indicators (5% chance of shock)
        shock = random.random() < 0.05
        self.update_economic_indicators(shock)
        
        # Update market state
        self.update_market_state()
        
        # Update asset prices
        for symbol, asset in self.assets.items():
            asset.update_price(self.market_state, self.economic_indicators)
        
        # Apply trading strategies
        for strategy in strategies:
            for symbol, asset in self.assets.items():
                signal = strategy.generate_signal(self, portfolio, asset)
                is_buy, order_type, price = signal
                
                if is_buy is not None:
                    # Determine order quantity (simple logic: use 5% of cash or holdings)
                    if is_buy:
                        # Buy order
                        available_cash = portfolio.cash * 0.05
                        quantity = int(available_cash / asset.price)
                        if quantity > 0:
                            order = Order(symbol, quantity, order_type, is_buy, price)
                            portfolio.place_order(self, order)
                    else:
                        # Sell order
                        current_holdings = portfolio.holdings.get(symbol, 0)
                        quantity = int(current_holdings * 0.05)
                        if quantity > 0:
                            order = Order(symbol, quantity, order_type, is_buy, price)
                            portfolio.place_order(self, order)
        
        # Process orders
        self.process_orders(portfolio)
        
        # Update portfolio performance
        portfolio.update_performance(self)
        
        # Update date
        self.date += dt.timedelta(days=1)
        self.trading_day += 1

class MarketSimulator:
    def __init__(self, seed: int = None):
        """
        Initialize a market simulator
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.market = Market()
        self.portfolio = Portfolio()
        self.strategies = []
    
    def add_asset(self, symbol: str, initial_price: float, volatility: float = 0.01) -> None:
        """
        Add an asset to the market
        
        Args:
            symbol: Asset symbol
            initial_price: Initial price
            volatility: Price volatility
        """
        asset = Asset(symbol, initial_price, volatility)
        self.market.add_asset(asset)
    
    def add_strategy(self, strategy: TradingStrategy) -> None:
        """
        Add a trading strategy
        
        Args:
            strategy: Trading strategy
        """
        self.strategies.append(strategy)
    
    def run_simulation(self, days: int, verbose: bool = True) -> Dict:
        """
        Run the simulation
        
        Args:
            days: Number of days to simulate
            verbose: Whether to print progress
            
        Returns:
            Dict of simulation results
        """
        if verbose:
            print(f"Starting simulation for {days} days...")
        
        for day in range(days):
            self.market.simulate_day(self.portfolio, self.strategies)
            
            if verbose and day % 30 == 0:
                portfolio_value = self.portfolio.performance_history[-1]
                print(f"Day {day}: Portfolio value = ${portfolio_value:.2f}")
        
        # Calculate metrics
        metrics = self.portfolio.calculate_metrics()
        
        if verbose:
            print("\nSimulation complete.")
            print(f"Final portfolio value: ${self.portfolio.performance_history[-1]:.2f}")
            print(f"Total return: {metrics['total_return']:.2%}")
            print(f"Annualized return: {metrics['annualized_return']:.2%}")
            print(f"Annualized volatility: {metrics['annualized_volatility']:.2%}")
            print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Maximum drawdown: {metrics['max_drawdown']:.2%}")
        
        # Prepare results
        results = {
            "portfolio_history": self.portfolio.performance_history,
            "asset_prices": {symbol: asset.price_history for symbol, asset in self.market.assets.items()},
            "economic_indicators": self.market.economic_indicators,
            "metrics": metrics,
            "trades": self.portfolio.trades
        }
        
        return results
    
    def plot_results(self, results: Dict) -> None:
        """
        Plot simulation results
        
        Args:
            results: Simulation results
        """
        # Create a figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot portfolio value
        axs[0].plot(results["portfolio_history"], label="Portfolio Value")
        axs[0].set_title("Portfolio Value Over Time")
        axs[0].set_xlabel("Days")
        axs[0].set_ylabel("Value ($)")
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot asset prices
        for symbol, prices in results["asset_prices"].items():
            normalized_prices = [price / prices[0] for price in prices]
            axs[1].plot(normalized_prices, label=symbol)
        axs[1].set_title("Normalized Asset Prices Over Time")
        axs[1].set_xlabel("Days")
        axs[1].set_ylabel("Normalized Price")
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot drawdown
        peak = np.maximum.accumulate(results["portfolio_history"])
        drawdown = (peak - results["portfolio_history"]) / peak
        axs[2].fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        axs[2].set_title("Portfolio Drawdown")
        axs[2].set_xlabel("Days")
        axs[2].set_ylabel("Drawdown (%)")
        axs[2].set_ylim(0, max(drawdown) * 1.1)
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
def run_example():
    # Initialize simulator
    simulator = MarketSimulator(seed=42)
    
    # Add assets
    simulator.add_asset("AAPL", 150.0, 0.015)
    simulator.add_asset("MSFT", 300.0, 0.012)
    simulator.add_asset("GOOGL", 2800.0, 0.018)
    simulator.add_asset("AMZN", 3500.0, 0.02)
    simulator.add_asset("TSLA", 900.0, 0.03)
    
    # Add strategies
    simulator.add_strategy(MovingAverageStrategy("MA Crossover", 10, 50))
    simulator.add_strategy(MeanReversionStrategy("Mean Reversion", 20, 1.5))
    simulator.add_strategy(TrendFollowingStrategy("Trend Following", 50, 0.05))
    simulator.add_strategy(MacroeconomicStrategy("Macro Strategy"))
    
    # Run simulation
    results = simulator.run_simulation(252, verbose=True)  # One trading year
    
    # Plot results
    simulator.plot_results(results)
    
    return simulator, results

if __name__ == "__main__":
    simulator, results = run_example()
