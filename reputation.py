

import numpy as np
from typing import List, Optional

def simple_mean(log_loss: float, old_reputation: float) -> float:
    """
    Simple average between current log loss (converted to 0-100) and old reputation.
    
    Args:
        log_loss: Current log loss value
        old_reputation: Previous reputation score (0-100)
    
    Returns:
        New reputation score (0-100)
    """
    current_score = log_loss * 100  # Convert log loss to 0-100 scale
    new_reputation = (current_score + old_reputation) / 2
    return np.clip(new_reputation, 0, 100)

def weighted_average(log_loss: float, old_reputation: float, current_weight: float = 0.5) -> float:
    """
    Weighted average between current log loss and old reputation.
    
    Args:
        log_loss: Current log loss value
        old_reputation: Previous reputation score (0-100)
        current_weight: Weight for current log loss (0-1), rest goes to old reputation
    
    Returns:
        New reputation score (0-100)
    """
    current_score = log_loss * 100  # Convert log loss to 0-100 scale
    old_weight = 1 - current_weight
    new_reputation = (current_weight * current_score) + (old_weight * old_reputation)
    return np.clip(new_reputation, 0, 100)

def exponential_moving_average(log_loss: float, old_reputation: float, alpha: float = 0.3) -> float:
    """
    Exponential moving average for reputation calculation.
    More recent values have higher influence.
    
    Args:
        log_loss: Current log loss value
        old_reputation: Previous reputation score (0-100)
        alpha: Smoothing factor (0-1), higher = more weight to current value
    
    Returns:
        New reputation score (0-100)
    """
    current_score = log_loss * 100  # Convert log loss to 0-100 scale
    new_reputation = alpha * current_score + (1 - alpha) * old_reputation
    return np.clip(new_reputation, 0, 100)

def reputation_history_weighted(log_loss: float, reputation_history: List[float], 
                               current_weight: float = 0.4, 
                               recent_weight: float = 0.3,
                               historical_weight: float = 0.3) -> float:
    """
    Calculate reputation using weighted combination of current, recent, and historical performance.
    
    Args:
        log_loss: Current log loss value
        reputation_history: List of previous reputation scores
        current_weight: Weight for current log loss
        recent_weight: Weight for recent reputation (last 2-3 values)
        historical_weight: Weight for older reputation values
    
    Returns:
        New reputation score (0-100)
    """
    current_score = log_loss * 100
    
    if len(reputation_history) == 0:
        return np.clip(current_score, 0, 100)
    
    # Calculate recent average (last 2-3 values)
    recent_count = min(3, len(reputation_history))
    recent_avg = np.mean(reputation_history[-recent_count:])
    
    # Calculate historical average (all previous values)
    historical_avg = np.mean(reputation_history)
    
    # Weighted combination
    new_reputation = (current_weight * current_score + 
                     recent_weight * recent_avg + 
                     historical_weight * historical_avg)
    
    return np.clip(new_reputation, 0, 100)

def momentum_based(log_loss: float, reputation_history: List[float], 
                  momentum_factor: float = 0.2) -> float:
    """
    Calculate reputation with momentum consideration.
    If reputation is improving, give bonus. If declining, apply penalty.
    
    Args:
        log_loss: Current log loss value
        reputation_history: List of previous reputation scores
        momentum_factor: How much momentum affects the calculation (0-1)
    
    Returns:
        New reputation score (0-100)
    """
    current_score = log_loss * 100
    
    if len(reputation_history) < 2:
        return np.clip(current_score, 0, 100)
    
    # Calculate momentum (trend over last few values)
    recent_values = reputation_history[-3:] if len(reputation_history) >= 3 else reputation_history
    if len(recent_values) >= 2:
        momentum = recent_values[-1] - recent_values[0]
    else:
        momentum = 0
    
    # Apply momentum adjustment
    momentum_adjustment = momentum * momentum_factor
    new_reputation = current_score + momentum_adjustment
    
    return np.clip(new_reputation, 0, 100)

def decay_based(log_loss: float, reputation_history: List[float], 
               decay_factor: float = 0.1) -> float:
    """
    Calculate reputation with exponential decay for older values.
    More recent values have exponentially higher influence.
    
    Args:
        log_loss: Current log loss value
        reputation_history: List of previous reputation scores
        decay_factor: Decay rate for older values (0-1)
    
    Returns:
        New reputation score (0-100)
    """
    current_score = log_loss * 100
    
    if len(reputation_history) == 0:
        return np.clip(current_score, 0, 100)
    
    # Calculate weighted average with exponential decay
    weights = []
    values = []
    
    # Add current value with weight 1.0
    weights.append(1.0)
    values.append(current_score)
    
    # Add historical values with decaying weights
    for i, hist_value in enumerate(reversed(reputation_history)):
        weight = (1 - decay_factor) ** (i + 1)
        weights.append(weight)
        values.append(hist_value)
    
    # Calculate weighted average
    weights = np.array(weights)
    values = np.array(values)
    new_reputation = np.average(values, weights=weights)
    
    return np.clip(new_reputation, 0, 100)

def percentile_based(log_loss: float, reputation_history: List[float], 
                   percentile: float = 75.0) -> float:
    """
    Calculate reputation based on percentile of historical performance.
    Current score is compared to historical distribution.
    
    Args:
        log_loss: Current log loss value
        reputation_history: List of previous reputation scores
        percentile: Percentile threshold for comparison (0-100)
    
    Returns:
        New reputation score (0-100)
    """
    current_score = log_loss * 100
    
    if len(reputation_history) == 0:
        return np.clip(current_score, 0, 100)
    
    # Calculate percentile threshold
    threshold = np.percentile(reputation_history, percentile)
    
    # If current score is better than threshold, boost it
    if current_score > threshold:
        boost_factor = 1.1  # 10% boost
        new_reputation = min(current_score * boost_factor, 100)
    else:
        # If below threshold, use weighted average
        new_reputation = 0.7 * current_score + 0.3 * np.mean(reputation_history)
    
    return np.clip(new_reputation, 0, 100)

def adaptive_weighted(log_loss: float, reputation_history: List[float], 
                    volatility_threshold: float = 10.0) -> float:
    """
    Adaptive reputation calculation that adjusts weights based on reputation volatility.
    If reputation is stable, use more historical data. If volatile, focus on recent values.
    
    Args:
        log_loss: Current log loss value
        reputation_history: List of previous reputation scores
        volatility_threshold: Threshold for determining volatility
    
    Returns:
        New reputation score (0-100)
    """
    current_score = log_loss * 100
    
    if len(reputation_history) < 2:
        return np.clip(current_score, 0, 100)
    
    # Calculate volatility (standard deviation of recent values)
    recent_values = reputation_history[-5:] if len(reputation_history) >= 5 else reputation_history
    volatility = np.std(recent_values)
    
    # Adjust weights based on volatility
    if volatility > volatility_threshold:
        # High volatility: focus more on current value
        current_weight = 0.7
        historical_weight = 0.3
    else:
        # Low volatility: use more historical data
        current_weight = 0.4
        historical_weight = 0.6
    
    historical_avg = np.mean(reputation_history)
    new_reputation = (current_weight * current_score + 
                     historical_weight * historical_avg)
    
    return np.clip(new_reputation, 0, 100)

# ===========================================================
# Utility Functions
# ===========================================================

def get_reputation_trend(reputation_history: List[float]) -> str:
    """
    Analyze reputation trend over time.
    
    Args:
        reputation_history: List of reputation scores
    
    Returns:
        String describing the trend: 'improving', 'declining', 'stable'
    """
    if len(reputation_history) < 2:
        return 'insufficient_data'
    
    # Calculate trend over last few values
    recent_values = reputation_history[-3:] if len(reputation_history) >= 3 else reputation_history
    trend = recent_values[-1] - recent_values[0]
    
    if trend > 2.0:
        return 'improving'
    elif trend < -2.0:
        return 'declining'
    else:
        return 'stable'

def get_reputation_volatility(reputation_history: List[float]) -> float:
    """
    Calculate reputation volatility (standard deviation).
    
    Args:
        reputation_history: List of reputation scores
    
    Returns:
        Volatility value
    """
    if len(reputation_history) < 2:
        return 0.0
    
    return np.std(reputation_history)

def get_reputation_consistency(reputation_history: List[float]) -> float:
    """
    Calculate reputation consistency (inverse of volatility, normalized to 0-100).
    
    Args:
        reputation_history: List of reputation scores
    
    Returns:
        Consistency score (0-100, higher is more consistent)
    """
    if len(reputation_history) < 2:
        return 100.0
    
    volatility = np.std(reputation_history)
    # Convert to consistency (inverse relationship)
    consistency = max(0, 100 - volatility * 2)
    return consistency

# ===========================================================
# Example Usage and Testing
# ===========================================================

def test_reputation_functions():
    """
    Test function to demonstrate different reputation calculation methods.
    """
    print("Testing Reputation Calculation Functions")
    print("="*50)
    
    # Example log loss and reputation history
    log_loss = 0.3  # Example current log loss
    reputation_history = [85.2, 87.1, 82.3, 89.5, 86.8]  # Example history
    
    print(f"Current log loss: {log_loss}")
    print(f"Reputation history: {reputation_history}")
    print()
    
    # Test different methods
    methods = [
        ("Simple Mean", simple_mean, log_loss, reputation_history[-1]),
        ("Weighted Average (50/50)", weighted_average, log_loss, reputation_history[-1], 0.5),
        ("Weighted Average (70/30)", weighted_average, log_loss, reputation_history[-1], 0.7),
        ("Exponential Moving Average", exponential_moving_average, log_loss, reputation_history[-1], 0.3),
        ("History Weighted", reputation_history_weighted, log_loss, reputation_history),
        ("Momentum Based", momentum_based, log_loss, reputation_history),
        ("Decay Based", decay_based, log_loss, reputation_history),
        ("Percentile Based", percentile_based, log_loss, reputation_history),
        ("Adaptive Weighted", adaptive_weighted, log_loss, reputation_history)
    ]
    
    for method_name, method_func, *args in methods:
        try:
            result = method_func(*args)
            print(f"{method_name:25}: {result:.2f}")
        except Exception as e:
            print(f"{method_name:25}: Error - {e}")
    
    print()
    print("Reputation Analysis:")
    print(f"Trend: {get_reputation_trend(reputation_history)}")
    print(f"Volatility: {get_reputation_volatility(reputation_history):.2f}")
    print(f"Consistency: {get_reputation_consistency(reputation_history):.2f}")

if __name__ == "__main__":
    test_reputation_functions()
