# Quant-Random-Forest-Predictor

A supervised machine learning pipeline designed to predict directional momentum for Indian equities (NSE) using a tuned Random Forest Classifier and custom feature engineering.

## Model Architecture & Feature Engineering
Instead of relying purely on raw price data, this engine engineers specific financial features to provide the model with "market context":
* **Relative Strength vs. Benchmark:** Measures outperformance against a broader index.
* **Mean Reversion Metrics:** Identifies price extremes and trend extension relative to rolling averages.
* **Volume Surge:** Tracks institutional footprint relative to customized windows.
* **Rolling Volatility:** Standard deviation analysis to measure price expansion.
* **Momentum Oscillators:** Multi-period analysis to identify overbought/oversold regimes.

## Dynamic "Take-Profit" Target Logic
Standard ML models often fail in trading because they evaluate a target on a fixed future date. This engine simulates a real-world **Limit Order Execution**. It utilizes a rolling maximum window to evaluate if the asset hit a proprietary profit target *at any point* during the holding period, reflecting active strategy performance.

## Interactive Risk Management (Threshold Optimization)
Financial data is noisy. This script features an interactive **Threshold Optimization Scanner** that recalculates the Precision/Recall trade-off across confidence levels before execution. This allows for dynamic risk adjustment based on current market volatility.

## Performance Showcase (Confidential Asset A)
*Note: Specific tickers and proprietary parameters are kept confidential to protect the strategy's edge. The results below represent an out-of-sample backtest.*

### **1. Feature Importance Mapping**
The model identifies which indicators are driving the price action. In this test case, Volatility and Price Location were the primary lead indicators.

![Feature Importance Plot](redacted_final_graph.png)

### **2. Execution Dashboard**
Using an optimized confidence threshold, the model achieved high-conviction results:

![Performance Table](redacted.png)

* **Target:** Proprietary % Take-Profit within a fixed-day window.
* **Live Signal Confidence:** Dynamically calculated per asset.

🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install yfinance pandas numpy matplotlib scikit-learn`
3. **Configuration:** Ensure your `config.json` is populated with your desired parameters.
4. Run the script: `python quantitative_move_pred_with_random_forest.py`
5. Enter a ticker.
