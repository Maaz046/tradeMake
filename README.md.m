🔺 1. Labeling Logic (HIGH PRIORITY)
	•	Issue: You’re using a basic future-return threshold (likely hardcoded like >1%, <-1%) without accounting for volatility or market regime.
	•	Impact: Labels are noisy, model learns non-generalizable rules.
	•	Fix: Use ATR, volatility-based thresholds, or cross-period returns. Possibly integrate a trend-based signal like breakout + hold time.
	•	Effort: Moderate. Labeling function change + retraining.

⸻

🔺 2. Hyperparameter Tuning (HIGH PRIORITY)
	•	Issue: You’re using static XGBClassifier params.
	•	Impact: Poor generalization. Could be underfitting or overfitting without validation control.
	•	Fix: Grid search or randomized search via optuna, GridSearchCV, or even simple manual sweeps.
	•	Effort: Medium. Wrap model in GridSearchCV.

⸻

🔺 3. Class Imbalance (QUICK WIN)
	•	Issue: Label distribution is skewed: Counter({2: 79, 0: 70, 1: 50})
	•	Impact: Class 1 (likely HOLD) is under-represented, model becomes biased toward others.
	•	Fix: Use scale_pos_weight, synthetic oversampling (SMOTE), or class weights.
	•	Effort: Very low — pass class weights to XGBoost or resample training data.

    ⚠️ Problems with Current Logic
	1.	Arbitrary thresholds (±1%):
        •	These fixed values don’t adapt to volatility.
        •	May be too small or too large depending on the asset.
	2.	Labeling based on 1-day forward return:
        •	Not aligned with realistic holding durations (e.g., trades might hold for 3–10 days).
        •	It increases noise sensitivity.
	3.	No incorporation of drawdowns/risk:
	    •	Even if an asset gains 1% in a day, it might’ve dropped 5% intraday.
	4.	No volume or momentum context:
	    •	The signal is purely return-based, ignoring other indicators.

⸻

🔺 4. Signal Filtering (QUICK WIN)
	•	Issue: Signals likely fire every time classifier says “buy,” even in choppy/noisy zones.
	•	Impact: False signals → unnecessary trades → drawdown.
	•	Fix: Add confidence filter (probability threshold), or require minimum time between trades, or price proximity filter.
	•	Effort: Very low — can do inside signals.py.
        1. Probability threshold        Filter predictions unless softmax prob > X
        2. Cooldown filter              Don’t trade again within N days
        3. Directional proximity        Only act if price is within X% of recent extrema
        4. Volatility filter            Don’t trade if ATR is above average


⸻

🔻 5. Feature Engineering (LATER)
	•	Issue: You use ~25 features. Some might be noisy or redundant.
	•	Impact: May dilute signal-to-noise ratio.
	•	Fix: Use PCA, correlation filters, or feature creation (e.g., “EMA crossover width”, “price/volume divergence”, etc.).
	•	Effort: Medium-high and requires experimentation.

⸻

🔻 6. Ensemble Modeling (LATER)
	•	Issue: You rely on 1 model.
	•	Impact: Low robustness. Market regime shifts can ruin it.
	•	Fix: Blend short/long-term models, or stack Random Forest + XGB, or use voting.
	•	Effort: High — requires multiple pipelines.