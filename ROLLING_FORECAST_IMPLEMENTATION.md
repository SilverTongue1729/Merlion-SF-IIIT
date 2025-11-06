# Rolling Forecast Implementation Details

## Exact Mechanism Explained

### TL;DR - You're Absolutely Right! ✅

1. **No retraining happens** during rolling forecast
2. **Fresh context is provided** to the model each time
3. **Zero-shot models (Moirai)**: Only use context, no training data dependency
4. **Traditional models (ARIMA)**: Use fitted parameters + context

---

## How It Works: Step by Step

### The Rolling Window Loop

```python
# From merlion/models/forecast/base.py, line 507-529
while start_idx + context_length + prediction_length <= total_length:
    # Step 1: Extract context window
    context_end_idx = start_idx + context_length
    context_ts = TimeSeries.from_pd(ts_df.iloc[start_idx:context_end_idx])
    
    # Step 2: Get timestamps to predict
    pred_timestamps = ts_df.index[context_end_idx:pred_end_idx].tolist()
    
    # Step 3: Call forecast() with NEW context
    forecast, err = self.forecast(
        time_stamps=pred_timestamps,
        time_series_prev=context_ts,  # ← KEY: Fresh context each iteration
        exog_data=window_exog_data,
        return_iqr=False,
    )
    
    # Step 4: Move window forward
    start_idx += prediction_stride
```

### Visual Example

```
Full Time Series: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                   ├─train─┤                                              

Rolling with context_length=4, prediction_length=2, stride=2:

Window 1:
  Context: [4, 5, 6, 7]      ──→ Predict: [8, 9]
                                    ↑
                             time_series_prev=context_ts
                             
Window 2:
  Context: [6, 7, 8, 9]      ──→ Predict: [10, 11]
                                    ↑
                             time_series_prev=NEW context_ts
                             
Window 3:
  Context: [8, 9, 10, 11]    ──→ Predict: [12, 13]
                                    ↑
                             time_series_prev=NEW context_ts
```

---

## Model Type Differences

### 1. Zero-Shot Models (Moirai, Moirai-MoE, Moirai2)

**During `train()`:**
```python
def _train(self, train_data: pd.DataFrame, train_config=None):
    # Load pretrained model (NO fitting on train_data)
    self._load_pretrained_model()
    
    # Just store metadata
    self.last_train_time = train_data.index[-1]
    self.target_name = train_data.columns[0]
    
    # Return empty - no actual training!
    return pd.DataFrame(), None
```

**During `forecast()` in rolling window:**
```python
def _forecast(self, time_stamps, time_series_prev, return_prev=False):
    # Use ONLY time_series_prev (context) - ignores train_data
    if time_series_prev is None:
        time_series_prev_ts = self.train_data  # Fallback only
    else:
        time_series_prev_ts = TimeSeries.from_pd(time_series_prev)
    
    # Prepare tensor from context ONLY
    past_target = self._prepare_input_tensor(time_series_prev_ts)
    
    # Generate forecast using pretrained model
    forecast = self.model.predict(past_target)
    
    # Model is stateless - doesn't remember previous calls
    return forecast_df, None
```

**Key Points:**
- ✅ Model parameters are frozen (pretrained)
- ✅ Each forecast uses ONLY the provided context
- ✅ No dependency on training data (it's just for metadata)
- ✅ Purely zero-shot: context in → prediction out

### 2. Traditional Statistical Models (ARIMA, SARIMA, Prophet)

**During `train()`:**
```python
def _train(self, train_data: pd.DataFrame, train_config=None):
    # Actually fit model parameters
    self.model = ARIMA(order=(p, d, q))
    self.model_fit = self.model.fit(train_data)  # ← Real training!
    
    # Store fitted parameters (coefficients, etc.)
    self.params = self.model_fit.params
    
    return predictions_on_train_data, stderr
```

**During `forecast()` in rolling window:**
```python
def _forecast(self, time_stamps, time_series_prev, return_prev=False):
    # Combine train_data with time_series_prev
    if time_series_prev is not None:
        full_context = pd.concat([self.train_data, time_series_prev])
    else:
        full_context = self.train_data
    
    # Use fitted parameters + new context
    forecast = self.model_fit.forecast(
        steps=len(time_stamps),
        exog=full_context  # Uses both fitted params AND new data
    )
    
    return forecast_df, stderr
```

**Key Points:**
- ✅ Model parameters fitted once during `train()`
- ✅ Parameters stay fixed during rolling forecast
- ✅ Each forecast uses: **fitted params + context**
- ✅ Context provides updated state for autoregressive terms

### 3. Deep Learning Models (DeepAR, Transformer)

**During `train()`:**
```python
def _train(self, train_data: pd.DataFrame, train_config=None):
    # Train neural network weights
    for epoch in range(num_epochs):
        for batch in train_data:
            loss = compute_loss(batch)
            loss.backward()
            optimizer.step()
    
    # Weights are updated and stored
    torch.save(self.model.state_dict(), 'weights.pt')
    
    return predictions, None
```

**During `forecast()` in rolling window:**
```python
def _forecast(self, time_stamps, time_series_prev, return_prev=False):
    # Use trained weights (frozen) with new context
    with torch.no_grad():  # No gradient computation
        context_tensor = prepare_tensor(time_series_prev)
        
        # Feed context through trained network
        forecast = self.model(context_tensor)
    
    return forecast_df, None
```

**Key Points:**
- ✅ Network weights trained once
- ✅ Weights frozen during rolling forecast
- ✅ Each forecast feeds new context through network
- ✅ Similar to zero-shot but with domain-specific training

---

## Comparison Table

| Aspect | Zero-Shot (Moirai) | Statistical (ARIMA) | Deep Learning (DeepAR) |
|--------|-------------------|---------------------|------------------------|
| **Training** | Load pretrained weights | Fit parameters | Train neural network |
| **What's stored** | Model weights (frozen) | Fitted coefficients | Network weights |
| **During rolling forecast** | Context → Model → Prediction | Params + Context → Prediction | Context → Network → Prediction |
| **Uses train_data?** | No (metadata only) | Yes (for state) | No (weights capture it) |
| **Context dependency** | 100% (only input) | Partial (with params) | 100% (only input) |
| **Stateless?** | Yes | No (has fitted state) | Yes (in inference mode) |

---

## Why No Retraining?

### Rolling Forecast (Our Implementation)
```python
# Train ONCE
model.train(train_data)  # Fit/load parameters

# Use multiple times with different contexts
for window in rolling_windows:
    forecast = model.forecast(
        time_stamps=future_times,
        time_series_prev=window_context,  # ← Only this changes
    )
```

**Benefits:**
- ⚡ Fast: No expensive retraining
- 🎯 Tests generalization: Same model, different contexts
- 🔄 Simulates real-world: Model deployed once, used many times

### Alternative: ForecastEvaluator (Retrain Each Step)
```python
# This is what ForecastEvaluator does
for window in windows:
    model.reset()  # ← Clear previous state
    model.train(window_train_data)  # ← RETRAIN every time
    forecast = model.forecast(future_times)
```

**When to use:**
- 📊 Simulating production with periodic retraining
- 🔄 Testing model drift adaptation
- 🎯 When model needs fresh parameter estimates

---

## Code Flow in Detail

### Moirai Rolling Forecast Flow

```
┌─────────────────────────────────────────────────┐
│ User calls: model.rolling_forecast()           │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ FOR each window:                                │
│   1. Extract context: ts[i:i+context_length]   │
│   2. Extract pred_times: ts[i+ctx:i+ctx+pred]  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ Call: self.forecast(                           │
│   time_stamps=pred_times,                      │
│   time_series_prev=context_ts  # ← NEW each    │
│ )                                               │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ Moirai._forecast():                            │
│   1. Convert context_ts → tensor               │
│   2. past_target = prepare_input(context_ts)   │
│   3. forecast = model.predict(past_target)     │
│   4. Return forecast (IGNORES train_data)      │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ Pretrained Moirai2 Model:                      │
│   - Encoder: process context                   │
│   - Decoder: generate predictions              │
│   - Return quantiles                           │
│   [Model is STATELESS - no memory of previous] │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ Back to rolling_forecast():                    │
│   - Collect forecast                           │
│   - Move window forward (start_idx += stride)  │
│   - Repeat until end                           │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ Concatenate all forecasts                      │
│ Remove duplicates (keep first)                 │
│ Return combined TimeSeries                     │
└─────────────────────────────────────────────────┘
```

---

## Summary

### Your Understanding is Correct! ✅

1. **"Are we giving the model new input every time?"**
   - ✅ YES! Each window provides fresh `time_series_prev` context

2. **"For models which need training, subsequent rolling context would not need training, just context is sufficient?"**
   - ✅ YES! Traditional models (ARIMA) use **fitted parameters + context**
   - No retraining, just new context combined with trained state

3. **"For zero-shot models like Moirai, there is no training, just context?"**
   - ✅ EXACTLY! Moirai's `train()` just loads pretrained weights
   - Each forecast uses **only the context** (time_series_prev)
   - 100% zero-shot: context → pretrained model → prediction

### The Beauty of This Design

- 🎯 **Universal**: Same API works for all model types
- ⚡ **Efficient**: No retraining overhead
- 🔄 **Realistic**: Simulates production deployment
- 🧩 **Flexible**: Easy to customize window parameters
