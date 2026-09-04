# India Momentum Scanner — mobile/cloud version

This app is a recommendations-only NSE/BSE equity momentum scanner using Zerodha Kite Connect.

## Cloud deployment

Deploy on Streamlit Community Cloud from GitHub.

Required Streamlit Secrets:

```toml
KITE_API_KEY = "YOUR_KITE_API_KEY"
KITE_API_SECRET = "YOUR_KITE_API_SECRET"
KITE_REDIRECT_URL = "https://YOUR-APP-NAME.streamlit.app/"
```

Never commit the secrets file to GitHub.

### Zerodha redirect

After the Streamlit app is deployed, copy its exact HTTPS URL into your Kite Developer app's Redirect URL, then keep the same URL in `KITE_REDIRECT_URL` in Streamlit Secrets.

## System

Hard filters:
- NSE/BSE cash equities
- Price >= ₹50
- 20-day average traded value >= ₹10 crore
- Latest traded value >= ₹10 crore
- 30-day return >= 5% OR 50-day return >= 5%
- Price > 20DMA > 50DMA
- Rising 50DMA
- Outperform Nifty 500
- ATR% between 1% and 6%

Ranking:
- 30D momentum: 20
- 50D momentum: 15
- Relative strength: 20
- Trend: 20
- Liquidity: 10
- Volatility: 10
- Recent acceleration: 5

Output: top 10, or fewer if fewer qualify, or 0 if none qualify.

No order placement.
