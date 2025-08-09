import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

#データ読み込み
@st.cache_data
def load_data():
    df = pd.read_csv("Walmart_Sales.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    return df

df = load_data()

#サイドバー：フィルター設定
st.sidebar.header("フィルター設定")
store_list = sorted(df["Store"].unique())
selected_store = st.sidebar.selectbox("店舗を選択", store_list)

date_min = df["Date"].min()
date_max = df["Date"].max()
date_range = st.sidebar.date_input("日付範囲", [date_min, date_max])

forecast_steps = st.sidebar.slider("予測期間（週）", 12, 52, 26)
st.sidebar.markdown("### 特徴量の倍率調整")
temperature_factor = st.sidebar.slider("Temperature 倍率", 0.5, 2.0, 1.0, 0.05)
fuel_price_factor = st.sidebar.slider("Fuel_Price 倍率", 0.5, 2.0, 1.0, 0.05)
cpi_factor = st.sidebar.slider("CPI 倍率", 0.5, 2.0, 1.0, 0.05)
unemployment_factor = st.sidebar.slider("Unemployment 倍率", 0.5, 2.0, 1.0, 0.05)

#データ絞り込み
filtered_df = df[(df["Store"] == selected_store) &
                 (df["Date"] >= pd.to_datetime(date_range[0])) &
                 (df["Date"] <= pd.to_datetime(date_range[1]))].copy()
filtered_df.sort_values("Date", inplace=True)
filtered_df.set_index("Date", inplace=True)

#時系列可視化ダッシュボード
st.title("📊 Walmart 売上 時系列ダッシュボード + 予測")
st.write(f"選択店舗: {selected_store}")

# 売上トレンド
st.subheader("売上トレンド")
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=filtered_df, x=filtered_df.index, y="Weekly_Sales", marker="o", ax=ax)
ax.set_title("Weekly_sales_trends", fontsize=14)
st.pyplot(fig)

# 移動平均
window = st.slider("移動平均 (週)", 1, 12, 4)
filtered_df["Rolling_Sales"] = filtered_df["Weekly_Sales"].rolling(window=window).mean()
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=filtered_df, x=filtered_df.index, y="Rolling_Sales", color="orange", ax=ax2)
ax2.set_title(f"{window}weekly_moving_average", fontsize=14)
st.pyplot(fig2)

# 祝日と通常週の売上比較
st.subheader("祝日 vs 通常週")
holiday_avg = filtered_df.groupby("Holiday_Flag")["Weekly_Sales"].mean()
st.bar_chart(holiday_avg)

#SARIMAX 予測
st.subheader("📈 売上予測（SARIMAX + 外部特徴量）")

factors = {
    "Temperature": temperature_factor,
    "Fuel_Price": fuel_price_factor,
    "CPI": cpi_factor,
    "Unemployment": unemployment_factor,
}

y = filtered_df["Weekly_Sales"]
exog = filtered_df[["Temperature", "Fuel_Price", "CPI", "Unemployment"]].copy()
for col in exog.columns:
    exog[col] *= factors[col]

def generate_exog_features(forecast_index, exog_df):
    future_exog = []
    for date in forecast_index:
        week_52 = date - pd.Timedelta(weeks=52)
        week_104 = date - pd.Timedelta(weeks=104)
        exog_52 = exog_df.loc[week_52] if week_52 in exog_df.index else None
        exog_104 = exog_df.loc[week_104] if week_104 in exog_df.index else None

        if exog_52 is not None and exog_104 is not None:
            avg_exog = (exog_52 + exog_104) / 2
        elif exog_52 is not None:
            avg_exog = exog_52
        elif exog_104 is not None:
            avg_exog = exog_104
        else:
            avg_exog = exog_df.iloc[-1]

        future_exog.append(avg_exog.to_list())

    return pd.DataFrame(future_exog, columns=exog_df.columns, index=forecast_index)

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 52)

with st.spinner("SARIMAXモデルを学習中..."):
    model = SARIMAX(endog=y, exog=exog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(weeks=1), periods=forecast_steps, freq="W")
    future_exog = generate_exog_features(forecast_index, exog)
    for col in future_exog.columns:
        future_exog[col] *= factors[col]

    train_pred = results.get_prediction(start=0, end=len(y)-1, exog=exog)
    train_ci = train_pred.conf_int()
    train_index = y.index
    train_fitted = train_pred.predicted_mean

    pred = results.get_forecast(steps=forecast_steps, exog=future_exog)
    pred_ci = pred.conf_int()
    predicted_sales = pred.predicted_mean

#プロット
fig3, ax3 = plt.subplots(figsize=(14, 5))
ax3.plot(y.index, y, label="real_sales", color="green")
ax3.plot(train_index, train_fitted, label="past_prediction", color="blue")
ax3.fill_between(train_index, train_ci.iloc[:, 0], train_ci.iloc[:, 1], color="blue", alpha=0.2)
ax3.plot(forecast_index, predicted_sales, label="sales_prediction(SARIMAX)", color="red")
ax3.fill_between(forecast_index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="pink", alpha=0.3)
ax3.set_title(f"store {selected_store} sales_prediction_by_SARIMAX")
ax3.set_ylabel("Weekly Sales")
ax3.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)

#データ表示
with st.expander("🧾 データ確認"):
    st.dataframe(filtered_df.tail(10))
