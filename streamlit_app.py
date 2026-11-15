import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Flower FL Dashboard", layout="wide")

st.title("🌸 Federated Learning Dashboard")
st.markdown("Simulasi Flower + PyTorch (CIFAR-10) | 1 Server + 3 Client")

CSV_PATH = "results/metrics.csv"

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)

    st.subheader("📊 Tabel Hasil per Ronde")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig_loss = px.line(
            df, x="round", y="avg_loss", markers=True,
            title="Average Loss per Round",
            labels={"round": "Ronde", "avg_loss": "Loss"}
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col2:
        fig_acc = px.line(
            df, x="round", y="avg_accuracy", markers=True,
            title="Average Accuracy per Round",
            labels={"round": "Ronde", "avg_accuracy": "Akurasi"}
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    st.subheader("📈 Ringkasan Akhir")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ronde Terakhir", int(df["round"].max()))
    c2.metric("Akurasi Terakhir", f"{df['avg_accuracy'].iloc[-1]:.4f}")
    c3.metric("Loss Terakhir", f"{df['avg_loss'].iloc[-1]:.4f}")

else:
    st.warning("⚠️ File `results/metrics.csv` belum ditemukan.")
    st.info("Jalankan simulasi Flower terlebih dahulu di terminal:")
    st.code(
        "flwr run . --run-config \"num-server-rounds=5 learning-rate=0.01\" --stream",
        language="bash"
    )