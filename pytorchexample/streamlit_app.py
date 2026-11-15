
import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(
    page_title="Flower Federated Learning Dashboard",
    page_icon="🌸",
    layout="wide"
)

# Judul Dashboard
st.title("🌸 Federated Learning Dashboard")
st.markdown(
    "Simulasi Federated Learning menggunakan Flower + PyTorch (CIFAR-10)"
)

st.info("Arsitektur: 1 Server dan 3 Client")

# Lokasi file CSV
CSV_PATH = os.path.join("results", "metrics.csv")

# Tombol refresh
if st.button("Refresh Data"):
    st.rerun()

if os.path.exists(CSV_PATH):

    df = pd.read_csv(CSV_PATH)

    if not df.empty:

        # Ringkasan
        st.subheader("Ringkasan Federated Learning")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Jumlah Client",
            "3 Client"
        )

        col2.metric(
            "Total Round",
            int(df["round"].max())
        )

        col3.metric(
            "Akurasi Terakhir",
            f"{df['avg_accuracy'].iloc[-1] * 100:.2f}%"
        )

        st.divider()

        # Tabel hasil
        st.subheader("📊 Tabel Hasil Evaluasi")

        display_df = df.copy()
        display_df["avg_accuracy"] = (
            display_df["avg_accuracy"] * 100
        ).round(2)

        display_df = display_df.rename(columns={
            "round": "Round",
            "avg_loss": "Average Loss",
            "avg_accuracy": "Average Accuracy (%)"
        })

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
     )   

        st.divider()

        # Grafik
        st.subheader("📈 Grafik Perkembangan Model")

        col1, col2 = st.columns(2)

        with col1:
            fig_loss = px.line(
                df,
                x="round",
                y="avg_loss",
                markers=True,
                title="Average Loss per Round",
                labels={
                    "round": "Round",
                    "avg_loss": "Loss"
                }
            )

            st.plotly_chart(
                 fig_loss,
                 width="stretch"
        )

        with col2:
            fig_acc = px.line(
                df,
                x="round",
                y="avg_accuracy",
                markers=True,
                title="Average Accuracy per Round",
                labels={
                    "round": "Round",
                    "avg_accuracy": "Accuracy"
                }
            )

            fig_acc.update_yaxes(tickformat=".0%")

            st.plotly_chart(
                fig_acc,
                width="stretch"
            )

        st.divider()

        # Ringkasan akhir
        st.subheader("Hasil Akhir Pelatihan")

        first_acc = df["avg_accuracy"].iloc[0] * 100
        last_acc = df["avg_accuracy"].iloc[-1] * 100
        first_loss = df["avg_loss"].iloc[0]
        last_loss = df["avg_loss"].iloc[-1]

        col1, col2 = st.columns(2)

        col1.metric(
            "Peningkatan Akurasi",
            f"{last_acc - first_acc:.2f} poin persentase"
        )

        col2.metric(
            "Penurunan Loss",
            f"{first_loss - last_loss:.2f}"
        )

    else:
        st.warning("File metrics.csv masih kosong.")

else:
    st.warning("File results/metrics.csv belum ditemukan.")

    st.info(
        "Jalankan simulasi Federated Learning terlebih dahulu."
    )

    st.code(
        'flwr run . --stream --federation-config="num-supernodes=3"',
        language="powershell"
    )

# Footer
st.divider()
st.caption(
    "Federated Learning Dashboard | Flower + PyTorch | CIFAR-10"
)
