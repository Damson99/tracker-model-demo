import streamlit as st
import tempfile
import os
import shutil
from datetime import datetime
from ultralytics import YOLO

MAX_SIZE_10MB = 10 * 1024 * 1024


def main():
    st.title("Detekcja osób na wideo")

    uploaded_file = st.file_uploader(
        "Wgraj plik wideo (max. 10 MB)",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file is not None:
        file_size = uploaded_file.size
        if file_size > MAX_SIZE_10MB:
            st.error("BŁĄD: Rozmiar przekracza 10 MB. Proszę wgrać mniejszy plik.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_video_path = tmp.name

        st.info(f"Zapisano plik tymczasowy: {tmp_video_path}")

        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        with st.spinner("Wykonuję detekcję..."):
            model = YOLO("yolov10n_100_epochs.pt")

            results = model.predict(
                source=tmp_video_path,
                conf=0.7,
                save=True,
                project="runs",
                name=folder_name,
                exist_ok=True,
                classes=[0]
            )

        save_dir = results[0].save_dir

        output_video_path = None
        for fname in os.listdir(save_dir):
            if fname.lower().endswith(".mp4"):
                output_video_path = os.path.join(save_dir, fname)
                break

        if output_video_path and os.path.isfile(output_video_path):
            st.success("Detekcja zakończona! Oto przetworzone wideo:")
            st.video(output_video_path)

            with open(output_video_path, "rb") as f:
                btn_clicked = st.download_button(
                    label="Pobierz wynikowy plik wideo",
                    data=f,
                    file_name="wynik_detekcji.mp4",
                    mime="video/mp4"
                )

            if btn_clicked:
                try:
                    shutil.rmtree(save_dir)
                    st.success(f"Usunięto folder {save_dir} z serwera.")
                except Exception as e:
                    st.error(f"Nie udało się usunąć folderu. Błąd: {e}")
        else:
            st.warning("Nie znaleziono pliku wynikowego .mp4 w katalogu detekcji.")


if __name__ == "__main__":
    main()
