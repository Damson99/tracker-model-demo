import os
import shutil
import tempfile
from datetime import datetime

import streamlit as st
from ultralytics import YOLO


def main():
    st.title("Detekcja osób na zdjęciach")

    uploaded_file = st.file_uploader(
        "Wgraj obraz do detekcji (jpg, png, jpeg)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_image_path = tmp.name

        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        with st.spinner("Wykonuję detekcję..."):
            model = YOLO("yolov10n_100_epochs.pt")

            results = model.predict(
                source=tmp_image_path,
                conf=0.7,
                save=True,
                project="runs",
                name=folder_name,
                exist_ok=True,
                classes=[0]
            )

        save_dir = results[0].save_dir

        output_image_path = None
        for fname in os.listdir(save_dir):
            if fname.lower().endswith(".jpg"):
                output_image_path = os.path.join(save_dir, fname)
                break

        if output_image_path and os.path.isfile(output_image_path):
            st.success("Detekcja zakończona! Oto przetworzone zdjęcie:")
            st.image(output_image_path, use_column_width=True)

            with open(output_image_path, "rb") as f:
                btn_clicked = st.download_button(
                    label="Pobierz wynikowy obraz",
                    data=f,
                    file_name="wynik_detekcji.jpg",
                    mime="image/jpeg"
                )

            if btn_clicked:
                try:
                    shutil.rmtree(save_dir)
                    st.success(f"Usunięto folder z wynikami: {save_dir}")
                except Exception as e:
                    st.error(f"Nie udało się usunąć folderu. Błąd: {e}")
        else:
            st.warning("Nie znaleziono pliku wynikowego .jpg w katalogu detekcji.")


if __name__ == "__main__":
    main()
