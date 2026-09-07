from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
	page_title="Modelo de IA para prendas",
	page_icon="👕",
	layout="centered",
)


MODEL_PATH = Path(__file__).resolve().parent / "fashion_mnist_model.keras"
CLASS_NAMES = [
	"Camiseta/top",
	"Pantalón",
	"Suéter",
	"Vestido",
	"Abrigo",
	"Sandalia",
	"Camisa",
	"Zapatilla",
	"Bolso",
	"Botín",
]


@st.cache_resource
def load_model():
	return tf.keras.models.load_model(MODEL_PATH, compile=False)


def prepare_image(canvas_image: np.ndarray) -> np.ndarray:
	canvas_image = np.asarray(canvas_image, dtype=np.uint8)
	if canvas_image.ndim == 3 and canvas_image.shape[-1] == 4:
		alpha = canvas_image[:, :, 3:4].astype(np.float32) / 255.0
		rgb = canvas_image[:, :, :3].astype(np.float32)
		canvas_image = np.clip(rgb * alpha, 0, 255).astype(np.uint8)
	grayscale = Image.fromarray(canvas_image).convert("L")
	resized = grayscale.resize((28, 28), Image.Resampling.LANCZOS)
	image = np.asarray(resized, dtype=np.float32)
	image = np.clip(image / 255.0, 0.0, 1.0)
	return image.reshape(28, 28)


st.title("Modelo de IA para prendas")
st.write(
	"Objetivo: dibuja una prenda en blanco y negro para que el modelo de "
	"inteligencia artificial identifique la categoría más probable."
)

st.subheader("Dibuja una prenda")
st.caption("Usa el dedo o el mouse dentro del lienzo.")

canvas_result = st_canvas(
	fill_color="#000000",
	stroke_width=18,
	stroke_color="#FFFFFF",
	background_color="#000000",
	width=280,
	height=280,
	drawing_mode="freedraw",
	display_toolbar=True,
	key="clothing_canvas",
)

predict = st.button("Clasificar prenda", type="primary", use_container_width=True)

if predict:
	if canvas_result.image_data is None or not np.any(prepare_image(canvas_result.image_data) > 0):
		st.warning("Dibuja una prenda antes de clasificarla.")
	else:
		try:
			model = load_model()
			image = prepare_image(canvas_result.image_data)
			prediction = model.predict(image[np.newaxis, ...], verbose=0)
			probabilities = np.asarray(prediction[0])
			predicted_index = int(np.argmax(probabilities))
			confidence = float(probabilities[predicted_index])

			st.success(
				f"Predicción: **{CLASS_NAMES[predicted_index]}** "
				f"({confidence:.1%} de confianza)"
			)
		except (OSError, ValueError, IndexError) as error:
			st.error(f"No fue posible cargar o ejecutar el modelo: {error}")

st.divider()
st.caption("Autor: Nicolas Abello SUarez")
st.caption("UNAB 2026")
