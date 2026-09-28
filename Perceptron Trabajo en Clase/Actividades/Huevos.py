import cv2
import numpy as np
import random
import os


# ============================================================
# FUNCIÓN PARA CREAR GRIETAS EN EL HUEVO
# ============================================================

def crear_huevo_quebrado(ruta_entrada, ruta_salida):
    """
    Toma una imagen de un huevo y genera una nueva versión
    con grietas visibles sobre la cáscara.
    """

    # --------------------------------------------------------
    # 1. CARGAR IMAGEN
    # --------------------------------------------------------

    imagen = cv2.imread(ruta_entrada, cv2.IMREAD_UNCHANGED)

    if imagen is None:
        print(f"ERROR: No se pudo abrir {ruta_entrada}")
        return False

    print(f"\nImagen cargada: {ruta_entrada}")

    alto, ancho = imagen.shape[:2]

    print(f"Tamaño: {ancho} x {alto}")

    # --------------------------------------------------------
    # 2. MANEJAR TRANSPARENCIA
    # --------------------------------------------------------

    if len(imagen.shape) == 3 and imagen.shape[2] == 4:

        b, g, r, alpha = cv2.split(imagen)

        imagen_bgr = cv2.merge([b, g, r])

    else:

        imagen_bgr = imagen.copy()

        alpha = np.full(
            (alto, ancho),
            255,
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # 3. CREAR MÁSCARA DEL HUEVO
    # --------------------------------------------------------

    # Solo dibujaremos las grietas donde la imagen
    # no sea transparente.

    mascara_huevo = alpha > 20

    # --------------------------------------------------------
    # 4. CREAR CAPA DE GRIETAS
    # --------------------------------------------------------

    capa_grietas = np.zeros(
        (alto, ancho),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # 5. CREAR ENTRE 2 Y 4 PUNTOS DE IMPACTO
    # --------------------------------------------------------

    cantidad_impactos = random.randint(2, 4)

    for impacto in range(cantidad_impactos):

        # Intentar encontrar un punto dentro del huevo
        for intento in range(100):

            cx = random.randint(
                int(ancho * 0.30),
                int(ancho * 0.70)
            )

            cy = random.randint(
                int(alto * 0.25),
                int(alto * 0.75)
            )

            if mascara_huevo[cy, cx]:
                break

        # ----------------------------------------------------
        # PEQUEÑO CENTRO DE IMPACTO
        # ----------------------------------------------------

        cv2.circle(
            capa_grietas,
            (cx, cy),
            random.randint(2, 4),
            255,
            -1
        )

        # ----------------------------------------------------
        # CREAR GRIETAS PRINCIPALES
        # ----------------------------------------------------

        numero_grietas = random.randint(6, 10)

        for _ in range(numero_grietas):

            angulo = random.uniform(
                0,
                2 * np.pi
            )

            longitud = random.randint(
                max(20, int(min(ancho, alto) * 0.08)),
                max(40, int(min(ancho, alto) * 0.25))
            )

            punto_actual = np.array(
                [float(cx), float(cy)]
            )

            puntos = [
                (int(cx), int(cy))
            ]

            # Cada grieta tendrá varios segmentos
            segmentos = random.randint(5, 10)

            longitud_segmento = longitud / segmentos

            angulo_actual = angulo

            for segmento in range(segmentos):

                # Cambiar ligeramente la dirección
                # para evitar líneas demasiado rectas.
                angulo_actual += random.uniform(
                    -0.35,
                    0.35
                )

                nuevo_x = (
                    punto_actual[0]
                    + np.cos(angulo_actual)
                    * longitud_segmento
                )

                nuevo_y = (
                    punto_actual[1]
                    + np.sin(angulo_actual)
                    * longitud_segmento
                )

                x = int(nuevo_x)
                y = int(nuevo_y)

                # Si sale de la imagen, detener grieta
                if (
                    x < 0
                    or x >= ancho
                    or y < 0
                    or y >= alto
                ):
                    break

                # Si sale del huevo, detener grieta
                if not mascara_huevo[y, x]:
                    break

                puntos.append((x, y))

                # ------------------------------------------------
                # RAMIFICACIÓN OCASIONAL
                # ------------------------------------------------

                if segmento > 1 and random.random() < 0.30:

                    angulo_rama = (
                        angulo_actual
                        + random.choice([-1, 1])
                        * random.uniform(0.4, 0.9)
                    )

                    longitud_rama = random.randint(
                        10,
                        max(12, int(longitud * 0.35))
                    )

                    rama_x = int(
                        x
                        + np.cos(angulo_rama)
                        * longitud_rama
                    )

                    rama_y = int(
                        y
                        + np.sin(angulo_rama)
                        * longitud_rama
                    )

                    if (
                        0 <= rama_x < ancho
                        and 0 <= rama_y < alto
                    ):

                        if mascara_huevo[rama_y, rama_x]:

                            cv2.line(
                                capa_grietas,
                                (x, y),
                                (rama_x, rama_y),
                                255,
                                1,
                                cv2.LINE_AA
                            )

                punto_actual = np.array(
                    [nuevo_x, nuevo_y]
                )

            # ------------------------------------------------
            # DIBUJAR GRIETA PRINCIPAL
            # ------------------------------------------------

            if len(puntos) >= 2:

                puntos_np = np.array(
                    puntos,
                    dtype=np.int32
                )

                cv2.polylines(
                    capa_grietas,
                    [puntos_np],
                    False,
                    255,
                    random.choice([1, 1, 2]),
                    cv2.LINE_AA
                )

    # --------------------------------------------------------
    # 6. EVITAR GRIETAS FUERA DEL HUEVO
    # --------------------------------------------------------

    capa_grietas[~mascara_huevo] = 0

    # --------------------------------------------------------
    # 7. CREAR SOMBRA ALREDEDOR DE LAS GRIETAS
    # --------------------------------------------------------

    sombra = cv2.GaussianBlur(
        capa_grietas,
        (5, 5),
        1
    )

    # Oscurecer ligeramente alrededor de la grieta
    factor_sombra = (
        sombra.astype(np.float32) / 255.0
    ) * 80

    resultado = imagen_bgr.astype(np.float32)

    for canal in range(3):

        resultado[:, :, canal] -= factor_sombra

    resultado = np.clip(
        resultado,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 8. DIBUJAR EL INTERIOR OSCURO DE LA GRIETA
    # --------------------------------------------------------

    mascara_linea = capa_grietas > 40

    # Color muy oscuro para simular profundidad
    resultado[mascara_linea] = [20, 20, 20]

    # --------------------------------------------------------
    # 9. CREAR UN PEQUEÑO BORDE CLARO
    # --------------------------------------------------------

    desplazada = np.roll(
        capa_grietas,
        1,
        axis=1
    )

    borde_claro = (
        (desplazada > 100)
        & (capa_grietas < 50)
        & mascara_huevo
    )

    # Mezclar borde claro con la cáscara
    resultado[borde_claro] = np.clip(
        resultado[borde_claro].astype(np.int16) + 45,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 10. RECUPERAR TRANSPARENCIA
    # --------------------------------------------------------

    b, g, r = cv2.split(resultado)

    resultado_final = cv2.merge(
        [b, g, r, alpha]
    )

    # --------------------------------------------------------
    # 11. GUARDAR IMAGEN MODIFICADA
    # --------------------------------------------------------

    correcto = cv2.imwrite(
        ruta_salida,
        resultado_final
    )

    if correcto:

        print("OK - Huevo modificado correctamente")
        print(f"Guardado en: {ruta_salida}")

        return True

    else:

        print("ERROR al guardar la imagen.")

        return False


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

print("\n==========================================")
print("      GENERADOR DE HUEVOS QUEBRADOS")
print("==========================================")


# ------------------------------------------------------------
# CARPETA DONDE ESTÁ HUEVOS.PY
# ------------------------------------------------------------

carpeta_actividades = os.path.dirname(
    os.path.abspath(__file__)
)


# ------------------------------------------------------------
# SUBIR A "PERCEPTRON TRABAJO EN CLASE"
# ------------------------------------------------------------

carpeta_principal = os.path.dirname(
    carpeta_actividades
)


# ------------------------------------------------------------
# CARPETA DE LAS IMÁGENES
# ------------------------------------------------------------

carpeta_imagenes = os.path.join(
    carpeta_principal,
    "Imagenes"
)


# ------------------------------------------------------------
# CARPETA DE RESULTADOS
# ------------------------------------------------------------

carpeta_salida = os.path.join(
    carpeta_actividades,
    "salida_huevos_quebrados"
)

os.makedirs(
    carpeta_salida,
    exist_ok=True
)


# ============================================================
# RUTAS DE LOS HUEVOS ORIGINALES
# ============================================================

huevo_0 = os.path.join(
    carpeta_imagenes,
    "imagen_0.png"
)

huevo_1 = os.path.join(
    carpeta_imagenes,
    "imagen_1.png"
)


# ============================================================
# RUTAS DE LOS HUEVOS MODIFICADOS
# ============================================================

huevo_0_quebrado = os.path.join(
    carpeta_salida,
    "huevo_0_quebrado.png"
)

huevo_1_quebrado = os.path.join(
    carpeta_salida,
    "huevo_1_quebrado.png"
)


# ============================================================
# COMPROBAR ARCHIVOS
# ============================================================

print("\nCarpeta de imágenes:")
print(carpeta_imagenes)

print("\nBuscando huevos...")


if os.path.exists(huevo_0):

    print("OK - imagen_0.png encontrada")

else:

    print("ERROR - imagen_0.png no encontrada")


if os.path.exists(huevo_1):

    print("OK - imagen_1.png encontrada")

else:

    print("ERROR - imagen_1.png no encontrada")


# ============================================================
# MODIFICAR LOS DOS HUEVOS
# ============================================================

print("\n==========================================")
print("MODIFICANDO HUEVO 0")
print("==========================================")

resultado_0 = crear_huevo_quebrado(
    huevo_0,
    huevo_0_quebrado
)


print("\n==========================================")
print("MODIFICANDO HUEVO 1")
print("==========================================")

resultado_1 = crear_huevo_quebrado(
    huevo_1,
    huevo_1_quebrado
)


# ============================================================
# RESULTADO FINAL
# ============================================================

print("\n==========================================")

if resultado_0 and resultado_1:

    print("       PROCESO TERMINADO CORRECTAMENTE")

else:

    print("       PROCESO TERMINADO CON ERRORES")

print("==========================================")

print("\nLos huevos originales NO fueron modificados.")

print("\nLos huevos quebrados están en:")

print(carpeta_salida)

print("\nArchivos creados:")

print("huevo_0_quebrado.png")
print("huevo_1_quebrado.png")