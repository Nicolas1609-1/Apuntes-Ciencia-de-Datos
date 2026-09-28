# Proyectos de Ciencia de Datos

Repositorio con apuntes, prácticas y demos desarrollados para aprender ciencia de datos y redes neuronales.

## Contenido

- [Apuntes-Ciencia-de-Datos](https://github.com/adiacla/Apuntes-Ciencia-de-Datos): cuadernos y materiales sobre perceptrones, redes neuronales densas, CNN y ejercicios aplicados.
- [Perceptron Trabajo en Clase](Perceptron%20Trabajo%20en%20Clase/Actividades/README.md): actividades, notebooks y archivos de práctica relacionados con redes neuronales.
- [convoluciones](https://github.com/adiacla/convoluciones): demo de convoluciones y filtros de imagen implementada con HTML y JavaScript.
- `salida_huevos_quebrados/`: archivos de salida generados por ejercicios de clasificación.

## Uso

Los notebooks se pueden abrir con Jupyter desde VS Code. Las dependencias pueden variar entre proyectos; consulta el archivo `requirements.txt` o el README de la carpeta correspondiente.

Para ejecutar la demo de convoluciones en Windows, abre una terminal en la carpeta `convoluciones` y ejecuta:

```powershell
py -m http.server 8000
```

Luego visita <http://localhost:8000/imagen.html> o <http://localhost:8000/camara.html>. La cámara puede requerir permisos del navegador.