# proyecto-integrador-linux
Proyecto de reconocimiento facial en tiempo real con TensorFlow y OpenCV elaborado en la Pico pi IMX-8

Descripción de proyecto

El proyecto se enfoca en desarrollar una aplicación de reconocimiento facial y detección de emociones en tiempo real utilizando tecnologías de computer vision y deep learning. La aplicación emplea algoritmos de detección y seguimiento de rostros basados en redes neuronales convolucionales (CNN) utilizando TensorFlow y OpenCV. Además, se aprovecha la técnica de transfer learning para adaptar modelos preentrenados, permitiendo identificar y clasificar rostros desde flujos de video en tiempo real, brindando una experiencia precisa y eficiente en la detección de rostros en diferentes entornos.

TensorFlow en Computer Vision - Uso en Modelos de Deep Learning

TensorFlow fue empleado para desarrollar y entrenar modelos de aprendizaje profundo para dos tareas específicas de Computer Vision, reconocimiento facial con seguimiento en tiempo real, y a esta característica le añadimos la capacidad de reconocer emociones, entrenando otro modelo adicional. Utilizamos su API de alto nivel para construir redes neuronales convolucionales (CNN), fundamentales para la detección de objetos, seguimiento de objetos, reconocimiento facial, entre otros.

OpenCV para Procesamiento de Imágenes y Videos

OpenCV fue crucial para el procesamiento de imágenes y video en este proyecto. Fue utilizado para generar el dataset para entrenar el modelo de reconocimiento en tiempo real, y posteriormente para consumir los modelos para reconocer rostros y emociones, y dibujar los recuadros y etiquetas para la detección de los mismos.

Data Augmentation para el Entrenamiento del Modelo

En el contexto de este proyecto, empleamos data augmentation para crear variaciones artificiales en imágenes, como rotaciones, desplazamientos, cambio de iluminación, entre otros, con el fin de enriquecer nuestro dataset de entrenamiento y mejorar la capacidad de generalización del modelo.

Transfer Learning para Aprovechar Modelos Preentrenados

Utilizamos Transfer Learning para aprovechar modelos preentrenados, en este caso utilizamos VGG16 para el modelo de reconocimeinto facial, y optamos por usar ImageNet para el segundo modelo de reconocimiento de emociones. En este proyecto, ajustamos los modelos preentrenados utilizando técnicas de transfer learning para tareas específicas de Real-Time Image Recognition
