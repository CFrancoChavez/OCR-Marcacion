# OCR Marcación - Sistema de Reconocimiento Óptico de Caracteres

Sistema avanzado de OCR (Optical Character Recognition) diseñado para extraer texto de etiquetas y documentos con múltiples estrategias de procesamiento de imagen para maximizar la precisión.

## Características

- **Múltiples Estrategias de OCR**: Combina EasyOCR y Tesseract con diferentes técnicas de preprocesamiento
- **Procesamiento Inteligente**: Aplica filtros adaptativos, CLAHE, y reducción de ruido
- **Corrección Automática**: Sistema de corrección inteligente basado en patrones conocidos
- **Interfaz Web Intuitiva**: Interfaz drag-and-drop para subir imágenes
- **Comparación de Resultados**: Muestra múltiples resultados con niveles de confianza
- **Visualización de Procesamiento**: Muestra las imágenes procesadas para cada estrategia

## Tecnologías

**Backend:**
- Python 3.x
- Flask (Framework web)
- OpenCV (Procesamiento de imágenes)
- Tesseract OCR
- EasyOCR
- PIL/Pillow
- NumPy

**Frontend:**
- HTML5
- CSS3
- JavaScript (Vanilla)

## Instalación

### Requisitos Previos

- Python 3.7 o superior
- Tesseract OCR instalado en el sistema

**Instalar Tesseract:**

**Windows:**
\`\`\`bash
# Descargar e instalar desde: https://github.com/UB-Mannheim/tesseract/wiki
\`\`\`

**macOS:**
\`\`\`bash
brew install tesseract
\`\`\`

**Linux (Ubuntu/Debian):**
\`\`\`bash
sudo apt-get update
sudo apt-get install tesseract-ocr
\`\`\`

### Instalación del Proyecto

1. **Clonar el repositorio:**
\`\`\`bash
git clone https://github.com/CFrancoChavez/OCR-Marcacion.git
cd OCR-Marcacion
\`\`\`

2. **Crear entorno virtual:**
\`\`\`bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
\`\`\`

3. **Instalar dependencias:**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. **Ejecutar la aplicación:**
\`\`\`bash
python app.py
\`\`\`

5. **Abrir en el navegador:**
\`\`\`
http://localhost:5000
\`\`\`

## Uso

1. **Subir Imagen**: Arrastra una imagen o haz clic para seleccionar un archivo
2. **Procesamiento Automático**: El sistema aplica múltiples estrategias de OCR
3. **Revisar Resultados**: Compara los resultados de diferentes estrategias
4. **Mejor Resultado**: El sistema destaca automáticamente el resultado con mayor confianza

## Estrategias de Procesamiento

El sistema implementa múltiples estrategias de preprocesamiento:

- **Original Sin Procesar**: Imagen en escala de grises sin modificaciones
- **Procesamiento Suave**: Filtro bilateral y ajuste de contraste ligero
- **Procesamiento Mínimo**: Solo escala de grises con ajuste mínimo
- **CLAHE Suave**: Mejora de contraste local adaptativo
- **Binarización Adaptativa**: Umbralización adaptativa de Gaussian
- **Binarización Otsu**: Umbralización automática de Otsu
- **Morfología Suave**: Operaciones morfológicas para limpieza de ruido

## Estructura del Proyecto

\`\`\`
OCR-Marcacion/
├── app.py                 # Aplicación Flask principal
├── requirements.txt       # Dependencias de Python
├── templates/
│   └── index.html        # Interfaz web
├── static/               # Archivos estáticos (CSS, JS, imágenes)
└── uploads/              # Carpeta temporal para imágenes subidas
\`\`\`

## Corrección Inteligente

El sistema incluye un módulo de corrección inteligente que:
- Detecta patrones comunes de errores de OCR
- Aplica correcciones basadas en contexto
- Utiliza similitud de secuencias para validar resultados
- Mantiene un diccionario de correcciones conocidas

## Casos de Uso

- Extracción de texto de etiquetas de productos
- Lectura de fechas de elaboración y vencimiento
- Procesamiento de códigos de lote
- Digitalización de documentos con texto impreso
- Automatización de entrada de datos

## Mejoras Futuras

- [ ] Soporte para procesamiento por lotes
- [ ] API REST para integración con otros sistemas
- [ ] Exportación de resultados a CSV/JSON
- [ ] Entrenamiento de modelos personalizados
- [ ] Soporte para múltiples idiomas
- [ ] Integración con bases de datos

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Contacto

**Franco Chávez**
- Email: cfrancochavezdev@gmail.com
- LinkedIn: [Franco Chávez](https://www.linkedin.com/in/franco-chavez-548b0a56/)
- GitHub: [@CFrancoChavez](https://github.com/CFrancoChavez)
- Portfolio: [My Full Stack Portfolio](https://github.com/CFrancoChavez/My-FullStack-Portfolio)

## Licencia

Este proyecto está disponible como código abierto bajo la licencia MIT.

---

Desarrollado con Python y Flask
