# Импортируем библиотеки
import io
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_image(uploaded_file):
    """
    Функция загружает изображение из переданного файла и отображает его с помощью библиотеки streamlit.
    
    Parameters:
    - uploadedFile: BytesIO
        Файл, содержащий изображение.
    
    Returns:
    - img: Image
        Объект изображения Image из библиотеки PIL (Python Imaging Library), сконвертированный в формат RGB.
        Если uploadedFile равен None, возвращается значение None.
    """
    if uploaded_file is None:
        return None
    else:
        # Получаем данные изображения из файла
        image_data = uploaded_file.getvalue()

        # Отображаем изображение с помощью библиотеки streamlit
        st.image(image_data, width=200)

        # Отображаем название загруженного изображения
        st.write("Image Input : ", uploaded_file.name)
        
        # Возвращаем объект изображения Image из библиотеки PIL (Python Imaging Library), сконвертированный в формат RGB
        return Image.open(io.BytesIO(image_data)).convert('RGB')


# Инициализация процессора для обработки изображений и генерации текстовых описаний
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")


# Функция для загрузки модели для генерации текстовых описаний на основе изображений
# Результат функции кешируется для повторного использования
@st.cache
def load_model():
     """
    Функция для загрузки модели для генерации текстовых описаний изображений.

    Returns:
    - model: BlipForConditionalGeneration
        Загруженная модель для генерации текстовых описаний изображений.
    """
    return BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


# Загрузка модели
model = load_model()

st.title("Final Project - Image to Text")
st.write("""
         #### TEAM MEMBER
         - Рахарди Сандикха РИМ-130908
         - Мухин Виктор Александрович РИМ-130908
         - Шлёгин Лев Русланович РИМ-130908
         - Сидоркин Георгий Владимирович РИМ-130908
         """)

st.write("""#### Our Project""")

# Загрузка изображения
uploadedFile = st.file_uploader('Upload image here')
raw_image = load_image(uploadedFile)

# Обработка изображения и генерация текстового описания при нажатии кнопки Submit
result = st.button('Submit')

if result:
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=1000)
    text_output = processor.decode(out[0], skip_special_tokens=True)
    st.write("=========================================")
    st.write("Output : ", str(text_output).capitalize())
