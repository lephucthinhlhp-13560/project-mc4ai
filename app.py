import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import set_random_seed
from keras.backend import clear_session
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.title("Handwriting recognition")
option = st.sidebar.radio("",
    ["Home", "💻 Train"],
    label_visibility="collapsed"
)
@st.cache_data
def load_npz(file):
    data = np.load(file)
    if 'images' in data and 'labels' in data:
        return data['images'], data['labels']
    else:
        return None, None

if option == "💻 Train":

    with st.expander("Dataset"):
        
        data_info = False
        dataset = st.file_uploader("Upload dataset", type=["npz"])
        if dataset is not None:
            X, y = load_npz(dataset)

            data_info = True
            num_classes = len(np.unique(y))

        view_data = st.toggle("View dataset")
        if data_info:
            st.success(f'Dataset loaded. {y.shape} samples. Input shape {X.shape[1:]}. {num_classes} classes', icon="✅")        
        if view_data:
            with st.spinner("Wait for it...", show_time=True):
                fig, axs = plt.subplots(num_classes, 10)
                fig.set_figheight(num_classes * 2)
                fig.set_figwidth(20)

                for i in range(num_classes):
                    ids = np.where(y == i)[0]
                    for j in range(10):
                        target = np.random.choice(ids)
                        axs[i][j].axis('off')
                        axs[i][j].imshow(X[target].reshape(32, 32), cmap='gray')
                st.pyplot(fig)
    col3, col4= st.columns(2)
    with col3:
        epochs = int(st.text_input("Epochs", "10"))
    with col4:
        test_size = float(st.text_input("Test size", "0.20"))
    if st.button("Train", use_container_width=True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        y_train_ohe = to_categorical(y_train, num_classes=num_classes)
        y_test_ohe = to_categorical(y_test, num_classes=num_classes)

        clear_session()
        set_random_seed(42)

        X_train = X_train.reshape(len(X_train), -1)
        X_test = X_test.reshape(len(X_test), -1)

        model = Sequential()
        model.add(Input(shape=X_train.shape[1:]))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        history = model.fit(X_train, y_train_ohe, epochs = epochs, verbose=0)
        loss, accuracy = model.evaluate(X_test, y_test_ohe)

        train_accuracy = history.history['accuracy'][-1] * 100
        test_accuracy = history.history['val_accuracy'][-1] * 100
        st.success(f'Train accuracy: {train_accuracy:.2f}%. Test accuracy: {test_accuracy:.2f}%', icon="✅")

        plt.figure(figsize=(8,4))

        plt.xlabel('Epochs')
        plt.plot(history.history['loss'], label = "Loss", color="blue")
        plt.plot(history.history['val_loss'], label = "Val Loss", color="cyan")
        plt.plot(history.history['accuracy'], label = "Accuracy", color="red")
        plt.plot(history.history['val_accuracy'], label = "Val Accuracy", color="pink")
        plt.legend()

        plt.show()

if option == "Home":
    st.write("Draw a character below:")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        update_streamlit=True,
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas"
        )

    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        img = img.convert('L')
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0

        X_pred = model.predict(img_array)
        y_pred = np.argmax(X_pred, axis=1)[0]
        confidence = X_pred[0][y_pred] * 100
    
        st.write(f"Predicted character: **{chr(65 + y_pred)}** (Label: {y_pred}). Confidence: {confidence:.2f}%")    