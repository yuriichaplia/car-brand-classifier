import os
import gradio as gr
from fastai.learner import load_learner

learn_inf = None

def get_learner():
    global learn_inf
    if learn_inf is None:
        learn_inf = load_learner("model.pkl", cpu=True)
    return learn_inf

def classify_car(img):
    learn = get_learner()
    pred, idx, probs = learn.predict(img)
    return {str(learn.dls.vocab[i]): float(probs[i]) for i in range(len(probs))}

app = gr.Interface(
    fn=classify_car,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Car Brand Classifier",
    description="Upload an image of BMW, Audi or Mercedes-Benz",
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
