import os
import gradio as gr
from fastai.learner import load_learner

learn_inf = load_learner("model.pkl", cpu=True)

def classify_car(img):
    pred, idx, probs = learn_inf.predict(img)

    return {
        str(learn_inf.dls.vocab[i]): float(probs[i])
        for i in range(len(probs))
    }

app = gr.Interface(
    fn=classify_car,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Car Brand Classifier",
    description="Upload an image of BMW, Audi or Mercedes-Benz"
)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )