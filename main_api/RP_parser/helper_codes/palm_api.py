import time

import vertexai
from vertexai.language_models import TextGenerationModel

vertexai.init(project="pyjamahr-ats-294009", location="us-central1")
parameters = {
    "temperature": 0.2,
    "max_output_tokens": 1024,
    "top_p": 0.8,
    "top_k": 40,
}
model = TextGenerationModel.from_pretrained("text-bison@001")


def generate_text(
    input_prompt, text_length=512, model=model, parameters=parameters
):
    """
    Generate text from input prompt
    Args:
        parameters: parameters for the model request
        model: model name
        input_prompt: input prompt
        text_length: length of a text to be generated

    Returns:
        response.text: generated text
    """
    parameters["max_output_tokens"] = text_length
    response = model.predict(input_prompt, **parameters)
    time.sleep(3)
    return response.text
