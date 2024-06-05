from scalellm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(
    model="google/gemma-7b",
    devices="cuda",
    draft_model="google/gemma-2b",
    draft_devices="cuda",
    num_speculative_tokens=4,
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
