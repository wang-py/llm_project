import sys
import subprocess

model_dir = "/home/pyw-home/.llama/checkpoints/"
torch_run_bin = "/home/pyw-home/miniconda3/envs/LLM/bin/torchrun"
script = "example_chat_completion.py"

model = {"8B": model_dir + "Llama3.1-8B/", "1B": model_dir + "Llama3.2-1B/",
         "8B-Instruct": model_dir + "Llama3.1-8B-Instruct/",
         "1B-Instruct": model_dir + "Llama3.2-1B-Instruct/"}

if __name__ == "__main__":
    model_type = sys.argv[1]
    torch_run_args = (torch_run_bin, script, model[model_type],
                      model[model_type] + "tokenizer.model",
                      "--max-seq-len", "512",
                      "--max_batch_size", "4")
    popen = subprocess.Popen(torch_run_args, stdout=sys.stdout)
    popen.wait()
