import os
import logging

from utils import file_to_string, prompt_valid_check
from models import get_response_method, vllm_model_setup, get_answer, get_token_log


class DoctorAgent:
    def __init__(self, max_infs=15, top_k_diagnosis=5, backend_str="gpt4", backend_api_type="gpt_azure",  prompt_dir=None, prompt_file=None, patient_info=None, client_params=None, verbose=False) -> None:
        self.prompt_dir = prompt_dir
        self.prompt_file = prompt_file
        self.infs = 0  # number of inference calls to the doctor
        self.max_infs = max_infs  # maximum number of inference calls to the doctor
        self.top_k_diagnosis = top_k_diagnosis
        self.backend = backend_str  # language model backend for doctor agents
        self.backend_api_type = backend_api_type
        self.patient_info = patient_info if patient_info is not None else {}
        self.client_params = client_params if client_params is not None else {}
        self.verbose = verbose
        
        self.client = get_response_method(self.backend_api_type)
        self.model = vllm_model_setup(self.backend) if self.backend_api_type == "vllm" else self.backend

        if verbose:
            logging.info(f"Setting doctor agent with backend: {self.model} ({self.backend_api_type})")

        # Load prompt text file
        self.system_prompt_text = file_to_string(os.path.join(self.prompt_dir, self.prompt_file + ".txt"))

        # prepare initial conditions for LLM
        self.doctor_greet = "Hello, how can I help you?"
        self.reset()
        if verbose:
            logging.info(f" - Prompt file: {self.prompt_file}")
            logging.info(f" - System prompt: {self.system_prompt()}")

    def system_prompt(self) -> str:
        system_prompt = self.system_prompt_text.format(
            total_idx=self.max_infs, curr_idx=self.infs, remain_idx=self.max_infs - self.infs, top_k_diagnosis=self.top_k_diagnosis, **self.patient_info
        )
        prompt_valid_check(system_prompt, self.patient_info)
        return system_prompt

    def reset(self) -> None:
        system_message = {"role": "system", "content": self.system_prompt()}
        self.messages = [system_message]
        self.token_log = {"prompt_tokens": [], "completion_tokens": [], "total_tokens": [], "extra_info": {}}

    def log_token_usage(self, response) -> None:
        token_usage = get_token_log(response)
        self.token_log["prompt_tokens"].append(token_usage["prompt_tokens"])
        self.token_log["completion_tokens"].append(token_usage["completion_tokens"])
        self.token_log["total_tokens"].append(token_usage["total_tokens"])
        if "extra_info" in token_usage:
            for key, value in token_usage["extra_info"].items():
                if key not in self.token_log["extra_info"]:
                    self.token_log["extra_info"][key] = [value]
                else:
                    self.token_log["extra_info"][key].append(value)

    def inference(self, question) -> str:
        answer = str()
        if self.infs >= self.max_infs:
            return "Maximum inferences reached"
        self.infs += 1
        self.messages[0]["content"] = self.system_prompt()  # update current turns
        self.messages.append({"role": "user", "content": f"{question}"})

        response = self.client(self.messages, model=self.model, **self.client_params)
        answer = get_answer(response)
        self.log_token_usage(response)
        self.messages.append({"role": "assistant", "content": f"{answer}"})
        return answer

