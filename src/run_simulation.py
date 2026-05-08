import os
import sys
import time
import json
import hydra
import random
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hydra.core.hydra_config import HydraConfig
from agent.doctor_agent import DoctorAgent
from agent.patient_agent import PatientAgent
from utils import file_to_string, save_to_dialogue, set_seed, detect_termination


class ScenarioLoaderMIMICIV:
    def __init__(self, data_dir, data_name="sample_info") -> None:
        with open(os.path.join(data_dir, f"{data_name}.json"), "r") as f:
            self.scenario_dict = json.load(f)
        self.num_scenarios = len(self.scenario_dict)
        logging.info(f"Load {self.num_scenarios} scenarios from {data_dir}")

    def sample_scenario(self):
        return self.scenario_dict[random.randint(0, self.num_scenarios - 1)]

    def get_scenario(self, id):
        if id is None:
            return self.sample_scenario()
        return self.scenario_dict[id]


def resolve_doctor_prompt_file(cfg):
    """Return (strategy, prompt_file_basename) for the doctor agent.

    Phase 2 ablation contains two arms:
      - passive_*  : doctor emits only a fixed cue every turn (4 surface forms)
      - active_*   : doctor actively interviews using a clinical protocol
                     (freestyle = OLD CARTS, sample = SAMPLE history)

    Comparing these two arms lets us see whether the metric correctly
    differentiates dialogues where the patient was actually probed from those
    where the patient simply leaked unprompted.
    """
    base = cfg.data.doctor_prompt_file
    strategy = (cfg.doctor_agent.get("strategy") or "passive_dots").lower()
    table = {
        "passive_dots": base + "_passive_dots",
        "passive_hmm":  base + "_passive_hmm",
        "passive_isee": base + "_passive_isee",
        "passive_oh":   base + "_passive_oh",
        "freestyle":    base + "_freestyle",
        "sample":       base + "_sample",
    }
    if strategy not in table:
        raise ValueError(f"Unknown doctor strategy: {strategy}; must be one of {list(table)}")
    return strategy, table[strategy]


def resolve_doctor_patient_info(scenario, strategy):
    """All passive variants only get basic patient info (no chart access)."""
    return {
        "gender": scenario["gender"],
        "age": scenario["age"],
        "arrival_transport": scenario["arrival_transport"],
    }


@hydra.main(config_path="./config", config_name="base", version_base="1.3")
def main(cfg):
    # Set random seed & create save directory
    set_seed(cfg.experiment.random_seed)
    cfg.save_dir = os.path.join(HydraConfig.get().run.dir, cfg.save_dir)
    os.makedirs(cfg.save_dir, exist_ok=True)
    if cfg.experiment.verbose:
        print(f"Save directory: {cfg.save_dir}")

    # Load scenarios
    scenario_loader = ScenarioLoaderMIMICIV(cfg.data_dir, cfg.data.data_file_name)
    logging.info(f"Load Datasets from {cfg.data_dir}, size: {scenario_loader.num_scenarios}")

    # Resolve prompt files based on disclosure mode + strategy
    patient_prompt_file = cfg.data.patient_prompt_file
    doctor_strategy, doctor_prompt_file = resolve_doctor_prompt_file(cfg)
    logging.info(f"Patient prompt_file={patient_prompt_file} (baseline PatientSim)")
    logging.info(f"Doctor strategy={doctor_strategy}, prompt_file={doctor_prompt_file}")
    logging.info(f"""Patient prompt template:\n\t{file_to_string(os.path.join(cfg.prompt_dir, patient_prompt_file + ".txt"))}""")
    logging.info(f"""Doctor prompt template:\n\t{file_to_string(os.path.join(cfg.prompt_dir, doctor_prompt_file + ".txt"))}""")

    # Pipeline for huggingface models
    num_scenarios = min(cfg.data.num_scenarios, scenario_loader.num_scenarios) if cfg.data.num_scenarios is not None else scenario_loader.num_scenarios
    scenario_id = cfg.data.get("scenario_id")
    scenario_ids = [scenario_id] if scenario_id is not None else list(range(0, num_scenarios))

    for _scenario_id in scenario_ids:
        # Initialize scenarios
        scenario = scenario_loader.get_scenario(id=_scenario_id)
        logging.info(f"\n=== Scenario {_scenario_id} / {num_scenarios} | hadm_id: {scenario['hadm_id']} ===")

        # Initialize agents
        patient_agent = PatientAgent(
            patient_profile=scenario,
            backend_str=cfg.patient_agent.backend,
            backend_api_type=cfg.patient_agent.api_type,
            prompt_dir=cfg.prompt_dir,
            prompt_file=patient_prompt_file,
            num_word_sample=cfg.data.num_word_sample,
            cefr_type=cfg.patient_agent.persona.cefr_type,
            personality_type=cfg.patient_agent.persona.personality_type,
            recall_level_type=cfg.patient_agent.persona.recall_level_option,
            dazed_level_type=cfg.patient_agent.persona.dazed_level_option,
            client_params=cfg.patient_agent.params,
            verbose=cfg.experiment.verbose,
        )
        doctor_patient_info = resolve_doctor_patient_info(scenario, doctor_strategy)
        doctor_agent = DoctorAgent(
            max_infs=cfg.doctor_agent.max_infs,
            top_k_diagnosis=cfg.doctor_agent.top_k_diagnosis,
            backend_str=cfg.doctor_agent.backend,
            backend_api_type=cfg.doctor_agent.api_type,
            prompt_dir=cfg.prompt_dir,
            prompt_file=doctor_prompt_file,
            patient_info=doctor_patient_info,
            client_params=cfg.doctor_agent.params,
            verbose=cfg.experiment.verbose,
        )

        # Start dialogue
        start_time = time.time()
        dialog_history = [{"role": "Doctor", "content": doctor_agent.doctor_greet}]
        doctor_agent.messages.append({"role": "assistant", "content": f"{doctor_agent.doctor_greet}"})
        logging.info(f"Doctor: {doctor_agent.doctor_greet}")

        crashed = False
        try:
            for inf_idx in range(cfg.experiment.total_inferences):
                # # Obtain response from patient
                patient_response = patient_agent.inference(dialog_history[-1]["content"])

                dialog_history.append({"role": "Patient", "content": patient_response})
                logging.info("Patient [{}%]: {}".format(int(((inf_idx + 1) / cfg.experiment.total_inferences) * 100), patient_response))

                # Obtain doctor dialogue
                if inf_idx == cfg.experiment.total_inferences - 1:
                    doctor_response = doctor_agent.inference(dialog_history[-1]["content"] + "\nThis is the final turn. Now, you must provide your top5 differential diagnosis.")
                else:
                    doctor_response = doctor_agent.inference(dialog_history[-1]["content"])
                dialog_history.append({"role": "Doctor", "content": doctor_response})
                logging.info("Doctor [{}%]: {}".format(int(((inf_idx + 1) / cfg.experiment.total_inferences) * 100), doctor_response))

                end_flag = detect_termination(doctor_response)
                if end_flag:
                    break

                # Prevent API timeouts
                time.sleep(1.0)
        except Exception as e:
            logging.error(f"Dialogue interrupted by exception: {e}; saving partial dialogue.")
            crashed = True

        end_time = time.time()
        dialog_info = {
            "hadm_id": scenario["hadm_id"],
            "patient_id": scenario.get("patient_id", scenario.get("hadm_id")),
            "doctor_engine_name": doctor_agent.backend,
            "patient_engine_name": patient_agent.backend,
            "doctor_api_type": doctor_agent.backend_api_type,
            "patient_api_type": patient_agent.backend_api_type,
            "doctor_strategy": doctor_strategy,
            "doctor_prompt_file": doctor_prompt_file,
            "patient_prompt_file": patient_prompt_file,
            "cefr_type": patient_agent.patient_profile["cefr_option"],
            "personality_type": patient_agent.patient_profile["personality_option"],
            "recall_level_type": patient_agent.patient_profile["recall_level_option"],
            "dazed_level_type":patient_agent.patient_profile["dazed_level_option"],
            "diagnosis": patient_agent.diagnosis,
            "dialog_history": dialog_history,
            "patient_token_log": patient_agent.token_log,
            "doctor_token_log": doctor_agent.token_log,
            "elapsed_time": end_time - start_time,
            "crashed": crashed,
        }
        save_to_dialogue(dialog_info, os.path.join(cfg.save_dir, "dialogue.jsonl"))


if __name__ == "__main__":
    main()
