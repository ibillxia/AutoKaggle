# Download every text corpus mentioned on the competition forum into extra/ds/<owner>__<name>/
# Only raw labelled passages matter for us; weight/head datasets are skipped on purpose.
import os
import subprocess
import sys

KAGGLE = r"C:\Users\deepg\.workbuddy\binaries\python\envs\default\Scripts\kaggle.exe"
ROOT = "extra/ds"

SETS = [
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
    "seyitkaangunes/ntl-extra-api-passages",
    "zbigniewkirchner/ntl-extra-gpt-passages",
    "willrice/name-that-llm-openrouter-harvest",
    "maniceet/name-that-llm-openrouter-openai-generations",
    "lucasboesen/ntl-free-proxy-generations",
    "abdullah1afify/ntl-generations",
    "dophuchung/ntl-proxy-generations",
    "prvsiyan/ntl-own-prompt-generations-v1",
    "michaelsemenoff/ntl-own-extra-v1",
    "haileycheng/name-that-llm-same-prompt-passages",
    "pjmathematician/ntl-train-prompts-reconstructed",
    "dhruvtibarewal/nlm-matched-clean",
    "ahmedberatozer/ntl-pool-v4",
]


def main():
    os.makedirs(ROOT, exist_ok=True)
    for ref in SETS:
        dst = os.path.join(ROOT, ref.replace("/", "__"))
        if os.path.isdir(dst) and os.listdir(dst):
            print(f"skip {ref}", flush=True)
            continue
        os.makedirs(dst, exist_ok=True)
        print(f"==> {ref}", flush=True)
        r = subprocess.run(
            [KAGGLE, "datasets", "download", "-d", ref, "-p", dst, "--unzip"],
            capture_output=True, text=True, timeout=1800,
        )
        tail = (r.stdout or "").strip().splitlines()[-2:]
        print("rc", r.returncode, "|", " / ".join(tail) or (r.stderr or "")[-300:], flush=True)


if __name__ == "__main__":
    main()
