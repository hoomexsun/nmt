from src.nmt_toolkit.split_corpus import main as split_corpus_main
from src.nmt_toolkit.main import main as toolkit_main

if __name__ == "__main__":
    print("[INFO] Running pre-split of corpora (train/val/test) from run.py...")
    split_corpus_main()
    toolkit_main()
